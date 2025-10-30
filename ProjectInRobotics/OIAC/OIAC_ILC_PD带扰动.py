import mujoco, mujoco.viewer
import time, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import interpolate


# 配置

MODEL_PATH = "mergedCopy.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

joint_name = "el_x"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
qpos_adr = model.jnt_qposadr[joint_id]
dof_adr  = model.jnt_dofadr[joint_id]

TORQUE_MIN = -4.1
TORQUE_MAX = 4.1

AMP_DEG = 60.0
FREQ = 0.5
def target_angle_rad(t):
    return math.radians(AMP_DEG) * abs(math.sin(2 * math.pi * FREQ * t))

alpha = 0.2   # torque low-pass filter
filtered_torque = 0.0
act_idx = 0   # <-- 如果actuator index 不是 0，改这里

# initialize qpos
data.qpos[qpos_adr] = math.radians(10.0)
mujoco.mj_forward(model, data)


# 扰动配置

class DisturbanceConfig:
    def __init__(self):
        self.enabled = True  # 是否启用扰动
        self.last_impulse_time = 0
        self.impulse_interval = 0.1  # 脉冲间隔(秒)
        self.current_impulse = 0


# OIAC: 改进版本在线阻抗自适应（增强了初始响应  之前响应过慢）

class ada_imp_con:
    def __init__(self, dof):
        self.DOF = dof
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        self.q = np.zeros((self.DOF,1))
        self.q_d = np.zeros((self.DOF,1))
        self.dq = np.zeros((self.DOF,1))
        self.dq_d = np.zeros((self.DOF,1))
        # 改进的自适应参数（增强初始响应）
        self.a = 0.2  # 2 增大a值增强初始响应
        self.b = 1  # 1 减小b值减缓衰减
        self.k = 0.05  #0.9
        # 设置合理的最小增益保证基础控制力
        self.K_MIN = 0.01   # 最小刚度，避免初始阶段控制力不足
        self.K_MAX = 100.0
        self.B_MIN = 0.5   # 最小阻尼
        self.B_MAX = 20.0

    def update_impedance(self, q, q_d, dq, dq_d):
        q = np.asarray(q).reshape((self.DOF,1))
        q_d = np.asarray(q_d).reshape((self.DOF,1))
        dq = np.asarray(dq).reshape((self.DOF,1))
        dq_d = np.asarray(dq_d).reshape((self.DOF,1))
        self.q, self.q_d, self.dq, self.dq_d = q, q_d, dq, dq_d

        pos_err = self.q - self.q_d
        vel_err = self.dq - self.dq_d
        track_err = self.k * vel_err + pos_err
        nrm = la.norm(track_err)
        
        # 改进的自适应因子计算：使用指数衰减，初始响应更强
        # 当误差很小时保持较强控制，误差大时适当衰减
        ad_factor = self.a * math.exp(-self.b * nrm)
        
        # 确保最小自适应因子，避免控制力过弱
        min_ad_factor = 0.3
        ad_factor = max(ad_factor, min_ad_factor)

        k_mat = ad_factor * (track_err @ pos_err.T)
        b_mat = ad_factor * (track_err @ vel_err.T)

        k_val = float(k_mat[0,0]) if k_mat.size>0 else 0.0
        b_val = float(b_mat[0,0]) if b_mat.size>0 else 0.0

        # 限制在合理范围内
        k_val = np.clip(k_val, self.K_MIN, self.K_MAX)
        b_val = np.clip(b_val, self.B_MIN, self.B_MAX)

        self.k_mat = np.array([[k_val]])
        self.b_mat = np.array([[b_val]])
        return self.k_mat, self.b_mat


# Iterative Learning Control (trial-wise), conservative & robust

class IterativeLearningControl:
    def __init__(self, max_trials=50, error_threshold_rad=math.radians(2),
                 learning_rate=0.2, target_length=1000, ff_clip=5.0):
        self.max_trials = max_trials
        self.error_threshold = error_threshold_rad
        self.learning_rate = learning_rate
        self.target_length = target_length
        self.ff_clip = ff_clip
        self.current_trial = 0
        self.learned_feedforward = []  # list of aligned numpy arrays
        self.average_errors = []
        self.trial_data = []
        self.reference_time = None

    def should_stop_learning(self, current_error_rad):
        self.current_trial += 1
        self.average_errors.append(current_error_rad)
        if current_error_rad <= self.error_threshold:
            print(f"停止学习! 试验 {self.current_trial}: 平均误差 = {math.degrees(current_error_rad):.2f}° ≤ {math.degrees(self.error_threshold):.0f}°")
            return True
        if self.current_trial >= self.max_trials:
            print(f"停止学习! 达到最大试验次数 {self.max_trials}")
            return True
        return False

    def align_trajectories(self, time_trajectory, data_trajectory):
        # align any-length trajectory to reference_time
        if len(time_trajectory) < 2:
            if self.reference_time is None:
                self.reference_time = np.linspace(0, 10.0, self.target_length)
            return self.reference_time, np.zeros(self.target_length)

        if self.reference_time is None:
            max_time = min(10.0, max(time_trajectory))
            self.reference_time = np.linspace(0, max_time, self.target_length)

        interp = interpolate.interp1d(time_trajectory, data_trajectory, kind='linear',
                                      bounds_error=False,
                                      fill_value=(data_trajectory[0], data_trajectory[-1]))
        aligned = interp(self.reference_time)
        # guard NaN/inf
        if np.any(np.isnan(aligned)) or np.any(np.isinf(aligned)):
            aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
        return self.reference_time, aligned

    def update_feedforward(self, error_trajectory, time_trajectory):
        aligned_time, aligned_err = self.align_trajectories(time_trajectory, error_trajectory)
        if not self.learned_feedforward:
            prev = np.zeros_like(aligned_err)
        else:
            prev = self.learned_feedforward[-1]

        # conservative update
        new_ff = prev + self.learning_rate * aligned_err
        # guard NaN/inf and clip
        new_ff = np.nan_to_num(new_ff, nan=0.0, posinf=self.ff_clip, neginf=-self.ff_clip)
        new_ff = np.clip(new_ff, -self.ff_clip, self.ff_clip)
        # simple smoothing to reduce high-frequency noise
        if new_ff.size >= 5:
            new_ff = np.convolve(new_ff, np.ones(5)/5.0, mode='same')
        self.learned_feedforward.append(new_ff)
        return new_ff, aligned_time

    def get_feedforward_torque(self, t, trial_idx):
        if self.reference_time is None:
            return 0.0
        if trial_idx < 0 or trial_idx >= len(self.learned_feedforward):
            return 0.0
        idx = np.argmin(np.abs(self.reference_time - t))
        ff_traj = self.learned_feedforward[trial_idx]
        if idx < len(ff_traj):
            val = float(ff_traj[idx])
            if not np.isfinite(val) or abs(val) > 100.0:  # safety guard
                return 0.0
            return val
        return 0.0

    def record_trial_data(self, time_log, q_log, q_des_log, torque_log, error_log, aligned_time, aligned_error):
        self.trial_data.append({
            'time': np.array(time_log),
            'q': np.array(q_log),
            'q_des': np.array(q_des_log),
            'torque': np.array(torque_log),
            'error': np.array(error_log),
            'aligned_time': np.array(aligned_time),
            'aligned_error': np.array(aligned_error)
        })

# 扰动
def apply_disturbance(t, data, dof_adr, dist_config):
    
    if not dist_config.enabled:
        data.qfrc_applied[dof_adr] = 0.0
        return 0.0
    
    disturbance = 0.0
    
    # 第1段：3-4秒，随机脉冲扰动
    if 3.0 < t < 4.0:
       
        if t - dist_config.last_impulse_time > dist_config.impulse_interval:
            dist_config.current_impulse = np.random.uniform(-2.0, 2.0)
            dist_config.last_impulse_time = t
        disturbance = dist_config.current_impulse
    
    # 第2段：6-7秒，周期性扰动
    elif 6.0 < t < 7.0:
        disturbance = 1.5 * math.sin(4 * math.pi * t)  # 2Hz正弦
    
    # 第3段：8-9秒，恒定扰动
    elif 8.0 < t < 9.0:
        disturbance = 1.8  # 恒定正向扰动
    
    else:
        disturbance = 0.0
        data.qfrc_applied[dof_adr] = 0.0
    
    
    data.qfrc_applied[dof_adr] = disturbance
    return disturbance


# instantiate controllers

ada_imp = ada_imp_con(dof=1)
dist_config = DisturbanceConfig()

#优化的控制参数
gain_ff = 3.0   # 2  1 0.5适当降低前馈增益，让OIAC和PD发挥更多作用
ilc = IterativeLearningControl(max_trials=5, error_threshold_rad=math.radians(2),
                               learning_rate=0.35, target_length=1000, ff_clip=4.0)

# 小量PD：提供基础稳定性（保证启动）
Kp_base = 15.0  # 适中的P增益，保证启动但不主导控制
Kd_base = 0.8   # 0.8 适中的D增益，提供阻尼


# main trial loop

all_avg_errors = []
MAX_TRIALS = ilc.max_trials

for trial in range(MAX_TRIALS):
    print(f"\n=== Trial {trial+1}/{MAX_TRIALS} ===")
    # reset state
    data.qpos[qpos_adr] = math.radians(20.0)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)

    time_log = []
    q_log = []
    q_des_log = []
    torque_log = []
    error_log = []
    disturbance_log = []  # 记录扰动

    filtered_torque = 0.0
    t0 = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            if t > 10.0:
                break

            q = float(data.qpos[qpos_adr])
            qdot = float(data.qvel[dof_adr])
            q_des = float(target_angle_rad(t))
            dq_des = 2 * math.pi * FREQ * math.radians(AMP_DEG) * math.cos(2 * math.pi * FREQ * t)

            error = q_des - q

            # feedforward from previous trial (safe read)
            if trial == 0:
                tau_ff = 0.0
            else:
                tau_ff = ilc.get_feedforward_torque(t, trial-1)
                if not np.isfinite(tau_ff) or abs(tau_ff) > 50.0:
                    # safety override: ignore bad ff
                    print(f"Warning: unsafe tau_ff={tau_ff:.4f} at t={t:.3f}, using 0 instead.")
                    tau_ff = 0.0

            # OIAC feedback (改进后响应更快)
            q_vec = np.array([[q]])
            qd_vec = np.array([[q_des]])
            dq_vec = np.array([[qdot]])
            dqd_vec = np.array([[dq_des]])

            K_mat, B_mat = ada_imp.update_impedance(q_vec, qd_vec, dq_vec, dqd_vec)
            pos_error_vec = (qd_vec - q_vec)
            vel_error_vec = (dqd_vec - dq_vec)
            tau_fb_vec = (K_mat @ pos_error_vec) + (B_mat @ vel_error_vec)
            tau_fb = float(tau_fb_vec.item())

            # 小量PD基线：主要提供启动保障和基础稳定性
            pd_torque = Kp_base * error + Kd_base * (dq_des - qdot)

            # 组合扭矩：三者协同工作
            torque_raw = (gain_ff * tau_ff) + tau_fb + pd_torque

            # 应用扰动
            current_disturbance = apply_disturbance(t, data, dof_adr, dist_config)

            # 信息（显示各分量贡献和扰动）
            if int(t * 1000) % 500 == 0 and int((t - 0.001) * 1000) % 500 != 0:
                print(f"DIAG t={t:.3f} tau_ff={tau_ff:.4f} tau_fb={tau_fb:.4f} pd={pd_torque:.4f} dist={current_disturbance:.4f} raw={torque_raw:.4f}")
                print(f"     当前阻抗: Kp={float(K_mat[0,0]):.2f}, Kd={float(B_mat[0,0]):.2f}")

            # saturate, filter, apply
            torque_clipped = max(TORQUE_MIN, min(torque_raw, TORQUE_MAX))
            filtered_torque = alpha * torque_clipped + (1.0 - alpha) * filtered_torque

            try:
                data.ctrl[act_idx] = float(filtered_torque)
            except Exception as e:
                print("写 data.ctrl 失败:", e)

            # record logs
            time_log.append(t)
            q_log.append(q)
            q_des_log.append(q_des)
            torque_log.append(filtered_torque)
            error_log.append(error)
            disturbance_log.append(current_disturbance)  # 记录扰动

            mujoco.mj_step(model, data)
            viewer.sync()

    # trial end: align and compute avg error
    aligned_time, aligned_err = ilc.align_trajectories(time_log, error_log)
    avg_err = np.mean(np.abs(aligned_err))
    all_avg_errors.append(avg_err)

    # 记录扰动数据
    ilc.record_trial_data(time_log, q_log, q_des_log, torque_log, error_log, aligned_time, aligned_err)

    print(f"Trial {trial+1} 平均绝对误差 = {math.degrees(avg_err):.3f}°")

    # stopping check
    if ilc.should_stop_learning(avg_err):
        break

    # update feedforward conservatively and print more info
    if trial < ilc.max_trials - 1:
        new_ff, ref_t = ilc.update_feedforward(error_log, time_log)
        print(f"更新前馈（learning_rate={ilc.learning_rate}, ff_clip={ilc.ff_clip}），新前馈 max={np.max(new_ff):.4f} min={np.min(new_ff):.4f}")


# plotting (增强图表显示扰动)

if len(all_avg_errors) > 0:
    plt.figure(figsize=(8,4))
    plt.plot([i+1 for i in range(len(all_avg_errors))], [math.degrees(x) for x in all_avg_errors], '-o')
    plt.axhline(y=2, color='r', linestyle='--', label='目标2°')
    plt.xlabel("Trial")
    plt.ylabel("Average absolute error (deg)")
    plt.legend()
    plt.grid(True)
    plt.title("ILC + OIAC + PD 控制性能 (带扰动)")
    plt.show()

if hasattr(ilc, 'trial_data') and ilc.trial_data:
    last = ilc.trial_data[-1]
    tlog = last['time']
    qlog = [math.degrees(x) for x in last['q']]
    qdeslog = [math.degrees(x) for x in last['q_des']]
    taulog = last['torque']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上子图：角度跟踪
    ax1.plot(tlog, qlog, label='q (deg)', linewidth=2)
    ax1.plot(tlog, qdeslog, '--', label='q_des (deg)', linewidth=2)
    ax1.set_ylabel('Angle (deg)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("最后试验的轨迹跟踪")
    
    # 下子图：扭矩和扰动
    ax2.plot(tlog, taulog, 'g-', label='控制扭矩 (Nm)', alpha=0.7)
    # 标记扰动区域
    ax2.axvspan(3.0, 4.0, alpha=0.2, color='red', label='Pulse disturbance')
    ax2.axvspan(6.0, 7.0, alpha=0.2, color='orange', label='Periodic disturbance') 
    ax2.axvspan(8.0, 9.0, alpha=0.2, color='purple', label='Constant disturbance')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

print("=== end ===")