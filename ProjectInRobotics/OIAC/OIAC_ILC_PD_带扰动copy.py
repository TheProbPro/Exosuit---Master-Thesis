import mujoco, mujoco.viewer
import time, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import interpolate

MODEL_PATH = "mergedCopy.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 关节/自由度
joint_name = "el_x"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
qpos_adr = model.jnt_qposadr[joint_id]
dof_adr  = model.jnt_dofadr[joint_id]

# 扭矩限制
TORQUE_MIN = -4.1
TORQUE_MAX = 4.1

# 目标轨迹
AMP_DEG = 60.0
FREQ = 0.5
def target_angle_rad(t):
    return math.radians(AMP_DEG) * abs(math.sin(2 * math.pi * FREQ * t))

# 扭矩低通滤波器
alpha = 0.2
filtered_torque = 0.0

# 执行器索引
act_idx = 0

# 初始化关节位置
data.qpos[qpos_adr] = math.radians(20.0)
mujoco.mj_forward(model, data)

# OIAC 学习类
class OIAC_Learner:
    """OIAC 学习器，根据图片中的想法实现迭代学习"""
    def __init__(self, trial_duration=4.0, dt=0.002):
        self.trial_duration = trial_duration  # 每次试验的持续时间
        self.dt = dt  # 时间步长
        
        # 学习参数
        self.learning_rate = 0.8  # 学习率
        self.k_ff = 8.0  # 前馈增益初始值
        
        # 存储每次试验的数据
        self.trials_data = []
        self.ff_functions = []  # 存储每次试验的前馈函数
        self.current_trial = 0
        
        # 当前试验的前馈补偿项
        self.ff_compensation = None
        
    def start_new_trial(self):
        """开始新的试验"""
        self.current_trial += 1
        print(f"开始第 {self.current_trial} 次试验")
        
        # 为当前试验初始化数据记录
        trial_data = {
            'time': [],
            'q_actual': [],
            'q_desired': [],
            'error': [],
            'torque': [],
            'ff_compensation_values': []  # 改为存储数值而不是函数
        }
        self.trials_data.append(trial_data)
        
        # 如果是第一次试验，初始化前馈补偿为0
        if self.current_trial == 1:
            self.ff_compensation = lambda t: 0.0
        else:
            # 基于上一次试验的误差更新前馈补偿
            self._update_ff_compensation()
        
        # 保存当前的前馈函数
        self.ff_functions.append(self.ff_compensation)
    
    def record_data(self, t, q_actual, q_desired, torque, ff_comp):
        """记录当前时间步的数据"""
        if self.current_trial == 0:
            return
            
        trial_data = self.trials_data[-1]
        trial_data['time'].append(t)
        trial_data['q_actual'].append(q_actual)
        trial_data['q_desired'].append(q_desired)
        trial_data['error'].append(q_desired - q_actual)
        trial_data['torque'].append(torque)
        trial_data['ff_compensation_values'].append(ff_comp)  # 存储数值
    
    def _update_ff_compensation(self):
        """基于上一次试验的误差更新前馈补偿项"""
        if len(self.trials_data) < 2:
            return
            
        prev_trial_data = self.trials_data[-2]
        
        # 获取时间和误差数据
        time_data = np.array(prev_trial_data['time'])
        error_data = np.array(prev_trial_data['error'])
        
        # 创建样条插值函数来平滑误差数据
        if len(time_data) > 10:  # 确保有足够的数据点
            try:
                # 使用样条插值创建平滑的误差函数
                tck = interpolate.splrep(time_data, error_data, s=0.1)
                error_function = lambda t: interpolate.splev(t, tck)
            except:
                # 如果样条失败，使用线性插值
                error_function = interpolate.interp1d(time_data, error_data, 
                                                    kind='linear', 
                                                    fill_value='extrapolate')
        else:
            # 数据点不足时使用线性插值
            error_function = interpolate.interp1d(time_data, error_data, 
                                                kind='linear', 
                                                fill_value='extrapolate')
        
        # 保存上一次试验的前馈补偿（如果有）
        old_ff = self.ff_compensation
        
        # 更新前馈补偿：ff_new(t) = ff_old(t) + learning_rate * e_prev(t)
        def new_ff_compensation(t):
            if old_ff is None:
                return self.learning_rate * error_function(t)
            else:
                return old_ff(t) + self.learning_rate * error_function(t)
        
        self.ff_compensation = new_ff_compensation
        
        # 计算上一次试验的误差指标
        max_error = np.max(np.abs(error_data))
        rms_error = np.sqrt(np.mean(error_data**2))
        print(f"试验 {self.current_trial-1} 结果: 最大误差={np.degrees(max_error):.2f}°, RMS误差={np.degrees(rms_error):.2f}°")
    
    def get_ff_compensation(self, t):
        """获取当前时间的前馈补偿"""
        if self.ff_compensation is None:
            return 0.0
        return self.ff_compensation(t)
    
    def should_end_trial(self, t):
        """检查是否应该结束当前试验"""
        return t >= self.trial_duration

# 在线阻抗自适应类
class ada_imp_con:
    def __init__(self, dof):
        self.DOF = dof
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))

        self.a = 0.9
        self.b = 2.0
        self.k = 0.9

    def update_impedance(self, q, q_d, dq, dq_d):
        self.q   = np.asarray(q).reshape((self.DOF,1))
        self.q_d = np.asarray(q_d).reshape((self.DOF,1))
        self.dq  = np.asarray(dq).reshape((self.DOF,1))
        self.dq_d= np.asarray(dq_d).reshape((self.DOF,1))

        pos_err = self.gen_pos_err()
        vel_err = self.gen_vel_err()
        track_err = self.gen_track_err()
        ad_factor = self.gen_ad_factor()

        self.k_mat = ad_factor * (track_err @ pos_err.T)
        self.b_mat = ad_factor * (track_err @ vel_err.T)

        return self.k_mat, self.b_mat

    def gen_pos_err(self):
        return (self.q - self.q_d)

    def gen_vel_err(self):
        return (self.dq - self.dq_d)

    def gen_track_err(self):
        return (self.k * self.gen_vel_err() + self.gen_pos_err())

    def gen_ad_factor(self):
        nrm = la.norm(self.gen_track_err())
        return self.a / (1.0 + self.b * (nrm * nrm + 1e-12))

# 初始化控制器和学习器
ada_imp = ada_imp_con(dof=1)
oiac_learner = OIAC_Learner(trial_duration=4.0)  # 每次试验4秒

# 用于最终绘图的数据记录
final_time_log = []
final_q_log = []
final_q_des_log = []
final_torque_log = []
final_ff_comp_log = []

# 进行多次试验学习
num_trials = 5  # 总共进行5次试验

with mujoco.viewer.launch_passive(model, data) as viewer:
    for trial in range(num_trials):
        # 开始新试验
        oiac_learner.start_new_trial()
        
        # 重置模拟状态
        data.qpos[qpos_adr] = math.radians(20.0)
        data.qvel[dof_adr] = 0.0
        mujoco.mj_forward(model, data)
        
        t0 = time.time()
        trial_start_time = time.time() - t0
        
        # 清空当前试验的记录（为最终绘图保留最后一次试验的数据）
        if trial == num_trials - 1:  # 最后一次试验
            final_time_log.clear()
            final_q_log.clear()
            final_q_des_log.clear()
            final_torque_log.clear()
            final_ff_comp_log.clear()
        
        while viewer.is_running():
            t = time.time() - t0 - trial_start_time
            
            # 检查是否结束当前试验
            if oiac_learner.should_end_trial(t):
                break
            
            # 读取状态
            q = float(data.qpos[qpos_adr])
            qdot = float(data.qvel[dof_adr])
            
            # 期望状态
            q_des = float(target_angle_rad(t))
            dq_des = 2 * math.pi * FREQ * math.radians(AMP_DEG) * math.cos(2 * math.pi * FREQ * t)
            
            # OIAC 前馈补偿
            ff_comp = oiac_learner.get_ff_compensation(t)
            
            # 构建控制向量
            q_vec   = np.array([[q]])
            qd_vec  = np.array([[q_des]])
            dq_vec  = np.array([[qdot]])
            dqd_vec = np.array([[dq_des]])
            
            # 更新阻抗
            K_mat, B_mat = ada_imp.update_impedance(q_vec, qd_vec, dq_vec, dqd_vec)
            
            # 计算反馈扭矩
            pos_error_vec = (qd_vec - q_vec)
            vel_error_vec = (dqd_vec - dq_vec)
            tau_fb_vec = (K_mat @ pos_error_vec) + (B_mat @ vel_error_vec)
            
            # 总扭矩 = 前馈补偿 + 反馈控制
            tau_total_vec = ff_comp + tau_fb_vec
            
            # 转换为标量
            torque = float(tau_total_vec.item())
            
            # 软限制
            if q < 0.0:
                K_soft = 150.0
                D_soft = 5.0
                soft_torque = K_soft * (0.0 - q) - D_soft * qdot
                if soft_torque < 0.0:
                    soft_torque = 0.0
                torque = max(torque, soft_torque)
            
            # 饱和限制
            torque = max(TORQUE_MIN, min(torque, TORQUE_MAX))
            
            # 平滑扭矩
            filtered_torque = alpha * torque + (1.0 - alpha) * filtered_torque
            
            # 应用控制
            data.ctrl[act_idx] = float(filtered_torque)
            
            # 记录数据
            oiac_learner.record_data(t, q, q_des, filtered_torque, ff_comp)
            
            if trial == num_trials - 1:  # 只记录最后一次试验用于最终绘图
                final_time_log.append(t + trial * oiac_learner.trial_duration)
                final_q_log.append(math.degrees(q))
                final_q_des_log.append(math.degrees(q_des))
                final_torque_log.append(filtered_torque)
                final_ff_comp_log.append(ff_comp)
            
            # 每约0.2秒打印一次（仅最后一次试验）
            if trial == num_trials - 1 and int(t * 1000) % 200 == 0 and int((t - 0.001) * 1000) % 200 != 0:
                error_deg = math.degrees(q_des - q)
                print(f"最终试验 t={t:.2f}s | 误差={error_deg:.2f}° | "
                      f"q_des={math.degrees(q_des):.2f}° | q={math.degrees(q):.2f}° | "
                      f"ff_comp={ff_comp:.2f}Nm")
            
            mujoco.mj_step(model, data)
            viewer.sync()
    
    print("所有试验完成！")

# 绘制学习过程结果
plt.figure(figsize=(15, 10))

# 1. 最终试验的跟踪性能
plt.subplot(2, 2, 1)
plt.plot(final_time_log, final_q_log, label="实际角度", linewidth=2)
plt.plot(final_time_log, final_q_des_log, label="期望角度", linestyle='--', linewidth=2)
plt.xlabel("时间 (s)")
plt.ylabel("角度 (°)")
plt.title("最终试验: 角度跟踪")
plt.legend()
plt.grid(True)

# 2. 扭矩和前馈补偿
plt.subplot(2, 2, 2)
plt.plot(final_time_log, final_torque_log, label="总扭矩", linewidth=2)
plt.plot(final_time_log, final_ff_comp_log, label="前馈补偿", linestyle='--', linewidth=2)
plt.xlabel("时间 (s)")
plt.ylabel("扭矩 (Nm)")
plt.title("最终试验: 控制扭矩")
plt.legend()
plt.grid(True)

# 3. 每次试验的误差收敛情况
plt.subplot(2, 2, 3)
max_errors = []
rms_errors = []
for i, trial_data in enumerate(oiac_learner.trials_data):
    if len(trial_data['error']) > 0:
        errors_deg = np.degrees(np.array(trial_data['error']))
        max_errors.append(np.max(np.abs(errors_deg)))
        rms_errors.append(np.sqrt(np.mean(errors_deg**2)))

if max_errors:  # 确保有数据
    plt.plot(range(1, len(max_errors) + 1), max_errors, 'o-', label='最大误差', linewidth=2)
    plt.plot(range(1, len(rms_errors) + 1), rms_errors, 's-', label='RMS误差', linewidth=2)
    plt.xlabel("试验次数")
    plt.ylabel("误差 (°)")
    plt.title("学习收敛过程")
    plt.legend()
    plt.grid(True)

# 4. 前馈补偿的演化
plt.subplot(2, 2, 4)
time_base = np.linspace(0, oiac_learner.trial_duration, 100)

# 绘制前几次试验的前馈补偿函数
for i in range(min(3, len(oiac_learner.ff_functions))):
    ff_values = [oiac_learner.ff_functions[i](t) for t in time_base]
    plt.plot(time_base, ff_values, label=f'试验 {i+1}', linewidth=2)

plt.xlabel("时间 (s)")
plt.ylabel("前馈补偿 (Nm)")
plt.title("前馈补偿的演化")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印学习总结
print("\n=== OIAC 学习总结 ===")
print(f"总试验次数: {num_trials}")
if len(max_errors) > 1:
    improvement = (max_errors[0] - max_errors[-1]) / max_errors[0] * 100
    print(f"最大误差改善: {improvement:.1f}% (从 {max_errors[0]:.2f}° 到 {max_errors[-1]:.2f}°)")
if len(rms_errors) > 1:
    improvement_rms = (rms_errors[0] - rms_errors[-1]) / rms_errors[0] * 100
    print(f"RMS误差改善: {improvement_rms:.1f}% (从 {rms_errors[0]:.2f}° 到 {rms_errors[-1]:.2f}°)")