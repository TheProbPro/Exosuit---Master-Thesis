import mujoco, mujoco.viewer
import time, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import interpolate

# ---------------------------
# 配置
# ---------------------------
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
act_idx = 0   

# initialize qpos
data.qpos[qpos_adr] = math.radians(20.0)
mujoco.mj_forward(model, data)


# OIAC: 在线阻抗自适应（保守）

class ada_imp_con:
    def __init__(self, dof):
        self.DOF = dof
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        self.q = np.zeros((self.DOF,1))
        self.q_d = np.zeros((self.DOF,1))
        self.dq = np.zeros((self.DOF,1))
        self.dq_d = np.zeros((self.DOF,1))
        # adaptive hyperparams (tune as needed)
        self.a = 0.9
        self.b = 2.0
        self.k = 0.9
        # bounds - K_MIN set to 0 to avoid forcing positive stiffness if original OIAC used different behavior
        self.K_MIN = 0.0
        self.K_MAX = 50.0
        self.B_MIN = 0.0
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
        ad_factor = self.a / (1.0 + self.b * (nrm * nrm + 1e-12))

        k_mat = ad_factor * (track_err @ pos_err.T)
        b_mat = ad_factor * (track_err @ vel_err.T)

        k_val = float(k_mat[0,0]) if k_mat.size>0 else 0.0
        b_val = float(b_mat[0,0]) if b_mat.size>0 else 0.0

        # clip to reasonable ranges (but K_MIN = 0 is conservative)
        k_val = np.clip(k_val, self.K_MIN, self.K_MAX)
        b_val = np.clip(b_val, self.B_MIN, self.B_MAX)

        self.k_mat = np.array([[k_val]])
        self.b_mat = np.array([[b_val]])
        return self.k_mat, self.b_mat

# ---------------------------
# Iterative Learning Control (trial-wise), conservative & robust
# ---------------------------
class IterativeLearningControl:
    def __init__(self, max_trials=50, error_threshold_rad=math.radians(5),
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

# ---------------------------
# instantiate controllers
# ---------------------------
ada_imp = ada_imp_con(dof=1)

# ---------- MODIFIED PARAMETERS (as you asked) ----------
gain_ff = 5.0   # runtime multiplier for learned feedforward (adjustable)
ilc = IterativeLearningControl(max_trials=20, error_threshold_rad=math.radians(2),
                               learning_rate=0.45, target_length=1000, ff_clip=4.0)
# --------------------------------------------------------

# optional small PD baseline (can be left zero)
Kp_base = 0.0
Kd_base = 0.0

# ---------------------------
# main trial loop
# ---------------------------
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

            # OIAC feedback
            q_vec = np.array([[q]])
            qd_vec = np.array([[q_des]])
            dq_vec = np.array([[qdot]])
            dqd_vec = np.array([[dq_des]])

            K_mat, B_mat = ada_imp.update_impedance(q_vec, qd_vec, dq_vec, dqd_vec)
            pos_error_vec = (qd_vec - q_vec)
            vel_error_vec = (dqd_vec - dq_vec)
            tau_fb_vec = (K_mat @ pos_error_vec) + (B_mat @ vel_error_vec)
            tau_fb = float(tau_fb_vec.item())

            # small PD baseline
            pd_torque = Kp_base * error + Kd_base * (dq_des - qdot)

            # compose torques (apply runtime gain on feedforward)
            torque_raw = (gain_ff * tau_ff) + tau_fb + pd_torque

            # diagnostic print every 0.5s
            if int(t * 1000) % 500 == 0 and int((t - 0.001) * 1000) % 500 != 0:
                print(f"DIAG t={t:.3f} tau_ff={tau_ff:.4f} tau_fb={tau_fb:.4f} pd={pd_torque:.4f} raw={torque_raw:.4f}")

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

            mujoco.mj_step(model, data)
            viewer.sync()

    # trial end: align and compute avg error
    aligned_time, aligned_err = ilc.align_trajectories(time_log, error_log)
    avg_err = np.mean(np.abs(aligned_err))
    all_avg_errors.append(avg_err)

    ilc.record_trial_data(time_log, q_log, q_des_log, torque_log, error_log, aligned_time, aligned_err)

    print(f"Trial {trial+1} 平均绝对误差 = {math.degrees(avg_err):.3f}°")

    # stopping check
    if ilc.should_stop_learning(avg_err):
        break

    # update feedforward conservatively and print more info
    if trial < ilc.max_trials - 1:
        new_ff, ref_t = ilc.update_feedforward(error_log, time_log)
        print(f"更新前馈（learning_rate={ilc.learning_rate}, ff_clip={ilc.ff_clip}），新前馈 max={np.max(new_ff):.4f} min={np.min(new_ff):.4f}")
        # print first 10 samples for quick inspection
        print("新前馈前10样本:", np.round(new_ff[:10], 4))

# ---------------------------
# plotting
# ---------------------------
if len(all_avg_errors) > 0:
    plt.figure(figsize=(8,4))
    plt.plot([i+1 for i in range(len(all_avg_errors))], [math.degrees(x) for x in all_avg_errors], '-o')
    plt.axhline(y=5, color='r', linestyle='--', label='目标5°')
    plt.xlabel("Trial")
    plt.ylabel("Average absolute error (deg)")
    plt.legend()
    plt.grid(True)
    plt.show()

if hasattr(ilc, 'trial_data') and ilc.trial_data:
    last = ilc.trial_data[-1]
    tlog = last['time']
    qlog = [math.degrees(x) for x in last['q']]
    qdeslog = [math.degrees(x) for x in last['q_des']]
    taulog = last['torque']

    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(tlog, qlog, label='q (deg)')
    ax1.plot(tlog, qdeslog, '--', label='q_des (deg)')
    ax1.set_ylabel('Angle (deg)')
    ax2 = ax1.twinx()
    ax2.plot(tlog, taulog, label='torque (Nm)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

print("=== 结束 ===")
