import mujoco, mujoco.viewer
import time, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

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
    # 也可以使用：return math.radians(AMP_DEG) * math.sin(2 * math.pi * FREQ * t)

# 扭矩低通滤波器
alpha = 0.2
filtered_torque = 0.0

# 执行器索引
act_idx = 0

# 初始化关节位置
data.qpos[qpos_adr] = math.radians(20.0)
mujoco.mj_forward(model, data)

# 在线阻抗自适应类（修正后的以前的貌似有BUG
class ada_imp_con:
    """在线阻抗自适应（单自由度或多自由度）。
       返回刚度K和阻尼B作为(dof,dof) numpy数组。
    """
    def __init__(self, dof):
        self.DOF = dof  # 自由度数量
        # 使用浮点数组(dof,dof)表示刚度和阻尼
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        self.ff_tau_mat = np.zeros((self.DOF, 1))

        # 状态占位符（列向量）
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))

        # 自适应超参数（这些参数可以调整）
        self.a = 0.9   # 自适应标量分子
        self.b = 2.0   # 自适应标量分母乘数
        self.k = 0.9   # 跟踪误差组合中使用的增益

    def update_impedance(self, q, q_d, dq, dq_d):
        """使用当前/期望状态更新刚度K和阻尼B。
           输入q, q_d, dq, dq_d应为形状为(DOF,1)的列向量
        """
        # 将输入复制到内部状态（确保形状正确）
        self.q   = np.asarray(q).reshape((self.DOF,1))
        self.q_d = np.asarray(q_d).reshape((self.DOF,1))
        self.dq  = np.asarray(dq).reshape((self.DOF,1))
        self.dq_d= np.asarray(dq_d).reshape((self.DOF,1))

        # 计算分量误差
        pos_err = self.gen_pos_err()    # (DOF,1)
        vel_err = self.gen_vel_err()    # (DOF,1)
        track_err = self.gen_track_err()# (DOF,1)

        # 自适应标量 (a / (1 + b * ||track_err||^2))
        ad_factor = self.gen_ad_factor()

        # 外积生成(DOF x DOF)矩阵：
        # track_err * pos_err^T 得到 (DOF x DOF)
        # 乘以ad_factor来缩放幅度
        # 如有需要，可以在分母中添加小的正则化项（此处还没有添加）
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
        # 返回标量
        nrm = la.norm(self.gen_track_err())
        return self.a / (1.0 + self.b * (nrm * nrm + 1e-12))  # 添加小值以确保稳定性


# 为单自由度实例化控制器 
ada_imp = ada_imp_con(dof=1)

# 用于绘图的数据记录
time_log = []
q_log = []
q_des_log = []
torque_log = []

# 简单前馈增益（可以用学习到的tau_ff替换）
ff_gain = 8    # 前端常数，产生 tau_ff = ff_gain * pos_err

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0

        # 读取状态（标量）
        q = float(data.qpos[qpos_adr])
        qdot = float(data.qvel[dof_adr])

        # 期望状态（标量）
        q_des = float(target_angle_rad(t))

        # 修正速度：考虑 abs(sin) 的符号
        raw_sin = math.sin(2 * math.pi * FREQ * t)
        raw_cos = math.cos(2 * math.pi * FREQ * t)
        dq_des = 2 * math.pi * FREQ * math.radians(AMP_DEG) * math.cos(2 * math.pi * FREQ * t)

        # 构建控制器的列向量
        q_vec   = np.array([[q]])
        qd_vec  = np.array([[q_des]])
        dq_vec  = np.array([[qdot]])
        dqd_vec = np.array([[dq_des]])

        # 更新阻抗（k_mat, b_mat 这里是1x1矩阵）
        K_mat, B_mat = ada_imp.update_impedance(q_vec, qd_vec, dq_vec, dqd_vec)

        # 计算反馈扭矩：tau_fb = K * (q_d - q) + B * (dq_d - dq)
        # 对于单自由度，这些是1x1矩阵 to 标量
        pos_error_vec = (qd_vec - q_vec)
        vel_error_vec = (dqd_vec - dq_vec)
        tau_fb_vec = (K_mat @ pos_error_vec) + (B_mat @ vel_error_vec)

        # 简单前馈（是一种选择）：与位置误差成正比（可以替换）
        tau_ff_vec = ff_gain * pos_error_vec

        # 总扭矩（向量）
        tau_total_vec = tau_ff_vec + tau_fb_vec

        # 转换为标量
        torque = float(tau_total_vec.item())

        # 软限制（根据需要可保留可删除）
        if q < 0.0:
            K_soft = 150.0
            D_soft = 5.0
            soft_torque = K_soft * (0.0 - q) - D_soft * qdot
            if soft_torque < 0.0:
                soft_torque = 0.0
            torque = max(torque, soft_torque)

        # 饱和限制
        torque = max(TORQUE_MIN, min(torque, TORQUE_MAX))

        # 平滑扭矩（低通滤波）
        filtered_torque = alpha * torque + (1.0 - alpha) * filtered_torque

        # 应用控制
        data.ctrl[act_idx] = float(filtered_torque)

        # 记录数据
        time_log.append(t)
        q_log.append(math.degrees(q))
        q_des_log.append(math.degrees(q_des))
        torque_log.append(filtered_torque)

        # 每约0.2秒打印一次
        if int(t * 1000) % 200 == 0 and int((t - 0.001) * 1000) % 200 != 0:
            print(f"t={t:.2f}s | q_des={math.degrees(q_des):.2f}° | q={math.degrees(q):.2f}° | "
                  f"qdot={math.degrees(qdot):.2f}°/s | torque={filtered_torque:.2f}Nm")

        mujoco.mj_step(model, data)
        viewer.sync()

# 在同一图上绘制角度和扭矩，使用双y轴
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Angle (degrees)", color='blue')
ax1.plot(time_log, q_log, label="Actual Angle (q)", color='blue')
ax1.plot(time_log, q_des_log, label="Target Angle (q_des)", color='red', linestyle='--')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Torque (Nm)", color='green')
ax2.plot(time_log, torque_log, label="Filtered Torque", color='green')
ax2.tick_params(axis='y', labelcolor='green')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.title("Joint Angle and Control Torque vs Time (Adaptive Impedance)")
plt.grid(True)
plt.tight_layout()
plt.show()