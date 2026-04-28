"""
EMG-OIAC Hardware Runner  —  EMG -> PILCO+GP -> OIAC
======================================================
架构：
  [EMG线程 2000Hz]  →  qd_queue  →  [控制主循环 200Hz]  →  Motor
  - EMG线程：采集肌电 → 滤波 → LSTM预测 → predicted_angle → qd_queue
  - 控制主循环：取最新theta_d → 构建state → 策略动作 → OIAC扭矩 → 电机

用法：
    python run_emg_oiac_hardware.py                         # 自动找 policy_final.pkl
    python run_emg_oiac_hardware.py policy_final.pkl        # 指定 PKL 文件
    python run_emg_oiac_hardware.py policy_final.pkl 5      # 指定文件 + 试验次数

数据记录：t, q, q_des, K, B, Kff, jerk, tau, reward  -> CSV
"""

import os, sys, math, time, pickle, signal, atexit, threading, queue
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# ──────────────────────────────────────────────────────────────
#  EMG 相关导入
# ──────────────────────────────────────────────────────────────
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimizer_6
import AdaptiveEmbodiedControlSystems.LSTM as LSTM


# ══════════════════════════════════════════════════════════════
#  全局参数
# ══════════════════════════════════════════════════════════════

# ── EMG 参数 ─────────────────────────────────────────────────
FS           = 2000          # EMG 采样率 (Hz)
EMG_DT       = 1.0 / FS
USER_NAME    = 'VictorBNielsen'
LSTM_PATH    = "Outputs/models/LSTM/Windowed_LSTM.pth"

# EMG 优化器参数（与原 EMG 脚本保持一致）
EMG_B        = 4.0
EMG_K        = np.pi * 10.0 * 2

# ── 关节范围（EMG 和控制器共享）─────────────────────────────
THETA_MIN    = np.deg2rad(0)    # 0 rad
THETA_MAX    = np.deg2rad(140)  # ~2.44 rad
THETA_RANGE  = THETA_MAX - THETA_MIN

# ── 控制器参数 ────────────────────────────────────────────────
SAMPLE_RATE  = 200
DT           = 1.0 / SAMPLE_RATE
TORQUE_MAX   = 10.1
TORQUE_MIN   = -TORQUE_MAX

# OIAC 增益范围
K_MIN,   K_MAX   = 5.0,  25.0
B_MIN,   B_MAX   = 0.5,  3.0
KFF_MIN, KFF_MAX = 0.0,  3.0

DELTA_MAX   = 0.03
STATE_DIM   = 7
STATE_SCALE = np.ones(STATE_DIM, dtype=np.float64)

# 奖励权重
W_TRACK,  SIGMA_TRACK = 1.0, 0.05
W_JERK,   SIGMA_JERK  = 0.5, 0.3

# 滤波参数
VEL_FILTER_ALPHA_CTRL = 0.5
VEL_FILTER_ALPHA_ACC  = 0.92
ACC_FILTER_ALPHA      = 0.70
TAU_FILTER_ALPHA      = 0.01
DDTHETA_SMOOTH_N      = 5
N_LAG                 = 3

# ── 电机参数 ──────────────────────────────────────────────────
MOTOR_PORT       = 'COM5'
MOTOR_BAUD       = 1_000_000
TORQUE_DIRECTION = 1

# 标定参数
SINE_CENTER_DEG  = 70.0          # 归中位置（对应 THETA 中点）
RAW_MIN          = -15427
RAW_MAX          = -2922
ANGLE_RANGE_DEG  = 140.0
RAW_RANGE        = RAW_MAX - RAW_MIN
VEL_UNIT_RAD_S   = 0.229 * 2.0 * math.pi / 60.0

# ── 试验参数 ──────────────────────────────────────────────────
TRIAL_DURATION_S = 30.0
NUM_TRIALS       = 3

# ── 输出目录 ──────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial_logs")

# ── 设备 ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 全局停止事件 ──────────────────────────────────────────────
stop_event = threading.Event()


# ══════════════════════════════════════════════════════════════
#  EMG 线程  （生产者，2000 Hz）
# ══════════════════════════════════════════════════════════════
def emg_thread_fn(qd_queue: queue.Queue):
    """
    采集肌电 → 滤波 → 激活度 → optimizer_6 → LSTM → predicted_angle
    结果写入 qd_queue，控制器取最新值。

    qd_queue 元素格式：(theta_d_rad, timestamp)
    """
    # 滤波器
    filter_bicep  = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(FS, lp_cutoff=2, order=2)

    # 激活度解释器
    interpreter = PMC(
        theta_min=THETA_MIN, theta_max=THETA_MAX,
        user_name=USER_NAME, BicepEMG=True, TricepEMG=True
    )

    # RMS 滑动窗口
    Bicep_RMS_queue  = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    # LSTM 模型
    model = LSTM.LSTMModel(
        input_size=1, hidden_size=64,
        output_size=1, num_layers=1, batch_first=True
    ).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()

    # optimizer_6 状态
    emg_v             = 0.0
    optimized_angle   = float(np.deg2rad(SINE_CENTER_DEG))  # 从中心位置起步

    # 启动传感器
    emg = DelsysEMG(channel_range=(0, 1))
    emg.start()
    time.sleep(1.0)
    print("[EMG] 线程启动，开始采集...")

    while not stop_event.is_set():
        reading    = emg.read()
        timestamp  = time.time()

        # 带通滤波
        filtered_bicep  = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        # 滑动 RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS  = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms  = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        # 激活度 → 净激活
        activation   = interpreter.compute_activation(
            [filtered_bicep_rms, filtered_tricep_rms]
        )
        net_a        = activation[0] - activation[1]
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        # optimizer_6 → 平滑角度
        optimized_angle, emg_v, _ = optimizer_6(
            filtered_net_a, emg_v, EMG_DT,
            optimized_angle, THETA_MIN, THETA_MAX,
            np.pi, EMG_B, EMG_K
        )

        # LSTM 精细化预测
        with torch.no_grad():
            lstm_input      = torch.tensor([[optimized_angle]], dtype=torch.float32).to(device)
            lstm_output     = model(lstm_input)
            predicted_angle = float(lstm_output.item())

        # 硬限位保护
        predicted_angle = float(np.clip(predicted_angle, THETA_MIN, THETA_MAX))

        # 写入队列：始终保持最新帧（非阻塞，丢旧帧）
        try:
            qd_queue.put_nowait((predicted_angle, timestamp))
        except queue.Full:
            try:
                qd_queue.get_nowait()
            except queue.Empty:
                pass
            qd_queue.put_nowait((predicted_angle, timestamp))

    # 清理
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    print("[EMG] 线程已停止。")


# ══════════════════════════════════════════════════════════════
#  OIAC 控制器
# ══════════════════════════════════════════════════════════════
class OIAC:
    @staticmethod
    def _to_n(v, lo, hi):   return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    @staticmethod
    def _from_n(n, lo, hi): return lo + float(np.clip(n, 0.0, 1.0)) * (hi - lo)

    def __init__(self, K0=10.0, B0=1.0, Kff0=2.0):
        self._Kn   = self._to_n(K0,   K_MIN, K_MAX)
        self._Bn   = self._to_n(B0,   B_MIN, B_MAX)
        self._Kffn = self._to_n(Kff0, KFF_MIN, KFF_MAX)

    @property
    def K(self):   return self._from_n(self._Kn,   K_MIN, K_MAX)
    @property
    def B(self):   return self._from_n(self._Bn,   B_MIN, B_MAX)
    @property
    def Kff(self): return self._from_n(self._Kffn, KFF_MIN, KFF_MAX)

    def to_norm(self):
        return np.array([self._Kn, self._Bn, self._Kffn], dtype=np.float64)

    def apply_delta(self, delta: np.ndarray):
        n = self.to_norm() + delta
        self._Kn, self._Bn, self._Kffn = (
            float(np.clip(n[0], 0, 1)),
            float(np.clip(n[1], 0, 1)),
            float(np.clip(n[2], 0, 1))
        )

    def torque(self, theta, theta_d, dtheta_f, dtheta_d, ddtheta_d, B_w=1.0):
        tau = (self.K  * (theta_d - theta)
             + self.B  * B_w * (dtheta_d - dtheta_f)
             + self.Kff * ddtheta_d)
        return float(np.clip(tau, TORQUE_MIN, TORQUE_MAX))


# ══════════════════════════════════════════════════════════════
#  线性策略（从 PKL 加载）
# ══════════════════════════════════════════════════════════════
class LinearPolicyNumpy:
    def __init__(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        self.W         = d['W'].astype(np.float64)
        self.b         = d['b'].astype(np.float64)
        self.tag       = d.get('tag', 'unknown')
        self.ts        = d.get('timestamp', 0)
        self.state_dim = self.W.shape[1]
        print(f"[Policy] 加载成功: tag={self.tag}  时间={time.ctime(self.ts)}")
        print(f"[Policy] W={self.W.shape}  b={self.b.shape}  state_dim={self.state_dim}")
        if self.state_dim != STATE_DIM:
            print(f"[Policy] [WARN] 维度不匹配: runner={STATE_DIM}, pkl={self.state_dim}，将自动裁剪/填充")

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if len(state) > self.state_dim:
            state = state[:self.state_dim]
        elif len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        sn = (state * STATE_SCALE[:self.state_dim]).reshape(1, -1)
        a  = DELTA_MAX * np.tanh(sn @ self.W.T + self.b).flatten()
        return np.clip(a, -DELTA_MAX, DELTA_MAX)


# ══════════════════════════════════════════════════════════════
#  状态构建
# ══════════════════════════════════════════════════════════════
def build_state(e_pos, e_vel, acc, oiac: OIAC, jerk_norm=0.0):
    kn = oiac.to_norm()
    return np.array([
        float(np.clip(e_pos,     -math.pi, math.pi)),
        float(np.clip(e_vel,     -5.0,     5.0)),
        float(np.clip(acc,       -5.0,     5.0)),
        kn[0], kn[1], kn[2],
        float(np.clip(jerk_norm, -1.0,     1.0))
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════
#  电机接口
# ══════════════════════════════════════════════════════════════
class Motor:
    def __init__(self):
        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        if _SCRIPT_DIR not in sys.path:
            sys.path.insert(0, _SCRIPT_DIR)

        from Motors.DynamixelHardwareInterface import Motors
        if not getattr(Motors, '_patched', False):
            _orig = Motors.__init__
            def _safe(s, *a, **kw):
                s.num_motors = 0; s.motor_ids = []
                _orig(s, *a, **kw)
            Motors.__init__ = _safe
            Motors._patched = True

        print(f"[Motor] 连接: {MOTOR_PORT} @ {MOTOR_BAUD}...")
        self._m   = Motors(port=MOTOR_PORT, baudrate=MOTOR_BAUD)
        if not self._m.motor_ids:
            raise RuntimeError("未找到任何电机！")
        self._mid = self._m.motor_ids[0]
        print(f"[Motor] ✅ ID={self._mid} 连接成功")

        self._set_current_mode()

        raw_now        = self._raw_signed()
        self._raw_zero = raw_now - int(SINE_CENTER_DEG / ANGLE_RANGE_DEG * RAW_RANGE)
        deg_now        = self._raw_to_deg(raw_now)
        print(f"[Motor] 零点校准: 当前={deg_now:.1f}°  (centered={deg_now - SINE_CENTER_DEG:.1f}°)")

        atexit.register(self.stop)

    def _raw_signed(self):
        v = int(self._m.get_position(motor_id=self._mid))
        return v - 4294967296 if v > 2147483647 else v

    def _raw_to_deg(self, raw_signed):
        return float((raw_signed - self._raw_zero) / RAW_RANGE * ANGLE_RANGE_DEG)

    def _set_current_mode(self):
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
            ph = PortHandler(MOTOR_PORT)
            pk = PacketHandler(2.0)
            if ph.openPort() and ph.setBaudRate(MOTOR_BAUD):
                pk.write1ByteTxRx(ph, self._mid, 64, 0)
                time.sleep(0.1)
                pk.write1ByteTxRx(ph, self._mid, 11, 0)
                time.sleep(0.1)
                pk.write1ByteTxRx(ph, self._mid, 64, 1)
                ph.closePort()
                print("[Motor] ✅ 电流控制模式")
        except Exception as e:
            print(f"[Motor] [WARN] 模式设置失败: {e}")

    def read(self):
        for attempt in range(5):
            try:
                raw     = self._raw_signed()
                vel_raw = float(self._m.get_velocity(motor_id=self._mid))
                if vel_raw > 1023: vel_raw -= 2048
                deg_abs = self._raw_to_deg(raw)
                # 转换为以 SINE_CENTER_DEG 为零点的 centered angle，再偏移到 THETA 空间
                # theta = 0 对应 0°，theta = THETA_MAX 对应 140°
                theta  = math.radians(deg_abs)           # 绝对角度 rad
                dtheta = vel_raw * VEL_UNIT_RAD_S
                return theta, dtheta
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"连续5次读取失败: {e}")
                time.sleep(0.1 * (attempt + 1))

    def send(self, tau: float):
        tau = float(np.clip(tau, TORQUE_MIN, TORQUE_MAX))
        try:
            deg_abs = self._raw_to_deg(self._raw_signed())
        except Exception:
            deg_abs = SINE_CENTER_DEG
        # 软限位：超出关节范围时清零扭矩
        if deg_abs <= 0.5 and tau * TORQUE_DIRECTION < 0:
            tau = 0.0
        elif deg_abs >= ANGLE_RANGE_DEG - 0.5 and tau * TORQUE_DIRECTION > 0:
            tau = 0.0
        try:
            self._m.sendMotorCommand(
                self._mid, self._m.torq2curcom(tau * TORQUE_DIRECTION)
            )
        except Exception as e:
            print(f"[Motor] [WARN] 发送失败: {e}")

    def stop(self):
        for _ in range(3):
            try:
                self._m.sendMotorCommand(self._mid, 0)
                time.sleep(0.02)
            except Exception:
                pass

    def home(self, target_deg=SINE_CENTER_DEG, gain=0.5, tol=1.0, max_steps=500):
        """归中到 target_deg（绝对角度）"""
        print(f"[Motor] 归中到 {target_deg:.1f}°...")
        for _ in range(max_steps):
            if stop_event.is_set():
                break
            try:
                theta, _ = self.read()
                err      = target_deg - math.degrees(theta)
                self.send(float(np.clip(err * gain, -5.0, 5.0)))
                if abs(err) < tol:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        self.stop()
        time.sleep(0.8)
        theta, _ = self.read()
        print(f"[Motor] 归中完成: {math.degrees(theta):.2f}°")


# ══════════════════════════════════════════════════════════════
#  EMG 参考源（替代 SineReference）
# ══════════════════════════════════════════════════════════════
class EMGReference:
    """
    从 qd_queue 中读取最新 EMG 目标角度。
    - 没有新数据时，保持上一帧（hold-last）
    - 数值微分估算 dtheta_d 和 ddtheta_d（一阶差分 + 低通）
    """
    def __init__(self, qd_queue: queue.Queue):
        self._queue     = qd_queue
        self._theta_d   = float(np.deg2rad(SINE_CENTER_DEG))  # 初始值：中心位置
        self._dtheta_d  = 0.0
        self._ddtheta_d = 0.0
        self._prev_theta_d  = self._theta_d
        self._prev_dtheta_d = 0.0
        # 速度/加速度平滑（避免差分噪声过大）
        self._vel_alpha = 0.3
        self._acc_alpha = 0.3

    def update(self) -> tuple[float, float, float]:
        """
        拉取最新帧（非阻塞）；无新帧则 hold-last。
        返回 (theta_d, dtheta_d, ddtheta_d)
        """
        # 尽量排空 queue，只取最新一帧
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            new_theta_d = float(latest[0])
            # 一阶差分估算速度
            raw_vel = (new_theta_d - self._prev_theta_d) / DT
            self._dtheta_d = (self._vel_alpha * raw_vel
                              + (1 - self._vel_alpha) * self._dtheta_d)
            # 二阶差分估算加速度
            raw_acc = (self._dtheta_d - self._prev_dtheta_d) / DT
            self._ddtheta_d = (self._acc_alpha * raw_acc
                               + (1 - self._acc_alpha) * self._ddtheta_d)

            self._prev_theta_d  = new_theta_d
            self._prev_dtheta_d = self._dtheta_d
            self._theta_d       = new_theta_d

        return self._theta_d, self._dtheta_d, self._ddtheta_d

    def current_theta_d(self) -> float:
        return self._theta_d


# ══════════════════════════════════════════════════════════════
#  单次试验（EMG 驱动）
# ══════════════════════════════════════════════════════════════
def run_trial(motor: Motor, policy: LinearPolicyNumpy,
              qd_queue: queue.Queue,
              trial_num: int, duration_s: float,
              pkl_path: str) -> dict | None:

    motor.home()

    ref  = EMGReference(qd_queue)
    oiac = OIAC()

    # 速度范围估算（用于动态阻尼，EMG 场景取固定值）
    vel_d_max       = float(THETA_RANGE) * 0.5   # 保守估计
    JERK_NORM_SCALE = vel_d_max * 50.0

    # 滤波状态
    vel_ctrl = vel_acc = prev_vel = acc_f = prev_acc = 0.0
    jerk_f   = tau_f = 0.0
    ddtheta_buf: list[float] = []

    # 数据记录
    timestamps = []; thetas = []; theta_ds = []
    Ks = []; Bs = []; Kffs = []
    jerks = []; torques = []; rewards = []
    errors = []; accs = []

    high_tau_count = 0

    print(f"\n[Trial {trial_num}] ▶ 开始，时长={duration_s}s  （移动手臂以控制目标角度）")
    t_start = t_last = time.time()

    while True:
        now     = time.time()
        elapsed = now - t_start
        if elapsed >= duration_s or stop_event.is_set():
            break

        dt_actual = now - t_last
        if dt_actual < DT:
            time.sleep(DT - dt_actual)
            continue
        t_last = now

        # ── 参考轨迹：EMG 输入 ─────────────────────────────────
        theta_d, dtheta_d, ddtheta_d = ref.update()

        # 前向预测（N_LAG 步后的 dtheta_d，用简单线性外推）
        dtheta_d_fut = dtheta_d + ddtheta_d * N_LAG * DT

        # ── 读取电机 ──────────────────────────────────────────
        try:
            theta, dtheta = motor.read()
        except RuntimeError as e:
            print(f"[Trial {trial_num}] [WARN] {e}，跳过本步")
            motor.stop()
            time.sleep(0.3)
            continue

        e_pos = theta_d - theta

        # ── 速度双路滤波 ──────────────────────────────────────
        vel_ctrl = (VEL_FILTER_ALPHA_CTRL * vel_ctrl
                    + (1 - VEL_FILTER_ALPHA_CTRL) * dtheta)
        vel_acc  = (VEL_FILTER_ALPHA_ACC  * vel_acc
                    + (1 - VEL_FILTER_ALPHA_ACC)  * dtheta)

        # ── 加速度估算 ────────────────────────────────────────
        acc_raw  = (vel_acc - prev_vel) / DT
        acc_f    = ACC_FILTER_ALPHA * acc_f + (1 - ACC_FILTER_ALPHA) * acc_raw
        acc_f    = float(np.clip(acc_f, -10.0, 10.0))
        prev_vel = vel_acc

        # ── Jerk ─────────────────────────────────────────────
        jerk_f = 0.5 * jerk_f + 0.5 * (acc_f - prev_acc) / DT
        jerk_f = float(np.clip(jerk_f, -5.0, 5.0))
        jerk_n = float(np.clip(jerk_f / JERK_NORM_SCALE, -1.0, 1.0))
        prev_acc = acc_f

        # ── 前馈加速度平滑 ────────────────────────────────────
        ddtheta_buf.append(ddtheta_d)
        if len(ddtheta_buf) > DDTHETA_SMOOTH_N:
            ddtheta_buf.pop(0)
        ddtheta_sm = float(np.mean(ddtheta_buf))

        # ── 策略 → OIAC 增益更新 ──────────────────────────────
        state  = build_state(e_pos, dtheta_d_fut - vel_ctrl, acc_f, oiac, jerk_n)
        action = policy.get_action(state)
        oiac.apply_delta(action)

        # ── 动态阻尼（EMG 运动一般较慢，B_w 偏高）─────────────
        B_w = float(np.clip(1.0 - 0.7 * (abs(dtheta_d) / vel_d_max)**2, 0.3, 1.0))

        # ── 扭矩计算 + 过载保护 + 低通 ───────────────────────
        tau_raw = oiac.torque(theta, theta_d, vel_ctrl, dtheta_d, ddtheta_sm, B_w)
        tau_raw = float(np.clip(tau_raw, TORQUE_MIN, TORQUE_MAX))

        if abs(tau_raw) > TORQUE_MAX * 0.7:
            high_tau_count += 1
            if high_tau_count > 10:
                tau_raw = float(np.sign(tau_raw) * TORQUE_MAX * 0.5)
        else:
            high_tau_count = 0

        tau_f = TAU_FILTER_ALPHA * tau_f + (1 - TAU_FILTER_ALPHA) * tau_raw
        motor.send(tau_f)

        # ── 奖励 ──────────────────────────────────────────────
        e_c = float(np.clip(e_pos, -math.pi, math.pi))
        r   = (W_TRACK * math.exp(-0.5 * (e_c    / SIGMA_TRACK)**2)
             + W_JERK  * math.exp(-0.5 * (jerk_n / SIGMA_JERK )**2))

        # ── 记录 ──────────────────────────────────────────────
        timestamps.append(elapsed)
        thetas.append(theta)
        theta_ds.append(theta_d)
        Ks.append(oiac.K)
        Bs.append(oiac.B)
        Kffs.append(oiac.Kff)
        jerks.append(jerk_f)
        torques.append(tau_f)
        rewards.append(r)
        errors.append(abs(math.degrees(e_c)))
        accs.append(acc_f)

        # 实时打印（每秒一次）
        if len(rewards) % SAMPLE_RATE == 0:
            print(f"    t={elapsed:.0f}s  "
                  f"θ_d={math.degrees(theta_d):.1f}°  "
                  f"θ={math.degrees(theta):.1f}°  "
                  f"err={math.degrees(e_c):+.1f}°  "
                  f"K={oiac.K:.1f}  B={oiac.B:.3f}  Kff={oiac.Kff:.2f}  "
                  f"τ={tau_f:.2f}Nm  r={r:.4f}")

    motor.stop()

    if not rewards:
        return None

    # ── CSV 导出 ──────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_stem = os.path.splitext(os.path.basename(pkl_path))[0]
    csv_path = os.path.join(LOG_DIR, f"emg_{pkl_stem}_trial{trial_num:02d}_{ts_str}.csv")

    df = pd.DataFrame({
        "t_s":       timestamps,
        "q_rad":     thetas,
        "q_des_rad": theta_ds,
        "K":         Ks,
        "B":         Bs,
        "Kff":       Kffs,
        "jerk":      jerks,
        "tau_Nm":    torques,
        "reward":    rewards,
    })
    df.to_csv(csv_path, index=False)
    print(f"\n[Trial {trial_num}] 💾 数据已保存: {csv_path}  ({len(df)} 行)")

    result = dict(
        avg_reward = float(np.mean(rewards)),
        track_rmse = float(np.sqrt(np.mean(np.square(errors)))),
        acc_rms    = float(np.sqrt(np.mean(np.square(accs)))),
        max_err    = float(np.max(errors)),
        max_tau    = float(np.max(np.abs(torques))),
        final_K    = oiac.K,
        final_B    = oiac.B,
        final_Kff  = oiac.Kff,
        csv_path   = csv_path,
    )
    print(f"\n[Trial {trial_num}] 结果:")
    print(f"    平均奖励   = {result['avg_reward']:.4f}")
    print(f"    跟踪 RMSE  = {result['track_rmse']:.3f}°")
    print(f"    最大误差   = {result['max_err']:.2f}°")
    print(f"    加速度 RMS = {result['acc_rms']:.3f} rad/s²")
    print(f"    最大扭矩   = {result['max_tau']:.2f} Nm")
    print(f"    最终增益   K={result['final_K']:.2f}  "
          f"B={result['final_B']:.3f}  Kff={result['final_Kff']:.2f}")
    return result


# ══════════════════════════════════════════════════════════════
#  汇总打印
# ══════════════════════════════════════════════════════════════
def print_summary(results: list, pkl_path: str):
    print(f"\n{'='*55}")
    print(f"  汇总  —  {os.path.basename(pkl_path)}")
    print(f"  试验次数: {len(results)}")
    print(f"{'='*55}")
    keys   = ['avg_reward', 'track_rmse', 'acc_rms', 'max_err', 'max_tau']
    labels = ['平均奖励', '跟踪RMSE(°)', '加速度RMS', '最大误差(°)', '最大扭矩(Nm)']
    for k, lb in zip(keys, labels):
        vals = [r[k] for r in results]
        best = np.max(vals) if k == 'avg_reward' else np.min(vals)
        print(f"  {lb:14s}  mean={np.mean(vals):.4f}  best={best:.4f}")
    print(f"\n  CSV 文件:")
    for r in results:
        print(f"    {r['csv_path']}")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════
#  PKL 自动查找
# ══════════════════════════════════════════════════════════════
def _find_pkl():
    policy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pilco_oiac_policies")
    for name in ["policy_final.pkl", "policy_latest.pkl"]:
        p = os.path.join(policy_dir, name)
        if os.path.exists(p):
            return p
    import glob
    found = sorted(glob.glob(os.path.join(policy_dir, "*.pkl")))
    return found[-1] if found else None


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pkl_path   = sys.argv[1] if len(sys.argv) > 1 else _find_pkl()
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_TRIALS

    if not pkl_path or not os.path.exists(pkl_path):
        print(f"❌ 找不到 PKL 文件: {pkl_path}")
        print("用法: python run_emg_oiac_hardware.py [policy.pkl] [num_trials]")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  EMG-PILCO-OIAC 硬件运行器")
    print(f"  PKL:   {pkl_path}")
    print(f"  试验数: {num_trials}  时长: {TRIAL_DURATION_S}s/次")
    print(f"{'='*55}")

    # 加载策略
    policy = LinearPolicyNumpy(pkl_path)

    # 连接电机
    try:
        motor = Motor()
    except Exception as e:
        print(f"\n❌ 电机连接失败: {e}")
        sys.exit(1)

    # 信号处理
    def _on_interrupt(*_):
        print("\n[中断] 正在安全停止...")
        stop_event.set()
        motor.stop()

    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    # 共享队列：maxsize=3，保证控制器始终取到最新帧
    qd_queue = queue.Queue(maxsize=3)

    # 启动 EMG 线程
    emg_thread = threading.Thread(
        target=emg_thread_fn,
        args=(qd_queue,),
        name="EMG-Thread",
        daemon=True          # 主程序退出时自动结束
    )
    emg_thread.start()

    # 等待 EMG 线程预热（等到第一帧数据进来）
    print("[Main] 等待 EMG 初始化...")
    while qd_queue.empty() and not stop_event.is_set():
        time.sleep(0.1)
    print("[Main] EMG 就绪，开始试验。")

    # 逐次试验
    all_results = []
    for i in range(num_trials):
        if stop_event.is_set():
            break
        print(f"\n  按 Enter 开始第 {i+1}/{num_trials} 次试验（Ctrl+C 退出）...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break

        res = run_trial(
            motor=motor,
            policy=policy,
            qd_queue=qd_queue,
            trial_num=i + 1,
            duration_s=TRIAL_DURATION_S,
            pkl_path=pkl_path,
        )
        if res:
            all_results.append(res)

    # 停止
    stop_event.set()
    motor.stop()
    emg_thread.join(timeout=3.0)

    if all_results:
        print_summary(all_results, pkl_path)

    print("程序结束，电机已安全停止。")