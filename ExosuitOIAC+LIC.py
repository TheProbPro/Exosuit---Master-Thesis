"""
EMG-OIAC Hardware Runner  —  EMG -> ada_imp_con + ILC (AAN) -> Motor
=====================================================================
架构：
  [EMG线程 2000Hz]  →  qd_queue  →  [控制主循环 200Hz]  →  Motor
  - EMG线程：采集肌电 → 滤波 → LSTM预测 → predicted_angle → qd_queue
  - 控制主循环：取最新theta_d → 构建state → ada_imp_con扭矩 + ILC前馈 → 电机

ILC逻辑：
  - Trial N运行时：从上一次update_learning得到的ff数组里 get_feedforward
  - Trial N结束后：update_learning(error_array) 更新ff数组，供Trial N+1使用
  - Trial 1时 learned_feedforward为空，ILC前馈全为0

用法：
    python run_emg_oiac_hardware.py                  # 直接运行
    python run_emg_oiac_hardware.py 5                # 指定试验次数

数据记录：t, q, q_des, K, B, jerk, tau, ilc_ff, reward  -> CSV
"""

import os, sys, math, time, signal, atexit, threading, queue
import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  EMG 相关导入
# ──────────────────────────────────────────────────────────────
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimizer_6
import AdaptiveEmbodiedControlSystems.LSTM as LSTM

# ──────────────────────────────────────────────────────────────
#  MOTOR 相关导入
# ──────────────────────────────────────────────────────────────
from Motors.DynamixelHardwareInterface import Motors


# ══════════════════════════════════════════════════════════════
#  全局参数
# ══════════════════════════════════════════════════════════════

# ── EMG 参数 ─────────────────────────────────────────────────
FS           = 2000
EMG_DT       = 1.0 / FS
USER_NAME    = 'VictorBNielsen'
LSTM_PATH    = "Outputs/models/LSTM/Windowed_LSTM.pth"

EMG_B        = 4.0
EMG_K        = np.pi * 1.7

plot_dq = []

# ── 关节范围 ──────────────────────────────────────────────────
THETA_MIN       = np.deg2rad(0)
THETA_MAX       = np.deg2rad(140)
THETA_RANGE     = THETA_MAX - THETA_MIN

# ── 控制器参数 ────────────────────────────────────────────────
plot_q   = []
plot_tau = []
SAMPLE_RATE  = 200
DT           = 1.0 / SAMPLE_RATE
TORQUE_MAX   = 10.1
TORQUE_MIN   = -TORQUE_MAX

# ── ada_imp_con 初始增益参数（可调）─────────────────────────
ADA_A = 1.0
ADA_B = 0.1
ADA_K = 0.005

# ── ILC 参数 ──────────────────────────────────────────────────
ILC_LR             = 0.1
ILC_REFERENCE_LEN  = int(TRIAL_DURATION_S := 30.0) * SAMPLE_RATE   # 6000

# ── 奖励权重 ──────────────────────────────────────────────────
W_TRACK,  SIGMA_TRACK = 1.0, 0.05
W_JERK,   SIGMA_JERK  = 0.5, 0.3

# ── 滤波参数 ──────────────────────────────────────────────────
VEL_FILTER_ALPHA_CTRL = 0.5
VEL_FILTER_ALPHA_ACC  = 0.92
ACC_FILTER_ALPHA      = 0.70
TAU_FILTER_ALPHA      = 0.01
DDTHETA_SMOOTH_N      = 5
N_LAG                 = 3

# ── 电机参数 ──────────────────────────────────────────────────
MOTOR_PORT       = 'COM4'
MOTOR_BAUD       = 4_500_000
TORQUE_DIRECTION = 1

# ── 标定参数 ──────────────────────────────────────────────────
SINE_CENTER_DEG  = 0.0
RAW_MIN          = -15427
RAW_MAX          = -2922
ANGLE_RANGE_DEG  = 140.0
RAW_RANGE        = RAW_MAX - RAW_MIN
VEL_UNIT_RAD_S   = 0.229 * 2.0 * math.pi / 60.0

# ── 试验参数 ──────────────────────────────────────────────────
NUM_TRIALS       = 3

# ── 输出目录 ──────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial_logs")

# ── 设备 
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
    """
    filter_bicep  = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(FS, lp_cutoff=2, order=2)

    interpreter = PMC(
        theta_min=THETA_MIN, theta_max=THETA_MAX,
        user_name=USER_NAME, BicepEMG=True, TricepEMG=True
    )

    Bicep_RMS_queue  = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    model = LSTM.LSTMModel(
        input_size=1, hidden_size=64,
        output_size=1, num_layers=1, batch_first=True
    ).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()

    window         = deque(maxlen=100)
    emg_v          = 0.0
    optimized_angle = float(np.deg2rad(SINE_CENTER_DEG))
    sample_counter  = 0
    net_a_prev      = 0.0

    emg = DelsysEMG(channel_range=(0, 1))
    emg.start()
    print("[EMG] 线程启动，开始采集...")

    while not stop_event.is_set():
        reading    = emg.read()
        sample_counter += 1

        filtered_bicep  = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get_nowait()
        Bicep_RMS_queue.put_nowait(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get_nowait()
        Tricep_RMS_queue.put_nowait(filtered_tricep)

        Bicep_RMS  = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms  = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        activation = interpreter.compute_activation(
            [filtered_bicep_rms, filtered_tricep_rms]
        )
        net_a     = activation[0] - activation[1]
        net_a_old = net_a
        net_a     = (net_a - net_a_prev) / EMG_DT
        net_a_prev = net_a_old

        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        optimized_angle, emg_v, _ = optimizer_6(
            filtered_net_a, emg_v, EMG_DT,
            optimized_angle, THETA_MIN, THETA_MAX,
            np.pi, EMG_B, EMG_K
        )

        window.append(optimized_angle)
        if len(window) < window.maxlen:
            continue

        if len(window) == window.maxlen and sample_counter % 10 == 0:
            with torch.inference_mode():
                input_tensor = torch.as_tensor(
                    window, dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(-1)
                lstm_output     = model(input_tensor)
                predicted_angle = float(lstm_output.detach().cpu().item())

            try:
                qd_queue.put_nowait(predicted_angle)
            except queue.Full:
                qd_queue.get_nowait()
                qd_queue.put_nowait(predicted_angle)

    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    print("[EMG] 线程已停止。")



#  自适应阻抗控制器  ada_imp_con  （DOF=1）

class AdaImpCon:
    """
    移植自 ada_imp_con，DOF=1 单自由度版本。
    所有内部矩阵保持原始 numpy.matrix 形式以保证运算一致性。
    对外接口统一返回 Python float，方便控制主循环使用。
    """
    def __init__(self):
        self.DOF = 1

        self.k_mat = np.asmatrix(np.zeros((self.DOF, self.DOF)))
        self.b_mat = np.asmatrix(np.zeros((self.DOF, self.DOF)))

        self.ff_tau_mat = np.asmatrix(np.zeros((self.DOF, 1)))
        self.fb_tau_mat = np.asmatrix(np.zeros((self.DOF, 1)))

        self.q    = np.asmatrix(np.zeros((self.DOF, 1)))
        self.q_d  = np.asmatrix(np.zeros((self.DOF, 1)))
        self.dq   = np.asmatrix(np.zeros((self.DOF, 1)))
        self.dq_d = np.asmatrix(np.zeros((self.DOF, 1)))

        self.a = ADA_A
        self.b = ADA_B
        self.k = ADA_K

    # ── 内部辅助方法（原封不动
    def _gen_pos_error(self):
        return self.q - self.q_d

    def _gen_vel_error(self):
        return self.dq - self.dq_d

    def _gen_track_error(self):
        return self.k * self._gen_vel_error() + self._gen_pos_error()

    def _gen_ad_factor(self):
        te = self._gen_track_error()
        return self.a / (1.0 + self.b * float(la.norm(te))**2)

    def _calc_tau_fb(self):
        self.fb_tau_mat = (self.k_mat * self._gen_pos_error()
                           + self.b_mat * self._gen_vel_error())
        return self.fb_tau_mat

    def _calc_tau_ff(self):
        self.ff_tau_mat = self._gen_ad_factor() * self._gen_track_error()
        return self.ff_tau_mat

    # ── 对外接口 ──────────────────────────────────────────────
    def update_impedance(self, q, q_d, dq, dq_d):
        """更新阻抗矩阵，返回 (K_scalar, B_scalar)"""
        self.q   = np.asmatrix([[float(q)]])
        self.q_d = np.asmatrix([[float(q_d)]])
        self.dq  = np.asmatrix([[float(dq)]])
        self.dq_d = np.asmatrix([[float(dq_d)]])

        ad = self._gen_ad_factor()
        te = self._gen_track_error()
        pe = self._gen_pos_error()
        ve = self._gen_vel_error()

        self.k_mat = np.clip(te * pe.T / ad, -100.0, 100.0)
        self.b_mat = np.clip(te * ve.T / ad, -100.0, 100.0)

        return float(self.k_mat[0, 0]), float(self.b_mat[0, 0])

    def get_tau(self, q, q_d, dq, dq_d) -> float:
        """
        一步计算：先更新阻抗矩阵，再返回总扭矩（反馈 + 前馈）。
        返回裁剪前的原始扭矩，外层再做限幅。
        """
        self.update_impedance(q, q_d, dq, dq_d)
        tau_mat = self._calc_tau_fb() + self._calc_tau_ff()
        return float(tau_mat[0, 0])

    @property
    def K(self) -> float:
        return float(self.k_mat[0, 0])

    @property
    def B(self) -> float:
        return float(self.b_mat[0, 0])



#  ILC  （原始简单版本，fixed-length array）

class ILC:
    """
    迭代学习控制器（AAN only）。
    - reference_length = TRIAL_DURATION_S × SAMPLE_RATE = 6000
    - Trial N运行时通过 get_feedforward() 按步顺序取值（ff_iterator递增）
    - Trial N结束后调用 update_learning(error_array) 更新学习，供Trial N+1使用
    - Trial 1时 learned_feedforward为空，前馈全为0
    """
    def __init__(self,
                 max_trials: int = 10,
                 trial_duration: float = TRIAL_DURATION_S,
                 frequency: float = SAMPLE_RATE,
                 lr: float = ILC_LR):
        self.max_trials      = max_trials
        self.current_trial   = 0
        self.learned_feedforward: list[np.ndarray] = []
        self.trial_duration  = trial_duration
        self.frequency       = frequency
        self.error_history: list[np.ndarray] = []
        self.lr              = lr
        self.ff_iterator     = 0

    # ── 每次trial开始前调用，重置迭代器 ──────────────────────
    def reset_iterator(self):
        self.ff_iterator = 0

    # ── 控制主循环内：按步顺序取前馈值 ───────────────────────
    def get_feedforward(self) -> float:
        """
        返回当前步的ILC前馈扭矩（Nm），并将迭代器前进一步。
        若尚无学习数据（Trial 1），返回 0.0。
        若迭代器越界，返回最后一个值。
        """
        if not self.learned_feedforward:
            return 0.0
        ff = self.learned_feedforward[-1]
        idx = min(self.ff_iterator, len(ff) - 1)
        self.ff_iterator += 1
        return float(ff[idx])

    # ── 每次trial结束后调用 ───────────────────────────────────
    def update_learning(self, error_array: list | np.ndarray) -> np.ndarray:
        """
        根据本次trial的误差序列更新前馈数组。
        error_array: 长度不定，会自动裁剪或补零对齐到 expected_len。
        返回新的前馈数组（同时存入 learned_feedforward）。
        """
        # 重置迭代器，下次trial开始时会再次调用 reset_iterator
        self.ff_iterator = 0

        expected_len = int(self.trial_duration * self.frequency)

        if len(error_array) == 0:
            print("[ILC] Warning: 空误差数组，跳过本次更新")
            ff_zero = np.zeros(expected_len)
            self.learned_feedforward.append(ff_zero)
            return ff_zero

        error_array = np.array(error_array, dtype=np.float64)

        # 对齐长度
        len_diff = expected_len - len(error_array)
        if len_diff > 0:
            # 短于期望：用上一次历史末段填充，若无历史则补零
            if self.error_history:
                pad = self.error_history[-1][-len_diff:]
            else:
                pad = np.zeros(len_diff)
            error_array = np.concatenate([error_array, pad])
        elif len_diff < 0:
            # 长于期望：截断
            error_array = error_array[:expected_len]

        assert len(error_array) == expected_len, \
            f"[ILC] 长度对齐失败: {len(error_array)} != {expected_len}"

        # 学习更新
        if not self.learned_feedforward:
            ff = np.zeros_like(error_array)
        else:
            ff = self.learned_feedforward[-1] + self.lr * error_array

        self.learned_feedforward.append(ff)
        self.error_history.append(error_array)
        self.current_trial += 1

        avg_error = np.mean(np.abs(error_array))
        max_error = np.max(np.abs(error_array))
        print(f"[ILC] Trial {self.current_trial} 学习完成:")
        print(f"      学习率:   {self.lr}")
        print(f"      平均误差: {math.degrees(avg_error):.3f}°")
        print(f"      最大误差: {math.degrees(max_error):.3f}°")
        print(f"      前馈范围: [{ff.min():.4f}, {ff.max():.4f}] Nm")

        return ff


# ══════════════════════════════════════════════════════════════
#  电机接口
# ══════════════════════════════════════════════════════════════
class Motor:
    def __init__(self):
        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        if _SCRIPT_DIR not in sys.path:
            sys.path.insert(0, _SCRIPT_DIR)

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
                theta   = math.radians(deg_abs)
                dtheta  = vel_raw * VEL_UNIT_RAD_S
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
#  EMG 参考源
# ══════════════════════════════════════════════════════════════
class EMGReference:
    """
    从 qd_queue 中读取最新 EMG 目标角度。
    无新数据时 hold-last；数值微分估算 dtheta_d 和 ddtheta_d。
    """
    def __init__(self, qd_queue: queue.Queue):
        self._queue     = qd_queue
        self._theta_d   = float(np.deg2rad(SINE_CENTER_DEG))
        self._dtheta_d  = 0.0
        self._ddtheta_d = 0.0
        self._prev_theta_d  = self._theta_d
        self._prev_dtheta_d = 0.0
        self._vel_alpha = 0.3
        self._acc_alpha = 0.3

    def update(self) -> tuple[float, float, float]:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            new_theta_d = float(latest)
            raw_vel = (new_theta_d - self._prev_theta_d) / DT
            self._dtheta_d = (self._vel_alpha * raw_vel
                              + (1 - self._vel_alpha) * self._dtheta_d)
            raw_acc = (self._dtheta_d - self._prev_dtheta_d) / DT
            self._ddtheta_d = (self._acc_alpha * raw_acc
                               + (1 - self._acc_alpha) * self._ddtheta_d)
            self._prev_theta_d  = new_theta_d
            self._prev_dtheta_d = self._dtheta_d
            self._theta_d       = new_theta_d

        return self._theta_d, self._dtheta_d, self._ddtheta_d

    def current_theta_d(self) -> float:
        return self._theta_d


pt = []


# ══════════════════════════════════════════════════════════════
#  单次试验
# ══════════════════════════════════════════════════════════════
def run_trial(motor: Motor,
              ada: AdaImpCon,
              ilc: ILC,
              qd_queue: queue.Queue,
              trial_num: int,
              duration_s: float) -> dict | None:
    motor.home()
    lowpass_dq = rt_desired_Angle_lowpass(106, lp_cutoff=2)
    ref = EMGReference(qd_queue)

    # ILC 迭代器重置（每次trial从头取前馈）
    ilc.reset_iterator()

    vel_d_max       = float(THETA_RANGE) * 0.5
    JERK_NORM_SCALE = vel_d_max * 50.0

    # 滤波状态
    vel_ctrl = vel_acc = prev_vel = acc_f = prev_acc = 0.0
    jerk_f   = tau_f = 0.0
    ddtheta_buf: list[float] = []

    # 数据记录
    timestamps = []; thetas = []; theta_ds = []
    Ks = []; Bs = []
    jerks = []; torques = []; ilc_ffs = []; rewards = []
    errors_rad: list[float] = []          # 用于 ILC 学习（弧度）
    errors_deg: list[float] = []          # 用于统计打印（度）
    accs = []

    high_tau_count = 0

    print(f"\n[Trial {trial_num}] ▶ 开始，时长={duration_s}s  "
          f"（移动手臂以控制目标角度）  ILC前馈={'有' if ilc.learned_feedforward else '无（Trial 1）'}")
    t_start = t_last = time.time()

    while True:
        now     = time.time()
        elapsed = now - t_start
        if elapsed >= duration_s or stop_event.is_set():
            break

        dt_actual = now - t_last
        pt.append(dt_actual)
        if dt_actual < DT:
            time.sleep(DT - dt_actual)
            continue
        t_last = now

        # ── 参考轨迹：EMG 输入 ─────────────────────────────────
        theta_d, dtheta_d, ddtheta_d = ref.update()
        theta_d = float(lowpass_dq.lowpass(np.atleast_1d(theta_d)))

        dtheta_d_fut = dtheta_d + ddtheta_d * N_LAG * DT

        # ── 读取电机 ──────────────────────────────────────────
        try:
            theta, dtheta = motor.read()
        except RuntimeError as e:
            print(f"[Trial {trial_num}] [WARN] {e}，跳过本步")
            motor.stop()
            time.sleep(0.3)
            continue

        plot_q.append(theta)
        plot_dq.append(theta_d)

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
        # ddtheta_sm 暂留，ada_imp_con 内部已处理前馈，此处不再单独使用
        # ddtheta_sm = float(np.mean(ddtheta_buf))

        # ── ada_imp_con：计算自适应阻抗扭矩 ───────────────────
        # 输入：当前位置、期望位置、当前速度（控制路径滤波）、期望速度（预测）
        tau_ada = ada.get_tau(
            q   = theta,
            q_d = theta_d,
            dq  = vel_ctrl,
            dq_d = dtheta_d_fut,
        )
        tau_ada = float(np.clip(tau_ada, TORQUE_MIN, TORQUE_MAX))

        # ── ILC 前馈叠加 ──────────────────────────────────────
        ilc_ff  = ilc.get_feedforward()
        tau_raw = tau_ada + ilc_ff
        tau_raw = float(np.clip(tau_raw, TORQUE_MIN, TORQUE_MAX))

        # ── 过载保护 ──────────────────────────────────────────
        if abs(tau_raw) > TORQUE_MAX * 0.7:
            high_tau_count += 1
            if high_tau_count > 10:
                tau_raw = float(np.sign(tau_raw) * TORQUE_MAX * 0.5)
        else:
            high_tau_count = 0

        # ── 扭矩低通平滑 ─────────────────────────────────────
        tau_f = TAU_FILTER_ALPHA * tau_f + (1 - TAU_FILTER_ALPHA) * tau_raw
        plot_tau.append(tau_f)
        motor.send(tau_f)

        # ── 奖励 ──────────────────────────────────────────────
        e_c = float(np.clip(e_pos, -math.pi, math.pi))
        r   = (W_TRACK * math.exp(-0.5 * (e_c    / SIGMA_TRACK)**2)
             + W_JERK  * math.exp(-0.5 * (jerk_n / SIGMA_JERK )**2))

        # ── 记录 ──────────────────────────────────────────────
        timestamps.append(elapsed)
        thetas.append(theta)
        theta_ds.append(theta_d)
        Ks.append(ada.K)
        Bs.append(ada.B)
        jerks.append(jerk_f)
        torques.append(tau_f)
        ilc_ffs.append(ilc_ff)
        rewards.append(r)
        errors_rad.append(e_c)                        # 用于ILC（弧度，有符号）
        errors_deg.append(abs(math.degrees(e_c)))     # 用于统计（度，绝对值）
        accs.append(acc_f)

    print(f"Processing time per step: mean={np.mean(pt)*1000:.2f}ms")
    motor.stop()

    if not rewards:
        return None

    # ── ILC 学习更新（本trial结束后执行，供下一trial使用）────
    ilc.update_learning(errors_rad)

    # ── CSV 导出 ──────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"emg_adailc_trial{trial_num:02d}_{ts_str}.csv")

    df = pd.DataFrame({
        "t_s":       timestamps,
        "q_rad":     thetas,
        "q_des_rad": theta_ds,
        "K":         Ks,
        "B":         Bs,
        "jerk":      jerks,
        "tau_Nm":    torques,
        "ilc_ff_Nm": ilc_ffs,
        "reward":    rewards,
    })
    df.to_csv(csv_path, index=False)
    print(f"\n[Trial {trial_num}] 💾 数据已保存: {csv_path}  ({len(df)} 行)")

    result = dict(
        avg_reward = float(np.mean(rewards)),
        track_rmse = float(np.sqrt(np.mean(np.square(errors_deg)))),
        acc_rms    = float(np.sqrt(np.mean(np.square(accs)))),
        max_err    = float(np.max(errors_deg)),
        max_tau    = float(np.max(np.abs(torques))),
        final_K    = ada.K,
        final_B    = ada.B,
        csv_path   = csv_path,
    )
    print(f"\n[Trial {trial_num}] 结果:")
    print(f"    平均奖励   = {result['avg_reward']:.4f}")
    print(f"    跟踪 RMSE  = {result['track_rmse']:.3f}°")
    print(f"    最大误差   = {result['max_err']:.2f}°")
    print(f"    加速度 RMS = {result['acc_rms']:.3f} rad/s²")
    print(f"    最大扭矩   = {result['max_tau']:.2f} Nm")
    print(f"    最终增益   K={result['final_K']:.4f}  B={result['final_B']:.4f}")
    return result


# ══════════════════════════════════════════════════════════════
#  汇总打印
# ══════════════════════════════════════════════════════════════
def print_summary(results: list):
    print(f"\n{'='*55}")
    print(f"  汇总  —  ada_imp_con + ILC (AAN)")
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
#  入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_TRIALS

    print(f"\n{'='*55}")
    print(f"  EMG-ada_imp_con-ILC 硬件运行器")
    print(f"  试验数: {num_trials}  时长: {TRIAL_DURATION_S}s/次")
    print(f"  ILC 参考长度: {ILC_REFERENCE_LEN}  学习率: {ILC_LR}")
    print(f"{'='*55}")

    # 初始化控制器（跨trial共享，ada_imp_con状态保留）
    ada = AdaImpCon()
    ilc = ILC(max_trials=num_trials + 1)

    # 连接电机
    try:
        motor = Motor()
    except Exception as e:
        print(f"\n❌ 电机连接失败: {e}")
        sys.exit(1)
    time.sleep(1.0)

    # 信号处理
    def _on_interrupt(*_):
        print("\n[中断] 正在安全停止...")
        stop_event.set()
        motor.stop()

    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    # 共享队列
    qd_queue = queue.Queue(maxsize=2)

    # 启动 EMG 线程
    emg_thread = threading.Thread(
        target=emg_thread_fn,
        args=(qd_queue,),
        name="EMG-Thread",
        daemon=True
    )
    emg_thread.start()

    print("[Main] 等待 EMG 初始化...")
    while qd_queue.empty() and not stop_event.is_set():
        time.sleep(0.1)
    print("[Main] EMG 就绪，开始试验。")

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
            motor      = motor,
            ada        = ada,
            ilc        = ilc,
            qd_queue   = qd_queue,
            trial_num  = i + 1,
            duration_s = TRIAL_DURATION_S,
        )
        if res:
            all_results.append(res)

        print(f"length of qd {len(plot_dq)}, length of q {len(plot_q)}, "
              f"operational frequency: {len(plot_q)/TRIAL_DURATION_S:.2f} Hz")
        t_qd = np.arange(len(plot_dq)) * DT
        t_q  = np.arange(len(plot_q))  * DT

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t_qd, plot_dq, label='θ_d (rad)')
        plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)')
        plt.subplot(1, 2, 2)
        plt.plot(t_q, plot_q, label='θ (rad)')
        plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)')
        plt.tight_layout()
        plt.show()

        data = pd.DataFrame({
            "t_qd":   t_qd,
            "qd_rad": plot_dq,
            "t_q":    t_q,
            "q_rad":  plot_q,
            "tau":    plot_tau,
        })
        save_dir = f"Outputs/RWExosuitResults/{USER_NAME}/3"
        os.makedirs(save_dir, exist_ok=True)
        data.to_csv(f"{save_dir}/trial_{i+1}.csv", index=False)

        plot_dq.clear()
        plot_q.clear()
        plot_tau.clear()

    # 停止
    stop_event.set()
    motor.stop()
    emg_thread.join(timeout=3.0)

    if all_results:
        print_summary(all_results)

    print("程序结束，电机已安全停止。")