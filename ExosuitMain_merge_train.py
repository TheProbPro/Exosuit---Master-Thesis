"""
PILCO + GP  ->  OIAC Parameter Tuning  [EMG-driven v1]
=======================================================
基于 pilco_oiac_hardware_v3.py 与 EMG 控制代码合并。

核心变化：
  - 用 EMG 实时信号替代固定正弦轨迹，作为期望角度 θ_d
  - EMGReference 类封装 qd_queue，对外接口与 SineReference 完全兼容
  - EMG 处理运行在独立线程（EMG_Processing），通过 queue 传递 (θ_d, timestamp)
  - 保留 SineReference 作为 fallback（EMG 未就绪时或仿真模式）
  - 训练流程与 v3 完全相同：随机采集 → GP 训练 → policy 在线优化 → 保存策略
  - 运行已保存策略时同样从 EMG 获取期望角度

使用方式：
  python pilco_oiac_emg_v1.py train_and_run   # 完整训练+运行
  python pilco_oiac_emg_v1.py run [tag]        # 仅运行已保存策略（tag 默认 final）
  python pilco_oiac_emg_v1.py sine             # 用正弦轨迹训练（不需要 EMG 硬件）
"""

import os, sys, time, pickle, csv, math, signal, threading, atexit, queue
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']  = ''
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
from gpflow import set_trainable, Module
from gpflow.utilities import to_default_float
from tensorflow_probability import distributions as tfd
from datetime import datetime

float_type = gpflow.config.default_float()  # float64


# ═══════════════════════════════════════════════════════════════
#  参数配置区
# ═══════════════════════════════════════════════════════════════

# 硬件
SAMPLE_RATE = 20
DT          = 1.0 / SAMPLE_RATE
TORQUE_MAX  = 10.1
TORQUE_MIN  = -TORQUE_MAX
USER_NAME   = 'VictorBNielsen'
MOTOR_PORT  = 'COM5'
MOTOR_BAUD  = 1_000_000
TORQUE_DIRECTION = 1

# EMG 硬件
EMG_FS      = 2000          # EMG 采样率 (Hz)
EMG_DT      = 1.0 / EMG_FS

# EMG 控制参数（来自 optimizer_6）
EMG_B       = 4.0
EMG_K       = np.pi * 10.0 * 2

# 关节角度范围
THETA_MIN_RAD = math.radians(0)
THETA_MAX_RAD = math.radians(140)
# 控制中心（弧度，居中后 0 对应此绝对角度）
SINE_CENTER_DEG = 70.0
SINE_CENTER     = math.radians(SINE_CENTER_DEG)

# Fallback 正弦参数（EMG 不可用时使用）
SINE_AMP_DEG    = 55.0
SINE_FREQ_HZ    = 0.04
SINE_AMP        = math.radians(SINE_AMP_DEG)

# LSTM 路径（EMG 解码用）
LSTM_PATH = "Outputs/models/LSTM/Windowed_LSTM.pth"

# OIAC 参数范围
K_MIN,   K_MAX   = 5.0,  25.0
B_MIN,   B_MAX   = 0.5,  3.0
KFF_MIN, KFF_MAX = 0.0,   3.0

DELTA_MAX       = 0.03
DELTA_RATE_MAX  = 0.01

# 状态维度（7维：e_pos, e_vel, integral, K_n, B_n, Kff_n, dtau_norm）
STATE_DIM   = 7
CONTROL_DIM = 3
STATE_SCALE = np.array([1.0] * 7, dtype=np.float64)

# 奖励权重
W_TRACK,   SIGMA_TRACK       = 1.0,  0.05
W_SMOOTH,  SIGMA_SMOOTH_NORM = 0.5,  0.3
W_TRACK_SMOOTH               = 0.7

# 两阶段切换阈值
SMOOTH_PHASE_THRESHOLD_DEG = 8.0

# 滤波器时间常数
TAU_C_VEL    = 0.04
TAU_C_TORQUE = 0.08
ACTION_SMOOTH_ALPHA = 0.6

# 扭矩变化率归一化基准 (Nm/s)
DTAU_MAX = 20.0

HORIZON            = 40
GP_INDUCED_MAX     = 100
TRIAL_DURATION_S   = 30.0
NUM_COLLECT_TRIALS = 12
NUM_ONLINE_TRIALS  = 40
GP_RETRAIN_EVERY   = 10

DIAG_PRINT_STEPS = 2
DDTHETA_SMOOTH_N = 3
N_LAG            = 3
INTEGRAL_DECAY   = 0.995
QUALITY_THRESHOLD = 0.82

_best_reward_ever  = -np.inf
_best_policy_ever  = None


# ═══════════════════════════════════════════════════════════════
#  EMG 处理线程
# ═══════════════════════════════════════════════════════════════
class EMGProcessingThread:
    """
    启动一个后台线程持续读取 EMG → 计算期望角度 → 写入 qd_queue。
    主线程通过 get_latest() 获取最新的 (θ_d_centered_rad, timestamp)。
    θ_d_centered_rad: 以 SINE_CENTER_DEG 为零点的角度（弧度），范围约 ±SINE_AMP_DEG
    """

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.qd_queue   = queue.Queue(maxsize=5)
        self._thread    = threading.Thread(target=self._run, daemon=True, name="EMGThread")
        self._ready     = threading.Event()   # EMG 初始化完成后 set

    def start(self):
        self._thread.start()
        print("  [EMG] 等待 EMG 初始化...")
        self._ready.wait(timeout=10.0)
        if not self._ready.is_set():
            print("  [EMG] ⚠️  初始化超时，将使用 fallback 正弦轨迹")

    def is_ready(self):
        return self._ready.is_set()

    def get_latest(self):
        """返回 (theta_d_centered_rad, timestamp) 或 None"""
        item = None
        while True:
            try:
                item = self.qd_queue.get_nowait()
            except queue.Empty:
                break
        return item

    def _run(self):
        try:
            from Sensors.EMGSensor import DelsysEMG
            from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
            from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
            from Optimizations import optimizer_6
            import torch, torch.nn as nn

            # --- 加载 LSTM ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                import AdaptiveEmbodiedControlSystems.LSTM as LSTM_module
                model = LSTM_module.LSTMModel(
                    input_size=1, hidden_size=64, output_size=1,
                    num_layers=1, batch_first=True).to(device)
                model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
                model.eval()
                use_lstm = True
            except Exception as e:
                print(f"  [EMG] LSTM 加载失败({e})，跳过 LSTM 预测")
                use_lstm = False

            # --- 滤波器 & 解释器 ---
            filter_bicep  = rt_filtering(EMG_FS, 450, 20, 2)
            filter_tricep = rt_filtering(EMG_FS, 450, 20, 2)
            net_a_lp      = rt_desired_Angle_lowpass(EMG_FS, lp_cutoff=2, order=2)
            interpreter   = PMC(theta_min=THETA_MIN_RAD, theta_max=THETA_MAX_RAD,
                                 user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
            bq = queue.Queue(maxsize=50)
            tq = queue.Queue(maxsize=50)

            # --- 优化器状态 ---
            v_emg = 0.0
            opt_angle = 0.0

            emg = DelsysEMG(channel_range=(0, 1))
            emg.start()
            time.sleep(1.0)
            self._ready.set()
            print("  [EMG] ✅ EMG 就绪")

            while not self.stop_event.is_set():
                reading    = emg.read()
                timestamp  = time.time()

                fb = filter_bicep.bandpass(reading[0])
                ft = filter_tricep.bandpass(reading[1])

                if bq.full(): bq.get()
                if tq.full(): tq.get()
                bq.put(fb); tq.put(ft)

                brms = np.sqrt(np.mean(np.array(list(bq.queue)) ** 2))
                trms = np.sqrt(np.mean(np.array(list(tq.queue)) ** 2))

                fbrms = float(filter_bicep.lowpass(np.atleast_1d(brms))[0])
                ftrms = float(filter_tricep.lowpass(np.atleast_1d(trms))[0])

                activation = interpreter.compute_activation([fbrms, ftrms])
                net_a      = activation[0] - activation[1]
                f_net_a    = float(net_a_lp.lowpass(np.atleast_1d(net_a))[0])

                opt_angle, v_emg, _ = optimizer_6(
                    f_net_a, v_emg, EMG_DT, opt_angle,
                    THETA_MIN_RAD, THETA_MAX_RAD, math.pi,
                    EMG_B, EMG_K)

                if use_lstm:
                    with torch.no_grad():
                        inp = torch.tensor([[opt_angle]], dtype=torch.float32).to(device)
                        predicted = model(inp).item()
                else:
                    predicted = opt_angle

                # 转换为以 SINE_CENTER 为零点的居中角度
                theta_d_centered = predicted - SINE_CENTER

                try:
                    self.qd_queue.put_nowait((theta_d_centered, timestamp))
                except queue.Full:
                    try:
                        self.qd_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.qd_queue.put_nowait((theta_d_centered, timestamp))

            emg.stop()

        except ImportError as e:
            print(f"  [EMG] ⚠️  EMG 模块未找到({e})，EMG 线程退出")
            # 不 set _ready → 主线程超时后使用 fallback
        except Exception as e:
            print(f"  [EMG] 意外错误: {e}")
            import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
#  参考轨迹（EMG 版 & 正弦版，统一接口）
# ═══════════════════════════════════════════════════════════════
class SineReference:
    """Fallback：固定正弦轨迹"""
    def __init__(self):
        self._omega = 2.0 * math.pi * SINE_FREQ_HZ

    def at(self, t: float):
        o   = self._omega
        pos =  SINE_AMP * math.sin(o * t)
        vel =  SINE_AMP * o * math.cos(o * t)
        acc = -SINE_AMP * o * o * math.sin(o * t)
        return pos, vel, acc


class EMGReference:
    """
    从 EMGProcessingThread.qd_queue 读取期望角度，对外接口与 SineReference 一致。
    速度通过有限差分估计，加速度近似为 0（EMG 信号低频，加速度可忽略）。
    fallback: EMG 未就绪时退回 SineReference。
    """
    def __init__(self, emg_thread: EMGProcessingThread):
        self._emg    = emg_thread
        self._sine   = SineReference()
        self._last_pos = 0.0
        self._last_t   = None
        self._last_vel = 0.0
        # 低通滤波系数（对速度估计做平滑）
        self._vel_alpha = math.exp(-DT / 0.1)

    def at(self, t: float):
        """
        t: elapsed time（仅 fallback 时使用）
        返回 (pos_rad, vel_rad_s, acc_rad_s2)，居中坐标系
        """
        if not self._emg.is_ready():
            return self._sine.at(t)

        item = self._emg.get_latest()
        now  = time.time()

        if item is None:
            # 队列空：保持上次值
            return self._last_pos, self._last_vel, 0.0

        theta_d, ts = item
        theta_d = float(np.clip(theta_d, -SINE_AMP, SINE_AMP))  # 限幅

        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 0.001:
                raw_vel = (theta_d - self._last_pos) / dt
                self._last_vel = (self._vel_alpha * self._last_vel
                                  + (1 - self._vel_alpha) * raw_vel)
        self._last_pos = theta_d
        self._last_t   = now
        return theta_d, self._last_vel, 0.0


# ═══════════════════════════════════════════════════════════════
#  OIAC 控制器
# ═══════════════════════════════════════════════════════════════
class OIAC:
    def __init__(self, K0=10.0, B0=1.0, Kff0=2.0):
        self._Kn   = self._to_n(K0,   K_MIN,   K_MAX)
        self._Bn   = self._to_n(B0,   B_MIN,   B_MAX)
        self._Kffn = self._to_n(Kff0, KFF_MIN, KFF_MAX)

    @staticmethod
    def _to_n(v, lo, hi):
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    @staticmethod
    def _from_n(n, lo, hi):
        return lo + float(np.clip(n, 0.0, 1.0)) * (hi - lo)

    @property
    def K(self):   return self._from_n(self._Kn,   K_MIN,   K_MAX)
    @property
    def B(self):   return self._from_n(self._Bn,   B_MIN,   B_MAX)
    @property
    def Kff(self): return self._from_n(self._Kffn, KFF_MIN, KFF_MAX)

    def to_norm(self):
        return np.array([self._Kn, self._Bn, self._Kffn], dtype=np.float64)

    def apply_delta(self, delta: np.ndarray):
        delta_clipped = np.clip(delta, -DELTA_RATE_MAX, DELTA_RATE_MAX)
        n = self.to_norm() + delta_clipped
        self._Kn, self._Bn, self._Kffn = (
            float(np.clip(n[0], 0, 1)),
            float(np.clip(n[1], 0, 1)),
            float(np.clip(n[2], 0, 1)))

    def compute_torque(self, theta, theta_d, dtheta_filtered, dtheta_d, ddtheta_d):
        e_pos = theta_d - theta
        e_vel = dtheta_d - dtheta_filtered
        tau   = self.K * e_pos + self.B * e_vel + self.Kff * ddtheta_d
        return float(np.clip(tau, TORQUE_MIN, TORQUE_MAX))


# ═══════════════════════════════════════════════════════════════
#  状态构建（7维）
# ═══════════════════════════════════════════════════════════════
def build_state(e_pos, e_vel, e_pos_integral, oiac: OIAC,
                dtau_norm: float = 0.0) -> np.ndarray:
    e_pos_c  = float(np.clip(e_pos,          -math.pi, math.pi))
    e_vel_c  = float(np.clip(e_vel,          -5.0,     5.0))
    integ_c  = float(np.clip(e_pos_integral, -1.0,     1.0))
    dtau_c   = float(np.clip(dtau_norm,      -1.0,     1.0))
    kn = oiac.to_norm()
    return np.array([e_pos_c, e_vel_c, integ_c,
                     kn[0], kn[1], kn[2], dtau_c], dtype=np.float64)


def norm_state(s: np.ndarray) -> np.ndarray:
    return s * STATE_SCALE


# ═══════════════════════════════════════════════════════════════
#  线性 Policy
# ═══════════════════════════════════════════════════════════════
class LinearPolicy(Module):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(
            np.zeros((CONTROL_DIM, STATE_DIM), dtype=np.float64),
            dtype=float_type, trainable=True, name="policy_W")
        self.b = tf.Variable(
            np.zeros((1, CONTROL_DIM), dtype=np.float64),
            dtype=float_type, trainable=True, name="policy_b")

    @property
    def trainable_variables(self):
        return [self.W, self.b]

    def compute_action(self, m, s, squash=True):
        M = m @ tf.transpose(self.W) + self.b
        S = self.W @ s @ tf.transpose(self.W)
        V = tf.transpose(self.W)
        if squash:
            tM = tf.tanh(M)
            M  = DELTA_MAX * tM
            d  = DELTA_MAX * (1.0 - tf.square(tM))
            D  = tf.linalg.diag(tf.reshape(d, [-1]))
            S  = D @ S @ tf.transpose(D)
            V  = V @ tf.transpose(D)
        return M, S, V

    def get_action(self, state_raw: np.ndarray, noise: float = 0.0) -> np.ndarray:
        sn = norm_state(state_raw).reshape(1, -1)
        m  = tf.constant(sn, dtype=float_type)
        s  = tf.zeros([STATE_DIM, STATE_DIM], dtype=float_type)
        M, _, _ = self.compute_action(m, s, squash=True)
        a = M.numpy().flatten()
        if noise > 0:
            a += noise * np.random.randn(CONTROL_DIM)
        return np.clip(a, -DELTA_MAX, DELTA_MAX)

    def randomize(self, scale=0.05):
        self.W.assign(np.random.randn(CONTROL_DIM, STATE_DIM) * scale)
        self.b.assign(np.zeros((1, CONTROL_DIM)))

    def save(self):
        return {'W': self.W.numpy().copy(), 'b': self.b.numpy().copy()}

    def load(self, d):
        W_val = np.array(d['W'], dtype=np.float64)
        b_val = np.array(d['b'], dtype=np.float64)
        if not np.all(np.isfinite(W_val)) or not np.all(np.isfinite(b_val)):
            print("  [Policy.load] 检测到 NaN/Inf，跳过加载")
            return
        self.W.assign(W_val)
        self.b.assign(b_val)


# ═══════════════════════════════════════════════════════════════
#  奖励函数
# ═══════════════════════════════════════════════════════════════
class TrackAndSmoothReward(Module):
    def __init__(self):
        super().__init__()
        self.tracking_phase = True

    def compute_reward(self, m, s):
        e_pos  = m[0, 0]
        dtau_n = m[0, 6]

        valid     = tf.math.is_finite(e_pos)
        e_safe    = tf.where(valid, e_pos,  tf.constant(10.0, dtype=float_type))
        dtau_safe = tf.where(tf.math.is_finite(dtau_n), dtau_n,
                             tf.constant(1.0, dtype=float_type))

        r_track  = tf.exp(-0.5 * tf.square(e_safe   / SIGMA_TRACK))
        r_smooth = tf.exp(-0.5 * tf.square(dtau_safe / SIGMA_SMOOTH_NORM))

        w_track = tf.cast(W_TRACK if self.tracking_phase else W_TRACK_SMOOTH, float_type)
        w_s     = tf.cast(0.0 if self.tracking_phase else W_SMOOTH, float_type)

        r = w_track * r_track + w_s * r_smooth
        r = tf.where(valid, r, tf.zeros_like(r))
        return r, tf.constant([[1e-6]], dtype=float_type)


# ═══════════════════════════════════════════════════════════════
#  稀疏多输出 GP
# ═══════════════════════════════════════════════════════════════
class SMGPR(Module):
    def __init__(self, data, n_induced: int, name=None):
        super().__init__(name)
        X, Y = data
        self.n_out     = Y.shape[1]
        self.n_in      = X.shape[1]
        self.n_induced = n_induced
        self._build(data)

    def _build(self, data):
        X, Y = data
        ls0 = np.full(self.n_in, 0.3, dtype=np.float64)
        ls0[STATE_DIM:] = 0.2
        self.models = []
        for i in range(self.n_out):
            k = gpflow.kernels.SquaredExponential(
                lengthscales=tf.constant(ls0.copy(), dtype=float_type))
            k.lengthscales.prior = tfd.Gamma(
                to_default_float(2.0), to_default_float(2.0))
            k.variance.prior = tfd.Gamma(
                to_default_float(1.5), to_default_float(3.0))
            k.variance.assign(tf.constant(0.5, dtype=float_type))
            Z = np.linspace(X.min(0), X.max(0), self.n_induced)
            m = gpflow.models.GPRFITC(
                (X, Y[:, i:i+1]), kernel=k, inducing_variable=Z)
            m.likelihood.variance.assign(1e-2)
            self.models.append(m)

    def set_data(self, data):
        self._build(data)

    def optimize(self, restarts: int = 2):
        opt = gpflow.optimizers.Scipy()
        for idx, m in enumerate(self.models):
            best_snap = self._snapshot(m)
            best_loss = np.inf
            for attempt in range(1 + restarts):
                if attempt > 0 and best_snap is not None:
                    self._restore(m, best_snap)
                    try:
                        ls = m.kernel.lengthscales.numpy().copy()
                        m.kernel.lengthscales.assign(
                            np.clip(np.abs(ls + np.random.randn(self.n_in) * 0.15),
                                    0.05, 2.0))
                    except Exception:
                        pass
                try:
                    opt.minimize(m.training_loss, m.trainable_variables,
                                 options={'maxiter': 300})
                except Exception as e:
                    print(f"    [GP warn] output={idx} attempt={attempt}: {e}")
                    continue
                if not self._all_finite(m):
                    if best_snap: self._restore(m, best_snap)
                    continue
                try:
                    loss = float(m.training_loss().numpy())
                except Exception:
                    continue
                if not np.isfinite(loss):
                    if best_snap: self._restore(m, best_snap)
                    continue
                snap = self._snapshot(m)
                if snap and loss < best_loss:
                    best_loss = loss
                    best_snap = snap
            if best_snap:
                self._restore(m, best_snap)
                try:
                    ls = np.clip(m.kernel.lengthscales.numpy(), 0.05, 1.5)
                    m.kernel.lengthscales.assign(ls)
                    kv = float(m.kernel.variance.numpy())
                    m.kernel.variance.assign(float(np.clip(kv, 0.05, 1.0)))
                    nv = float(m.likelihood.variance.numpy())
                    m.likelihood.variance.assign(float(np.clip(nv, 1e-3, 1.0)))
                except Exception:
                    pass

    @staticmethod
    def _snapshot(m):
        try:
            return {p.name: p.numpy().copy() for p in m.trainable_parameters}
        except Exception:
            return None

    @staticmethod
    def _restore(m, snap):
        if snap is None: return
        for p in m.trainable_parameters:
            v = snap.get(p.name)
            if v is not None:
                try: p.assign(v)
                except Exception: pass

    @staticmethod
    def _all_finite(m):
        try:
            for p in m.trainable_parameters:
                if not np.all(np.isfinite(p.numpy())): return False
            return True
        except Exception:
            return False

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self._factorize()
        return self._predict(m, s, iK, beta)

    def _factorize(self):
        eye  = tf.eye(self.n_induced, batch_shape=[self.n_out], dtype=float_type)
        Kmm  = self._K(self.Z) + 1e-6 * eye
        Kmn  = self._K(self.Z, self.X)
        L    = tf.linalg.cholesky(Kmm)
        V    = tf.linalg.triangular_solve(L, Kmn)
        G    = tf.sqrt(1.0 + (self.var[:, None]
               - tf.reduce_sum(tf.square(V), axis=1)) / self.noise[:, None])
        V    = V / G[:, None, :]
        Am   = tf.linalg.cholesky(
                   tf.matmul(V, V, transpose_b=True) + self.noise[:, None, None] * eye)
        At   = tf.matmul(L, Am)
        iAt  = tf.linalg.triangular_solve(At, eye)
        Y_   = tf.transpose(self.Y)[:, :, None]
        beta = tf.linalg.triangular_solve(
                   L,
                   tf.linalg.cholesky_solve(Am, (V / G[:, None, :]) @ Y_),
                   adjoint=True)[:, :, 0]
        iB   = tf.matmul(iAt, iAt, transpose_a=True) * self.noise[:, None, None]
        iK   = tf.linalg.cholesky_solve(L, eye) - iB
        return iK, beta

    def _predict(self, m, s, iK, beta):
        m = tf.where(tf.math.is_finite(m), m, tf.zeros_like(m))
        s = tf.where(tf.math.is_finite(s), s, tf.eye(self.n_in, dtype=float_type) * 1e-4)

        s_  = tf.tile(s[None, None], [self.n_out, self.n_out, 1, 1])
        inp = tf.tile(self._cinp(m)[None], [self.n_out, 1, 1])
        iL  = tf.linalg.diag(1.0 / self.ls)
        iN  = inp @ iL
        B   = iL @ s @ iL + tf.eye(self.n_in, dtype=float_type)
        B   = B + 1e-3 * tf.eye(self.n_in, dtype=float_type)
        LB  = tf.linalg.cholesky(B + 1e-6 * tf.eye(self.n_in, dtype=float_type))
        t   = tf.linalg.cholesky_solve(LB, tf.linalg.matrix_transpose(iN))
        t   = tf.linalg.matrix_transpose(t)
        lb  = tf.exp(-tf.reduce_sum(iN * t, -1) / 2.0) * beta
        tiL = t @ iL
        _, ldB = tf.linalg.slogdet(B)
        c   = self.var * tf.exp(-tf.stop_gradient(ldB) / 2.0)
        M   = (tf.reduce_sum(lb, -1) * c)[:, None]
        V_  = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]
        R   = (s_ @ tf.linalg.diag(
                    1/tf.square(self.ls[None]) + 1/tf.square(self.ls[:, None]))
               + tf.eye(self.n_in, dtype=float_type))
        LR  = tf.linalg.cholesky(R + 1e-6 * tf.eye(self.n_in, dtype=float_type))
        Q   = tf.linalg.cholesky_solve(LR, s_) / 2.0
        X_  =  inp[None] / tf.square(self.ls[:, None, None, :])
        X2_ = -inp[:, None] / tf.square(self.ls[None, :, None, :])
        maha = (-2*tf.matmul(X_ @ Q, X2_, adjoint_b=True)
                + tf.reduce_sum(X_ @ Q * X_, -1)[:, :, :, None]
                + tf.reduce_sum(X2_ @ Q * X2_, -1)[:, :, None, :])
        k   = (tf.math.log(self.var)[:, None]
               - tf.reduce_sum(tf.square(iN), -1) / 2.0)
        Lm  = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S   = (tf.tile(beta[:, None, None, :], [1, self.n_out, 1, 1])
               @ Lm
               @ tf.tile(beta[None, :, :, None], [self.n_out, 1, 1, 1]))[:, :, 0, 0]
        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(Lm)))
        S   = S - tf.linalg.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        _, ldR = tf.linalg.slogdet(R)
        S   = (S * tf.exp(-tf.stop_gradient(ldR) / 2.0)
               + tf.linalg.diag(self.var) - M @ tf.transpose(M))
        M = tf.where(tf.math.is_finite(M), M, tf.zeros_like(M))
        S = tf.where(tf.math.is_finite(S), S, tf.eye(self.n_out, dtype=float_type) * 0.1)
        V_= tf.where(tf.math.is_finite(V_), V_, tf.zeros_like(V_))
        return tf.transpose(M), S, tf.transpose(V_)

    def _cinp(self, m): return self.Z - m
    def _K(self, X1, X2=None):
        return tf.stack([mo.kernel.K(X1, X2) for mo in self.models])

    @property
    def X(self):     return self.models[0].data[0]
    @property
    def Y(self):     return tf.concat([m.data[1] for m in self.models], axis=1)
    @property
    def Z(self):     return self.models[0].inducing_variable.Z
    @property
    def ls(self):    return tf.stack([m.kernel.lengthscales for m in self.models])
    @property
    def var(self):   return tf.stack([m.kernel.variance     for m in self.models])
    @property
    def noise(self): return tf.stack([m.likelihood.variance for m in self.models])


# ═══════════════════════════════════════════════════════════════
#  PILCO
# ═══════════════════════════════════════════════════════════════
class PILCO(gpflow.models.BayesianModel):
    def __init__(self, data, Y_std, n_induced=None, horizon=HORIZON,
                 policy=None, reward=None, m_init=None, S_init=None, name=None):
        super().__init__(name)
        X, Y = data
        ni   = n_induced or min(GP_INDUCED_MAX, max(10, X.shape[0] // 8))
        self.gp      = SMGPR(data, ni)
        self.sd      = Y.shape[1]
        self.horizon = horizon
        self.policy  = policy or LinearPolicy()
        self.reward  = reward or TrackAndSmoothReward()
        self.Y_std   = tf.constant(Y_std.reshape(1, -1), dtype=float_type)
        if m_init is None:
            self.m_init = tf.cast(X[0:1, :self.sd], float_type)
            self.S_init = tf.linalg.diag(
                tf.constant([0.01, 0.05, 0.1, 0.01, 0.01, 0.01, 0.01], dtype=float_type))
        else:
            self.m_init = tf.cast(m_init, float_type)
            self.S_init = tf.cast(S_init, float_type)

        self._persistent_optimizer = tf.optimizers.Adam(learning_rate=0.008)
        self._global_best_r   = -np.inf
        self._global_best_par = self.policy.save()

    @property
    def maximum_log_likelihood_objective(self):
        return -self.training_loss()

    def training_loss(self):
        _, _, reward = self._rollout(self.m_init, self.S_init, self.horizon)
        loss = -reward
        loss = tf.where(tf.math.is_finite(loss), loss,
                        tf.constant(1e6, dtype=float_type))
        return loss

    def train_gp(self, restarts=2):
        print("  [GP] optimizing dynamics model...")
        t0 = time.time()
        self.gp.optimize(restarts=restarts)
        print(f"  [GP] done in {time.time()-t0:.1f}s")
        for i, m in enumerate(self.gp.models):
            ls = np.round(m.kernel.lengthscales.numpy(), 3)
            kv = float(m.kernel.variance.numpy())
            lv = float(m.likelihood.variance.numpy())
            ok = (0.001 < kv < 100.0) and (lv < kv * 10)
            print(f"    GP[{i}] {'✅' if ok else '❌'} ls={ls} kvar={kv:.5f} noise={lv:.6f}")

    def train_policy(self, maxiter=150, restarts=2, warm_start=False) -> float:
        print("\n=== rollout 诊断 ===")
        total_r = self._rollout_debug(self.m_init, self.S_init, min(self.horizon, 10))
        print(f"partial_reward(10步) = {total_r:.4f}")
        print("=== 诊断结束 ===\n")

        for p in self.gp.trainable_parameters:
            set_trainable(p, False)

        if warm_start:
            self.policy.load(self._global_best_par)
            try:
                for i in range(maxiter):
                    with tf.GradientTape() as tape:
                        loss = self.training_loss()
                    grads = tape.gradient(loss, self.policy.trainable_variables)
                    if any(g is None for g in grads): break
                    if any(not tf.reduce_all(tf.math.is_finite(g)) for g in grads): break
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    self._persistent_optimizer.apply_gradients(
                        zip(grads, self.policy.trainable_variables))
                r = float(-self.training_loss().numpy())
            except Exception as e:
                print(f"  [Policy] warm_start error: {e}")
                r = self._global_best_r

            if np.isfinite(r) and r > self._global_best_r:
                snap = self.policy.save()
                if np.all(np.isfinite(snap['W'])) and np.all(np.isfinite(snap['b'])):
                    self._global_best_r   = r
                    self._global_best_par = snap
            else:
                self.policy.load(self._global_best_par)

            for p in self.gp.trainable_parameters:
                set_trainable(p, True)
            return self._global_best_r

        inits = [
            self._global_best_par,
            {'W': np.zeros((CONTROL_DIM, STATE_DIM)), 'b': np.zeros((1, CONTROL_DIM))},
            {'W': np.zeros((CONTROL_DIM, STATE_DIM)), 'b': np.full((1, CONTROL_DIM), -0.3)},
            {'W': np.zeros((CONTROL_DIM, STATE_DIM)), 'b': np.full((1, CONTROL_DIM),  0.3)},
        ]
        for _ in range(restarts):
            inits.append(None)

        for init_idx, init in enumerate(inits):
            if init is None:
                self.policy.randomize()
            else:
                self.policy.load(init)

            opt = tf.optimizers.Adam(learning_rate=0.008)
            try:
                for i in range(maxiter):
                    with tf.GradientTape() as tape:
                        loss = self.training_loss()
                    grads = tape.gradient(loss, self.policy.trainable_variables)
                    if any(g is None for g in grads): break
                    if any(not tf.reduce_all(tf.math.is_finite(g)) for g in grads): break
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    opt.apply_gradients(zip(grads, self.policy.trainable_variables))
                r = float(-self.training_loss().numpy())
            except Exception as e:
                print(f"  [Policy] optimize error (init {init_idx}): {e}")
                r = -np.inf

            if np.isfinite(r) and r > self._global_best_r:
                snap = self.policy.save()
                if np.all(np.isfinite(snap['W'])) and np.all(np.isfinite(snap['b'])):
                    self._global_best_r   = r
                    self._global_best_par = snap
                    self._persistent_optimizer = tf.optimizers.Adam(learning_rate=0.008)
            print(f"  [Policy] init={init_idx} reward={r:.5f} (global_best={self._global_best_r:.5f})")

        self.policy.load(self._global_best_par)
        for p in self.gp.trainable_parameters:
            set_trainable(p, True)
        print(f"  [Policy] 全局最优 reward={self._global_best_r:.5f}")
        return self._global_best_r

    def get_action(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        return self.policy.get_action(state, noise=noise)

    def _rollout(self, m_x, s_x, n):
        loop_vars = [tf.constant(0, tf.int32), m_x, s_x,
                     tf.constant(0.0, float_type)]
        _, m_x, s_x, reward = tf.while_loop(
            lambda j, *_: j < n,
            lambda j, mx, sx, r: (
                j + 1,
                *self._propagate(mx, sx),
                r + tf.cast(self.reward.compute_reward(mx, sx)[0], float_type)),
            loop_vars)
        return m_x, s_x, reward

    def _rollout_debug(self, m_x, s_x, n):
        total_r = 0.0
        mx = m_x.numpy().flatten()
        sx = s_x.numpy()
        for step in range(n):
            r_val = float(self.reward.compute_reward(
                tf.constant(mx.reshape(1, -1), dtype=float_type),
                tf.constant(sx, dtype=float_type))[0].numpy())
            if not np.isfinite(r_val):
                break
            mx_t, sx_t = self._propagate(
                tf.constant(mx.reshape(1, -1), dtype=float_type),
                tf.constant(sx, dtype=float_type))
            mx = mx_t.numpy().flatten()
            sx = sx_t.numpy()
            total_r += r_val
        return total_r

    def _propagate(self, m_x, s_x):
        m_x = tf.where(tf.math.is_finite(m_x), m_x, tf.zeros_like(m_x))
        s_x = tf.where(tf.math.is_finite(s_x), s_x,
                        tf.eye(self.sd, dtype=float_type) * 1e-3)

        m_u, s_u, c_xu = self.policy.compute_action(m_x, s_x)
        m  = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x @ c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x @ c_xu), s_u], axis=1)
        s  = tf.concat([s1, s2], axis=0)

        M_d, S_d, C_d = self.gp.predict_on_noisy_inputs(m, s)

        ys  = self.Y_std
        M_d = M_d * ys
        S_d = S_d * tf.transpose(ys) * ys

        M_x = M_d + m_x

        M_low  = tf.constant([[-3.14, -5.0, -5.0, 0.0, 0.0, 0.0, -1.0]], dtype=float_type)
        M_high = tf.constant([[ 3.14,  5.0,  5.0, 1.0, 1.0, 1.0,  1.0]], dtype=float_type)
        M_x    = tf.clip_by_value(M_x, M_low, M_high)

        S_x = S_d + s_x + s1 @ C_d + tf.matmul(C_d, s1, transpose_a=True, transpose_b=True)
        S_x = (S_x + tf.transpose(S_x)) / 2.0

        diag = tf.linalg.diag_part(S_x)
        eps_psd = tf.maximum(-tf.reduce_min(diag) + 1e-4, 1e-4)
        S_x = S_x + tf.eye(self.sd, dtype=float_type) * eps_psd

        diag = tf.clip_by_value(tf.linalg.diag_part(S_x), 1e-4, 2.0)
        S_x  = tf.linalg.set_diag(S_x, diag)
        S_x  = tf.clip_by_value(S_x, -2.0, 2.0)
        S_x  = S_x + tf.eye(self.sd, dtype=float_type) * 1e-4

        M_x = tf.where(tf.math.is_finite(M_x), M_x, tf.zeros_like(M_x))
        S_x = tf.where(tf.math.is_finite(S_x), S_x,
                        tf.eye(self.sd, dtype=float_type) * 1e-3)

        M_x.set_shape([1, self.sd])
        S_x.set_shape([self.sd, self.sd])
        return M_x, S_x


# ═══════════════════════════════════════════════════════════════
#  CSV 日志
# ═══════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, out_dir=None):
        if out_dir is None:
            out_dir = os.path.join(_SCRIPT_DIR, "pilco_oiac_logs")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ep_file  = os.path.join(out_dir, f"episodes_{ts}.csv")
        self.ts_file  = os.path.join(out_dir, f"timesteps_{ts}.csv")
        self.log_file = os.path.join(out_dir, f"log_{ts}.txt")
        with open(self.ep_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'Trial', 'Mode', 'Avg_Reward',
                'Track_RMSE_deg', 'Tau_RMS', 'DTau_RMS',
                'Avg_K', 'Avg_B', 'Avg_Kff', 'GlobalBestReward'])
        with open(self.ts_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'Trial', 'Step', 'Time_s',
                'Pos_rad', 'DesPos_rad', 'PosErr_rad', 'PosErr_deg',
                'Vel_rad_s', 'DesVel_rad_s',
                'Torque_Nm', 'DTorque_Nm_s',
                'K', 'B', 'Kff', 'Reward', 'EMG_mode'])
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"PILCO-OIAC EMG v1 Log  {datetime.now()}\n{'='*60}\n")

    def log(self, msg, also_print=True):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
        if also_print:
            print(msg)

    def record_episode(self, trial, mode, rewards, errors_deg,
                       torques, dtorques, Ks, Bs, Kffs, global_best_r):
        with open(self.ep_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                trial, mode,
                f"{np.mean(rewards):.6f}",
                f"{np.sqrt(np.mean(np.square(errors_deg))):.4f}",
                f"{np.sqrt(np.mean(np.square(torques))):.4f}",
                f"{np.sqrt(np.mean(np.square(dtorques))):.4f}",
                f"{np.mean(Ks):.3f}", f"{np.mean(Bs):.4f}", f"{np.mean(Kffs):.3f}",
                f"{global_best_r:.5f}"])

    def record_step(self, trial, step, t_s, pos, des_pos, vel, des_vel,
                    torque, dtorque, K, B, Kff, reward, emg_mode='sine'):
        e_rad = pos - des_pos
        with open(self.ts_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                trial, step, f"{t_s:.4f}",
                f"{pos:.6f}", f"{des_pos:.6f}",
                f"{e_rad:.6f}", f"{math.degrees(e_rad):.4f}",
                f"{vel:.6f}", f"{des_vel:.6f}",
                f"{torque:.4f}", f"{dtorque:.4f}",
                f"{K:.3f}", f"{B:.4f}", f"{Kff:.3f}",
                f"{reward:.6f}", emg_mode])


# ═══════════════════════════════════════════════════════════════
#  硬件接口
# ═══════════════════════════════════════════════════════════════
class HardwareInterface:
    def __init__(self, user_name=USER_NAME, debug=True):
        self.debug = debug
        self._high_torque_count = 0

        from Motors.DynamixelHardwareInterface import Motors
        if not getattr(Motors, '_patched', False):
            _orig = Motors.__init__
            def _safe(self_m, *a, **kw):
                self_m.num_motors = 0
                self_m.motor_ids  = []
                _orig(self_m, *a, **kw)
            Motors.__init__ = _safe
            Motors._patched = True

        self._raw_min             = -15427
        self._raw_max             = -2922
        self._angle_range_deg     = 140.0
        self._raw_range           = self._raw_max - self._raw_min
        self._raw_min_corrected   = -13804
        self._vel_unit_to_rad_s   = 0.229 * 2.0 * math.pi / 60.0

        print(f"正在连接电机: 端口={MOTOR_PORT}, 波特率={MOTOR_BAUD}...")
        try:
            self._motor = Motors(port=MOTOR_PORT, baudrate=MOTOR_BAUD)
            if not self._motor.motor_ids:
                raise RuntimeError("未找到任何电机！")
            self._mid = self._motor.motor_ids[0]
            print(f"使用电机ID: {self._mid}")
            self._clear_error()
            self._force_current_control_mode()
            raw_now = int(self._motor.get_position(motor_id=self._mid))
            if raw_now > 2147483647:
                raw_now -= 4294967296
            self._raw_min_corrected = raw_now - int(
                SINE_CENTER_DEG / self._angle_range_deg * self._raw_range)
            print(f"  自动校准: _raw_min_corrected = {self._raw_min_corrected}")
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            raise

        atexit.register(self._atexit_stop)
        print("硬件就绪")

    def _force_current_control_mode(self):
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
            ADDR_TORQUE_ENABLE  = 64
            ADDR_OPERATING_MODE = 11
            PROTOCOL_VERSION    = 2.0
            port_h  = PortHandler(MOTOR_PORT)
            handler = PacketHandler(PROTOCOL_VERSION)
            if not port_h.openPort(): return
            if not port_h.setBaudRate(MOTOR_BAUD): port_h.closePort(); return
            handler.write1ByteTxRx(port_h, self._mid, ADDR_TORQUE_ENABLE, 0); time.sleep(0.1)
            handler.write1ByteTxRx(port_h, self._mid, ADDR_OPERATING_MODE, 0); time.sleep(0.1)
            handler.write1ByteTxRx(port_h, self._mid, ADDR_TORQUE_ENABLE, 1); time.sleep(0.1)
            port_h.closePort()
            print("  ✅ 电机已强制设为电流控制模式")
        except Exception as e:
            print(f"  [WARN] 强制设置电机模式失败: {e}")

    def _clear_error(self):
        try:
            self._motor.sendMotorCommand(self._mid, 0)
            time.sleep(0.1)
        except Exception:
            pass

    def _safe_reconnect(self):
        print("  尝试软件重置电机...")
        try:
            self._motor.sendMotorCommand(self._mid, 0)
            time.sleep(0.5)
            self._clear_error()
            time.sleep(0.5)
        except Exception:
            pass
        for attempt in range(8):
            try:
                raw = self._motor.get_position(motor_id=self._mid)
                return True
            except Exception:
                time.sleep(0.3 * (attempt + 1))
        return False

    def raw_to_deg_abs(self, raw):
        raw_int = int(raw)
        if raw_int > 2147483647:
            raw_int -= 4294967296
        return float((raw_int - self._raw_min_corrected) / self._raw_range * self._angle_range_deg)

    def read(self):
        for attempt in range(5):
            try:
                raw     = self._motor.get_position(motor_id=self._mid)
                vel_raw = self._motor.get_velocity(motor_id=self._mid)
                deg_abs      = self.raw_to_deg_abs(raw)
                deg_centered = deg_abs - SINE_CENTER_DEG
                vel_raw_signed = float(vel_raw)
                if vel_raw_signed > 1023:
                    vel_raw_signed -= 2048
                dtheta = vel_raw_signed * self._vel_unit_to_rad_s
                theta  = math.radians(deg_centered)
                return theta, dtheta
            except Exception as e:
                if self.debug:
                    print(f"[WARN] 读取尝试 {attempt+1}/5 失败: {e}")
                time.sleep(0.15 * (attempt + 1))
        raise RuntimeError("连续5次读取电机失败！")

    def send_torque(self, tau: float):
        t = float(np.clip(tau, TORQUE_MIN, TORQUE_MAX))
        if abs(t) > TORQUE_MAX * 0.7:
            self._high_torque_count += 1
            if self._high_torque_count > 10:
                t = float(np.sign(t) * TORQUE_MAX * 0.5)
        else:
            self._high_torque_count = 0
        t = t * TORQUE_DIRECTION
        try:
            raw     = self._motor.get_position(motor_id=self._mid)
            deg_abs = self.raw_to_deg_abs(raw)
        except Exception:
            deg_abs = SINE_CENTER_DEG
        if deg_abs <= 0.5 and t < 0:
            t = 0.0
        elif deg_abs >= (self._angle_range_deg - 0.5) and t > 0:
            t = 0.0
        try:
            self._motor.sendMotorCommand(self._mid, self._motor.torq2curcom(t))
        except Exception as e:
            if self.debug:
                print(f"[WARN] 发送扭矩失败: {e}")

    def stop(self):
        try: self._motor.sendMotorCommand(self._mid, 0)
        except Exception: pass

    def close(self):
        self.stop()
        try: self._motor.close()
        except Exception: pass

    def _atexit_stop(self):
        try:
            self._motor.sendMotorCommand(self._mid, 0)
            time.sleep(0.05)
            self._motor.sendMotorCommand(self._mid, 0)
        except Exception: pass


# ═══════════════════════════════════════════════════════════════
#  单次试验执行（支持 EMG 或正弦参考）
# ═══════════════════════════════════════════════════════════════
def run_trial(hw: HardwareInterface,
              ref,                      # EMGReference 或 SineReference
              policy_or_none,
              oiac: OIAC,
              logger: CSVLogger,
              trial_num: int,
              mode: str,
              duration_s=TRIAL_DURATION_S,
              exploration_noise=0.0,
              stop_event=None,
              global_best_r=-np.inf) -> dict:

    emg_mode = 'emg' if isinstance(ref, EMGReference) and ref._emg.is_ready() else 'sine'

    try:
        # --- 归中 ---
        print("  归中到中心位置...")
        old_debug = hw.debug; hw.debug = False
        for _ in range(400):
            try:
                theta, _ = hw.read()
                error    = 0.0 - math.degrees(theta)
                torque   = float(np.clip(error * 0.5, -5.0, 5.0))
                hw.send_torque(torque)
                if abs(error) < 1.0: break
            except RuntimeError:
                if not hw._safe_reconnect():
                    input("请手动重插电源和USB后按Enter继续...")
                break
            except Exception: pass
            time.sleep(0.05)
        hw.send_torque(0.0); time.sleep(1.0)
        hw.debug = old_debug

        theta, _ = hw.read()
        print(f"  归中完成: {math.degrees(theta):.1f}° [mode={emg_mode}]")
        hw.send_torque(0.0); time.sleep(0.1)

        # --- 初始化 ---
        t_start        = time.time()
        t_last         = t_start
        vel_for_ctrl   = 0.0
        step           = 0
        ddtheta_d_buf  = []
        e_pos_integral = 0.0
        tau_smooth     = 0.0
        prev_tau_smooth= 0.0
        prev_action    = np.zeros(CONTROL_DIM)
        theta_buf      = []
        t_buf          = []
        state          = np.zeros(STATE_DIM)

        rewards, errors_deg = [], []
        torques, dtorques   = [], []
        Ks, Bs, Kffs        = [], [], []
        raw_rows            = []

        def compute_reward_trial(e_pos_rad, dtau_norm_val, tracking_phase):
            r_track = math.exp(-0.5 * (e_pos_rad / SIGMA_TRACK) ** 2)
            if tracking_phase:
                return W_TRACK * r_track
            r_smooth = math.exp(-0.5 * (dtau_norm_val / SIGMA_SMOOTH_NORM) ** 2)
            return W_TRACK_SMOOTH * r_track + W_SMOOTH * r_smooth

        # --- 主循环 ---
        while True:
            now = time.time()
            if now - t_start >= duration_s: break
            if stop_event and stop_event.is_set(): break

            elapsed = now - t_start
            dt      = now - t_last
            if dt < DT:
                time.sleep(DT - dt); continue
            t_last = now

            # 从参考轨迹获取期望角度
            theta_d, dtheta_d, ddtheta_d = ref.at(elapsed)

            # N_LAG 步预测（EMG 时 ddtheta_d 为 0，直接用 dtheta_d）
            t_future = elapsed + N_LAG * DT
            _, dtheta_d_future, _ = ref.at(t_future)

            # 读取电机
            try:
                theta, dtheta_raw = hw.read()
            except RuntimeError as e:
                logger.log(f"  [WARN] 读取失败: {e}")
                if hw._safe_reconnect():
                    try: theta, dtheta_raw = hw.read()
                    except Exception:
                        hw.send_torque(0.0); time.sleep(0.5); continue
                else:
                    hw.send_torque(0.0); time.sleep(0.5); continue

            # 速度估计
            theta_buf.append(theta); t_buf.append(now)
            if len(theta_buf) > 3: theta_buf.pop(0); t_buf.pop(0)

            if len(theta_buf) >= 3:
                dt01 = t_buf[1] - t_buf[0]; dt12 = t_buf[2] - t_buf[1]
                if dt01 > 0.001 and dt12 > 0.001:
                    v01 = (theta_buf[1] - theta_buf[0]) / dt01
                    v12 = (theta_buf[2] - theta_buf[1]) / dt12
                    dtheta_est = (v01 + 2.0 * v12) / 3.0
                else: dtheta_est = 0.0
            elif len(theta_buf) >= 2 and (t_buf[-1] - t_buf[-2]) > 0.001:
                dtheta_est = (theta_buf[-1] - theta_buf[-2]) / (t_buf[-1] - t_buf[-2])
            else: dtheta_est = 0.0
            dtheta_est = float(np.clip(dtheta_est, -10.0, 10.0))

            alpha_vel    = math.exp(-dt / TAU_C_VEL)
            vel_for_ctrl = alpha_vel * vel_for_ctrl + (1 - alpha_vel) * dtheta_est

            e_pos          = theta_d - theta
            e_pos_integral = INTEGRAL_DECAY * e_pos_integral + e_pos * dt
            e_pos_integral = float(np.clip(e_pos_integral, -1.0, 1.0))
            e_vel          = dtheta_d_future - vel_for_ctrl

            ddtheta_d_buf.append(ddtheta_d)
            if len(ddtheta_d_buf) > DDTHETA_SMOOTH_N: ddtheta_d_buf.pop(0)
            ddtheta_d_smooth = float(np.mean(ddtheta_d_buf))

            # 获取 action
            if policy_or_none is not None:
                raw_action = policy_or_none.get_action(state, noise=exploration_noise)
            else:
                scale = DELTA_MAX * 3 if mode == 'collect' else DELTA_MAX
                raw_action = np.random.uniform(-scale, scale, CONTROL_DIM)

            action      = ACTION_SMOOTH_ALPHA * prev_action + (1.0 - ACTION_SMOOTH_ALPHA) * raw_action
            action      = np.clip(action, -DELTA_MAX, DELTA_MAX)
            prev_action = action.copy()

            oiac.apply_delta(action)
            tau_raw = oiac.compute_torque(theta, theta_d, vel_for_ctrl, dtheta_d, ddtheta_d_smooth)
            tau_raw = float(np.clip(tau_raw, TORQUE_MIN, TORQUE_MAX))

            alpha_tau  = math.exp(-dt / TAU_C_TORQUE)
            tau_smooth = alpha_tau * tau_smooth + (1.0 - alpha_tau) * tau_raw
            tau_smooth = float(np.clip(tau_smooth, TORQUE_MIN, TORQUE_MAX))
            hw.send_torque(tau_smooth)

            dtau           = (tau_smooth - prev_tau_smooth) / max(dt, 1e-3)
            prev_tau_smooth = tau_smooth
            dtau_norm      = float(np.clip(dtau / DTAU_MAX, -1.0, 1.0))

            state = build_state(e_pos, e_vel, e_pos_integral, oiac, dtau_norm)

            tracking_phase = True
            if isinstance(policy_or_none, PILCO):
                tracking_phase = policy_or_none.reward.tracking_phase
            r = compute_reward_trial(e_pos, dtau_norm, tracking_phase)

            if DIAG_PRINT_STEPS > 0 and step % DIAG_PRINT_STEPS == 0:
                print(f"  [DIAG t={elapsed:.1f}s] "
                      f"theta={math.degrees(theta):.1f}deg "
                      f"theta_d={math.degrees(theta_d):.1f}deg "
                      f"e={math.degrees(e_pos):.1f}deg "
                      f"tau={tau_smooth:.3f}Nm K={oiac.K:.1f} [{emg_mode}]")

            logger.record_step(trial_num, step, elapsed,
                               theta, theta_d, vel_for_ctrl, dtheta_d,
                               tau_smooth, dtau, oiac.K, oiac.B, oiac.Kff, r, emg_mode)
            rewards.append(r); errors_deg.append(abs(math.degrees(e_pos)))
            torques.append(tau_smooth); dtorques.append(dtau)
            Ks.append(oiac.K); Bs.append(oiac.B); Kffs.append(oiac.Kff)
            raw_rows.append((state.copy(), action.copy()))
            step += 1

        hw.stop()

        track_rmse = float(np.sqrt(np.mean(np.square(errors_deg)))) if errors_deg else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        dtau_rms   = float(np.sqrt(np.mean(np.square(dtorques)))) if dtorques else 0.0

        logger.record_episode(trial_num, mode, rewards, errors_deg,
                              torques, dtorques, Ks, Bs, Kffs, global_best_r)
        logger.log(f"  Trial {trial_num:03d} [{mode:7s}/{emg_mode}]  "
                   f"reward={avg_reward:.4f}  track={track_rmse:.2f}deg  "
                   f"dtau_rms={dtau_rms:.1f}Nm/s  K={np.mean(Ks):.1f}")

        return dict(avg_reward=avg_reward, track_rmse=track_rmse,
                    dtau_rms=dtau_rms, raw_rows=raw_rows,
                    Ks=Ks, Bs=Bs, Kffs=Kffs,
                    errors_deg=errors_deg, torques=torques, dtorques=dtorques)

    except KeyboardInterrupt:
        print("\n  [安全停止] 检测到CTRL+C")
        return None
    except Exception as e:
        print(f"\n  [异常停止] {e}")
        import traceback; traceback.print_exc()
        return None
    finally:
        hw.send_torque(0.0); time.sleep(0.05); hw.send_torque(0.0)
        print("  [安全停止] 扭矩已清零")


# ═══════════════════════════════════════════════════════════════
#  GP 训练数据构建
# ═══════════════════════════════════════════════════════════════
def build_gp_dataset(all_raw_rows: list):
    Xs, Ys = [], []
    for raw_rows in all_raw_rows:
        for t in range(len(raw_rows) - 1):
            s_t,  a_t = raw_rows[t]
            s_t1, _   = raw_rows[t + 1]
            Xs.append(np.concatenate([norm_state(s_t), a_t]))
            Ys.append(norm_state(s_t1) - norm_state(s_t))
    X = np.array(Xs, dtype=np.float64)
    Y = np.array(Ys, dtype=np.float64)

    Y_std        = np.maximum(Y.std(axis=0), 0.01)
    Y_normalized = np.clip(Y / Y_std, -5.0, 5.0)

    print(f"  [数据] X={X.shape}  Y均值={np.round(Y.mean(0), 4)}  Y_std={np.round(Y_std, 4)}")
    return X, Y_normalized, Y_std


def compute_Y_std(all_raw_rows: list) -> np.ndarray:
    Ys = []
    for raw_rows in all_raw_rows:
        for t in range(len(raw_rows) - 1):
            s_t, _ = raw_rows[t]; s_t1, _ = raw_rows[t + 1]
            Ys.append(norm_state(s_t1) - norm_state(s_t))
    Y = np.array(Ys, dtype=np.float64)
    return np.maximum(Y.std(axis=0), 1e-6)


# ═══════════════════════════════════════════════════════════════
#  Policy 存储
# ═══════════════════════════════════════════════════════════════
POLICY_DIR = os.path.join(_SCRIPT_DIR, "pilco_oiac_policies")

def save_policy(pilco: PILCO, tag: str = "latest"):
    os.makedirs(POLICY_DIR, exist_ok=True)
    path = os.path.join(POLICY_DIR, f"policy_{tag}.pkl")
    with open(path, 'wb') as f:
        pickle.dump({**pilco.policy.save(),
                     'global_best_r': pilco._global_best_r,
                     'timestamp': time.time(), 'tag': tag}, f)
    print(f"  saved -> {path}  (global_best_r={pilco._global_best_r:.4f})")
    return path

def load_policy(tag: str = "final") -> LinearPolicy:
    path = os.path.join(POLICY_DIR, f"policy_{tag}.pkl")
    with open(path, 'rb') as f:
        d = pickle.load(f)
    p = LinearPolicy()
    p.load(d)
    print(f"  loaded <- {path}  ({time.ctime(d['timestamp'])})")
    return p


# ═══════════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════════
def plot_summary(results: list, save_dir=None):
    save_dir = save_dir or POLICY_DIR
    os.makedirs(save_dir, exist_ok=True)
    trials = list(range(1, len(results) + 1))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 9), constrained_layout=True)
    fig.suptitle("PILCO+GP -> OIAC  EMG v1  |  EMG tracking + dtau smoothness",
                 fontsize=11, fontweight='bold')
    axs[0].plot(trials, [r['avg_reward']  for r in results], 'g-o', ms=4)
    axs[0].set_ylabel("Avg Reward"); axs[0].grid(alpha=0.4)
    axs[1].plot(trials, [r['track_rmse'] for r in results], 'r-o', ms=4)
    axs[1].set_ylabel("Track RMSE (deg)"); axs[1].grid(alpha=0.4)
    axs[2].plot(trials, [r.get('dtau_rms', 0) for r in results], 'b-o', ms=4)
    axs[2].set_ylabel("DTorque RMS (Nm/s)"); axs[2].set_xlabel("Trial"); axs[2].grid(alpha=0.4)
    out = os.path.join(save_dir, "summary_emg_v1.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  plot -> {out}")


# ═══════════════════════════════════════════════════════════════
#  主训练流程
# ═══════════════════════════════════════════════════════════════
_hw_global = None

def _emergency_stop(signum, frame):
    global _hw_global
    print("\n[紧急停止] 收到中断信号")
    if _hw_global is not None:
        for _ in range(10):
            try: _hw_global.send_torque(0.0); time.sleep(0.01)
            except Exception: pass
    sys.exit(0)


def train_and_run(use_emg: bool = True):
    global _hw_global
    hw           = None
    emg_thread   = None
    stop_event   = threading.Event()

    try:
        # --- 硬件初始化 ---
        hw = HardwareInterface(USER_NAME, debug=True)
        _hw_global = hw
        signal.signal(signal.SIGINT,  _emergency_stop)
        signal.signal(signal.SIGTERM, _emergency_stop)

        logger = CSVLogger()

        # --- EMG 线程启动 ---
        if use_emg:
            emg_thread = EMGProcessingThread(stop_event)
            emg_thread.start()
            ref = EMGReference(emg_thread)
            logger.log(f"  参考信号: {'EMG' if emg_thread.is_ready() else 'Sine(fallback)'}")
        else:
            ref = SineReference()
            logger.log("  参考信号: Sine (强制)")

        # --- 归中 ---
        print("\n自动归中...")
        for _ in range(600):
            try:
                theta, _ = hw.read()
                error    = 0.0 - math.degrees(theta)
                hw.send_torque(float(np.clip(error * 0.5, -5.0, 5.0)))
                if abs(error) < 1.0: break
            except Exception: pass
            time.sleep(0.05)
        hw.send_torque(0.0); time.sleep(0.3)

        # --- 零点校准 ---
        print("\n自动校准零点...")
        for _ in range(300):
            try:
                raw = int(hw._motor.get_position(motor_id=hw._mid))
                if raw > 2147483647: raw -= 4294967296
                deg = (raw - hw._raw_min_corrected) / hw._raw_range * hw._angle_range_deg
                err = SINE_CENTER_DEG - deg
                hw.send_torque(float(np.clip(err * 0.5, -3.0, 3.0)))
                time.sleep(0.05)
                if abs(err) < 0.3: break
            except Exception: time.sleep(0.1); continue

        hw.send_torque(0.0); time.sleep(0.3)
        raw_now = None
        for _ in range(10):
            try:
                raw_now = int(hw._motor.get_position(motor_id=hw._mid))
                if raw_now > 2147483647: raw_now -= 4294967296
                break
            except Exception: time.sleep(0.5)

        if raw_now is None:
            print("  ❌ 无法读取电机位置"); hw.close(); return

        hw._raw_min_corrected = raw_now - int(SINE_CENTER_DEG / hw._angle_range_deg * hw._raw_range)
        print(f"  零点校准完成，按 Enter 继续训练...")
        input()

        logger.log("=" * 60)
        logger.log("PILCO+GP -> OIAC EMG v1  tracking + dtau smoothness")
        logger.log(f"EMG: b={EMG_B} k={EMG_K:.2f}  LSTM={LSTM_PATH}")
        logger.log(f"Reward: W_TRACK={W_TRACK} W_SMOOTH={W_SMOOTH} SIGMA_TRACK={SIGMA_TRACK}")
        logger.log("=" * 60)

        all_results = []
        all_raw     = []

        # ── Phase 1: 随机数据收集 ──────────────────────────────
        logger.log(f"\n=== Phase 1: random data collection ({NUM_COLLECT_TRIALS} trials) ===")
        init_params = [
            (10, 1.0, 1.0), (15, 1.5, 1.5), (20, 2.0, 2.0),
            (10, 0.8, 0.8), (12, 1.2, 1.2), (18, 1.8, 1.8),
            (8,  0.6, 0.6), (14, 1.4, 1.4), (20, 2.5, 2.5),
            (10, 1.0, 1.0), (15, 1.5, 1.5), (18, 2.0, 2.0),
        ]
        for i in range(NUM_COLLECT_TRIALS):
            if stop_event.is_set(): break
            K0, B0, Kff0 = init_params[i % len(init_params)]
            oiac = OIAC(K0=K0, B0=B0, Kff0=Kff0)
            logger.log(f"\n  Trial {i+1}/{NUM_COLLECT_TRIALS}  K={K0} B={B0} Kff={Kff0}")
            res = run_trial(hw, ref, None, oiac, logger,
                            trial_num=i+1, mode='collect',
                            stop_event=stop_event)
            if res is None: break
            all_results.append(res)
            all_raw.append(res['raw_rows'])

        if len(all_raw) < 2:
            logger.log("Not enough data. Exiting."); hw.close(); return

        # ── Phase 2: GP 训练 ───────────────────────────────────
        logger.log("\n=== Phase 2: GP dynamics training ===")
        X, Y_normalized, Y_std = build_gp_dataset(all_raw)

        policy = LinearPolicy()
        reward = TrackAndSmoothReward()
        reward.tracking_phase = True

        m_init = np.array([[0.05, 0.1, 0.0, 0.5, 0.5, 0.5, 0.0]], dtype=np.float64)
        S_init = np.diag([0.005, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])

        pilco = PILCO(data=(X, Y_normalized), Y_std=Y_std, horizon=HORIZON,
                      policy=policy, reward=reward,
                      m_init=m_init, S_init=S_init)
        pilco.train_gp(restarts=2)

        # ── Phase 3: 初始 policy 训练 ─────────────────────────
        logger.log("\n=== Phase 3: initial policy training ===")
        best_r = pilco.train_policy(maxiter=200, restarts=3)
        save_policy(pilco, tag="init")

        # ── Phase 4: 在线改进 ──────────────────────────────────
        logger.log(f"\n=== Phase 4: online improvement ({NUM_ONLINE_TRIALS} trials) ===")
        best_track           = np.inf
        track_rmse_history   = []

        for trial_idx in range(NUM_ONLINE_TRIALS):
            if stop_event.is_set(): break

            ep_num = NUM_COLLECT_TRIALS + trial_idx + 1
            noise  = max(0.003, 0.04 * math.exp(-0.08 * trial_idx))
            oiac   = OIAC()

            logger.log(f"\n  Trial {ep_num} [pilco]  noise={noise:.3f}  "
                       f"phase={'tracking' if pilco.reward.tracking_phase else 'smooth'}  "
                       f"global_best_r={pilco._global_best_r:.4f}")

            res = run_trial(hw, ref, pilco, oiac, logger,
                            trial_num=ep_num, mode='pilco',
                            exploration_noise=noise,
                            stop_event=stop_event,
                            global_best_r=pilco._global_best_r)
            if res is None: break
            all_results.append(res)
            track_rmse_history.append(res['track_rmse'])

            # 两阶段切换
            if pilco.reward.tracking_phase and len(track_rmse_history) >= 5:
                recent_rmse = np.mean(track_rmse_history[-5:])
                if recent_rmse < SMOOTH_PHASE_THRESHOLD_DEG:
                    pilco.reward.tracking_phase = False
                    logger.log(f"  ★★ 切换到平滑阶段！RMSE={recent_rmse:.2f}deg")
                    pilco.train_policy(maxiter=150, restarts=2, warm_start=False)
                    save_policy(pilco, tag=f"smooth_ep{ep_num:03d}")

            if res['avg_reward'] >= QUALITY_THRESHOLD:
                all_raw.append(res['raw_rows'])

            if res['track_rmse'] < best_track:
                best_track = res['track_rmse']
                save_policy(pilco, tag=f"best_ep{ep_num:03d}")
                logger.log(f"  ** new best track_rmse = {best_track:.2f}deg")

            if (trial_idx + 1) % GP_RETRAIN_EVERY == 0:
                logger.log(f"\n  [retrain] using {len(all_raw)} quality trials")
                best_par_before = pilco.policy.save()
                reward_before   = pilco._global_best_r
                old_gp_snaps    = [SMGPR._snapshot(m) for m in pilco.gp.models]

                X_all, Y_all_norm, Y_all_std = build_gp_dataset(all_raw)
                pilco.gp.set_data((X_all, Y_all_norm))
                pilco.Y_std = tf.constant(Y_all_std.reshape(1, -1), dtype=float_type)
                pilco.train_gp(restarts=2)
                pilco.policy.load(best_par_before)

                try:
                    reward_after_gp = float(-pilco.training_loss().numpy())
                except Exception:
                    reward_after_gp = -np.inf

                if reward_after_gp < reward_before * 0.75:
                    logger.log(f"  [retrain] GP变差，回滚")
                    for m, snap in zip(pilco.gp.models, old_gp_snaps):
                        SMGPR._restore(m, snap)
                    pilco.Y_std = tf.constant(
                        compute_Y_std(all_raw).reshape(1, -1), dtype=float_type)
                else:
                    pilco.train_policy(maxiter=150, restarts=2, warm_start=False)

            elif (trial_idx + 1) % 3 == 0:
                pilco.train_policy(maxiter=80, restarts=0, warm_start=True)

            if (trial_idx + 1) % 10 == 0:
                save_policy(pilco, tag=f"ep{ep_num:03d}")

        # ── Phase 5: 保存与绘图 ────────────────────────────────
        logger.log("\n=== Phase 5: save & plot ===")
        save_policy(pilco, tag="final")
        plot_summary(all_results, save_dir=POLICY_DIR)
        logger.log(f"\nDone.  best_track={best_track:.2f}deg  global_best_r={pilco._global_best_r:.4f}")
        hw.close()
        return pilco

    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n训练异常: {e}")
        import traceback; traceback.print_exc()
    finally:
        stop_event.set()
        if hw is not None:
            print("正在停止电机...")
            for _ in range(5):
                try: hw.send_torque(0.0); time.sleep(0.02)
                except Exception: pass
            hw.close()
        print("电机已安全停止")


# ═══════════════════════════════════════════════════════════════
#  部署已保存的 Policy（用 EMG 驱动）
# ═══════════════════════════════════════════════════════════════
def run_with_policy(tag="final", num_trials=5, use_emg=True):
    p      = load_policy(tag)
    hw     = HardwareInterface(USER_NAME, debug=True)
    logger = CSVLogger()

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    if use_emg:
        emg_thread = EMGProcessingThread(stop_event)
        emg_thread.start()
        ref = EMGReference(emg_thread)
    else:
        ref = SineReference()

    class _Wrap:
        def get_action(self, state, noise=0.0):
            return p.get_action(state, noise=noise)

    results = []
    for i in range(num_trials):
        if stop_event.is_set(): break
        if i == 0:
            print(f"\nRun {i+1}/{num_trials} — 按 Enter 开始...")
            input()
        res = run_trial(hw, ref, _Wrap(), OIAC(), logger,
                        trial_num=i+1, mode='run', stop_event=stop_event)
        if res is None: break
        results.append(res)

    stop_event.set()
    hw.close()

    if results:
        tracks = [r['track_rmse'] for r in results]
        dtaus  = [r.get('dtau_rms', 0) for r in results]
        print(f"\nTrack RMSE: mean={np.mean(tracks):.2f}deg  best={np.min(tracks):.2f}deg")
        print(f"DTorque RMS: mean={np.mean(dtaus):.1f}Nm/s  best={np.min(dtaus):.1f}Nm/s")
        plot_summary(results)
    return results


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== PILCO-OIAC EMG v1 配置确认 ===")
    print(f"EMG_FS          = {EMG_FS} Hz")
    print(f"EMG_B           = {EMG_B}  EMG_K = {EMG_K:.2f}")
    print(f"SINE_CENTER_DEG = {SINE_CENTER_DEG}°")
    print(f"K范围: [{K_MIN}, {K_MAX}]  B范围: [{B_MIN}, {B_MAX}]")
    print(f"TAU_C_TORQUE    = {TAU_C_TORQUE}s")
    print(f"ACTION_SMOOTH   = {ACTION_SMOOTH_ALPHA}")
    print(f"W_SMOOTH        = {W_SMOOTH}  SIGMA_SMOOTH_NORM={SIGMA_SMOOTH_NORM}")
    print(f"SMOOTH_PHASE_THRESHOLD_DEG = {SMOOTH_PHASE_THRESHOLD_DEG}")
    print("===================================")

    mode = sys.argv[1] if len(sys.argv) > 1 else "train_and_run"

    if mode == "train_and_run":
        train_and_run(use_emg=True)
    elif mode == "sine":
        # 不需要 EMG 硬件，用正弦轨迹训练（方便调试）
        train_and_run(use_emg=False)
    elif mode == "run":
        tag = sys.argv[2] if len(sys.argv) > 2 else "final"
        run_with_policy(tag=tag, use_emg=True)
    elif mode == "run_sine":
        tag = sys.argv[2] if len(sys.argv) > 2 else "final"
        run_with_policy(tag=tag, use_emg=False)
    else:
        print("Usage:")
        print("  python pilco_oiac_emg_v1.py train_and_run   # EMG 驱动训练")
        print("  python pilco_oiac_emg_v1.py sine            # 正弦轨迹训练（无需EMG）")
        print("  python pilco_oiac_emg_v1.py run [tag]       # EMG 驱动运行策略")
        print("  python pilco_oiac_emg_v1.py run_sine [tag]  # 正弦轨迹运行策略")