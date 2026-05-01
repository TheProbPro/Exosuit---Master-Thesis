"""
PKL Hardware Runner  —  PILCO+GP -> OIAC
=========================================
功能：加载 PKL 策略文件，连接真实 Dynamixel 电机，执行正弦轨迹跟踪。
数据记录：K, B, Kff, jerk, torque, theta, theta_d, reward（每步）-> CSV

用法：
    python run_pkl_hardware.py                          # 自动找 policy_final.pkl
    python run_pkl_hardware.py policy_final.pkl         # 指定 PKL 文件
    python run_pkl_hardware.py policy_best_ep010.pkl 5  # 指定文件 + 试验次数
"""

import os, sys, math, time, pickle, signal, atexit, threading
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  参数配置（与训练脚本保持一致）
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 200
DT          = 1.0 / SAMPLE_RATE
TORQUE_MAX  = 10.1
TORQUE_MIN  = -TORQUE_MAX

SINE_AMP_DEG    = 55.0
SINE_CENTER_DEG = 70.0
SINE_FREQ_HZ    = 0.04
SINE_AMP        = math.radians(SINE_AMP_DEG)

K_MIN,   K_MAX   = 5.0,  25.0
B_MIN,   B_MAX   = 0.5,  3.0
KFF_MIN, KFF_MAX = 0.0,  3.0

DELTA_MAX   = 0.03
STATE_DIM   = 7
CONTROL_DIM = 3
STATE_SCALE = np.ones(STATE_DIM, dtype=np.float64)

W_TRACK,  SIGMA_TRACK = 1.0, 0.05
W_JERK,   SIGMA_JERK  = 0.5, 0.3

VEL_FILTER_ALPHA_CTRL = 0.5
VEL_FILTER_ALPHA_ACC  = 0.92
ACC_FILTER_ALPHA      = 0.70
TAU_FILTER_ALPHA      = 0.01
DDTHETA_SMOOTH_N      = 5
N_LAG                 = 3

MOTOR_PORT       = 'COM4'
MOTOR_BAUD       = 4_500_000
TORQUE_DIRECTION = 1

TRIAL_DURATION_S = 30.0
NUM_TRIALS       = 3

# 标定参数
RAW_MIN         = -15427
RAW_MAX         = -2922
ANGLE_RANGE_DEG = 140.0
RAW_RANGE       = RAW_MAX - RAW_MIN
VEL_UNIT_RAD_S  = 0.229 * 2.0 * math.pi / 60.0

# CSV 输出目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial_logs")


# ─────────────────────────────────────────────────────────────
#  正弦参考轨迹
# ─────────────────────────────────────────────────────────────
class SineReference:
    _omega = 2.0 * math.pi * SINE_FREQ_HZ

    def at(self, t: float):
        o = self._omega
        return (SINE_AMP * math.sin(o * t),
                SINE_AMP * o * math.cos(o * t),
               -SINE_AMP * o * o * math.sin(o * t))


# ─────────────────────────────────────────────────────────────
#  OIAC 控制器
# ─────────────────────────────────────────────────────────────
class OIAC:
    @staticmethod
    def _to_n(v, lo, hi): return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
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
            float(np.clip(n[2], 0, 1)))

    def torque(self, theta, theta_d, dtheta_f, dtheta_d, ddtheta_d, B_w=1.0):
        tau = (self.K * (theta_d - theta)
             + self.B * B_w * (dtheta_d - dtheta_f)
             + self.Kff * ddtheta_d)
        return float(np.clip(tau, TORQUE_MIN, TORQUE_MAX))


# ─────────────────────────────────────────────────────────────
#  纯 NumPy 线性策略（从 PKL 加载）
# ─────────────────────────────────────────────────────────────
class LinearPolicyNumpy:
    def __init__(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        self.W   = d['W'].astype(np.float64)
        self.b   = d['b'].astype(np.float64)
        self.tag = d.get('tag', 'unknown')
        self.ts  = d.get('timestamp', 0)
        self.state_dim = self.W.shape[1]   # 从 W 自动推断期望维度
        print(f"  PKL 加载成功: tag={self.tag}  时间={time.ctime(self.ts)}")
        print(f"  W shape={self.W.shape}  b shape={self.b.shape}")
        print(f"  策略期望 state_dim={self.state_dim}  (runner STATE_DIM={STATE_DIM})")
        if self.state_dim != STATE_DIM:
            print(f"  [WARN] 维度不匹配，将自动裁剪/填充 state "
                  f"({STATE_DIM} -> {self.state_dim})")

    def get_action(self, state: np.ndarray) -> np.ndarray:
        # 裁剪或零填充，使 state 维度与 W 匹配
        if len(state) > self.state_dim:
            state = state[:self.state_dim]
        elif len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        sn = (state * STATE_SCALE[:self.state_dim]).reshape(1, -1)
        a  = DELTA_MAX * np.tanh(sn @ self.W.T + self.b).flatten()
        return np.clip(a, -DELTA_MAX, DELTA_MAX)


# ─────────────────────────────────────────────────────────────
#  状态构建
# ─────────────────────────────────────────────────────────────
def build_state(e_pos, e_vel, acc, oiac, jerk_norm=0.0):
    kn = oiac.to_norm()
    return np.array([
        float(np.clip(e_pos,      -math.pi, math.pi)),
        float(np.clip(e_vel,      -5.0,     5.0)),
        float(np.clip(acc,        -5.0,     5.0)),
        kn[0], kn[1], kn[2],
        float(np.clip(jerk_norm,  -1.0,     1.0))
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────
#  电机接口
# ─────────────────────────────────────────────────────────────
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

        print(f"  连接电机: {MOTOR_PORT} @ {MOTOR_BAUD}...")
        self._m   = Motors(port=MOTOR_PORT, baudrate=MOTOR_BAUD)
        if not self._m.motor_ids:
            raise RuntimeError("未找到任何电机！")
        self._mid = self._m.motor_ids[0]
        print(f"  ✅ 电机 ID={self._mid} 连接成功")

        self._set_current_mode()

        raw_now = self._raw_signed()
        self._raw_zero = raw_now - int(SINE_CENTER_DEG / ANGLE_RANGE_DEG * RAW_RANGE)
        deg_now = self._raw_to_deg(raw_now)
        print(f"  零点校准完成: 当前绝对角度={deg_now:.1f}°  "
              f"(centered={deg_now - SINE_CENTER_DEG:.1f}°)")

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
                print("  ✅ 电机已设为电流控制模式")
        except Exception as e:
            print(f"  [WARN] 模式设置失败: {e}")

    def read(self):
        for attempt in range(5):
            try:
                raw     = self._raw_signed()
                vel_raw = float(self._m.get_velocity(motor_id=self._mid))
                if vel_raw > 1023: vel_raw -= 2048
                deg_abs = self._raw_to_deg(raw)
                theta   = math.radians(deg_abs - SINE_CENTER_DEG)
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
                self._mid, self._m.torq2curcom(tau * TORQUE_DIRECTION))
        except Exception as e:
            print(f"  [WARN] 发送扭矩失败: {e}")

    def stop(self):
        for _ in range(3):
            try: self._m.sendMotorCommand(self._mid, 0); time.sleep(0.02)
            except Exception: pass

    def home(self, target_deg=0.0, gain=0.5, tol=1.0, max_steps=500):
        print(f"  归中到 {target_deg:.1f}°...")
        for _ in range(max_steps):
            try:
                theta, _ = self.read()
                err = target_deg - math.degrees(theta)
                self.send(float(np.clip(err * gain, -5.0, 5.0)))
                if abs(err) < tol: break
            except Exception: pass
            time.sleep(0.05)
        self.stop()
        time.sleep(0.8)
        theta, _ = self.read()
        print(f"  归中完成: {math.degrees(theta):.2f}° (centered)")


# ─────────────────────────────────────────────────────────────
#  单次试验
# ─────────────────────────────────────────────────────────────
def run_trial(motor: Motor, policy: LinearPolicyNumpy,
              trial_num: int, duration_s: float,
              stop_event: threading.Event,
              pkl_path: str) -> dict | None:

    motor.home()

    ref  = SineReference()
    oiac = OIAC()

    omega           = 2.0 * math.pi * SINE_FREQ_HZ
    vel_d_max       = SINE_AMP * omega
    JERK_NORM_SCALE = SINE_AMP * omega**3 * 50

    # 滤波状态
    vel_ctrl = vel_acc = prev_vel = acc_f = prev_acc = 0.0
    jerk_f = tau_f = 0.0
    ddtheta_buf = []

    # ── 数据记录数组 ──────────────────────────────────────────
    timestamps = []
    thetas     = []   # q       实际关节角 (rad)
    theta_ds   = []   # q_des   期望关节角 (rad)
    Ks         = []
    Bs         = []
    Kffs       = []
    jerks      = []   # 物理 jerk (rad/s³)
    torques    = []   # 滤波后扭矩 (Nm)
    rewards    = []
    errors     = []   # 跟踪误差 (deg)
    accs       = []
    # ─────────────────────────────────────────────────────────

    high_tau_count = 0

    print(f"\n  ▶ 试验 {trial_num} 开始，时长={duration_s}s")
    t_start = t_last = time.time()

    while True:
        now     = time.time()
        elapsed = now - t_start
        if elapsed >= duration_s or stop_event.is_set():
            break

        dt = now - t_last
        if dt < DT:
            time.sleep(DT - dt)
            continue
        t_last = now

        # 参考轨迹
        theta_d, dtheta_d, ddtheta_d = ref.at(elapsed)
        _, dtheta_d_fut, _           = ref.at(elapsed + N_LAG * DT)

        # 读取电机
        try:
            theta, dtheta = motor.read()
        except RuntimeError as e:
            print(f"  [WARN] {e}，跳过本步")
            motor.stop(); time.sleep(0.3)
            continue

        e_pos = theta_d - theta

        # 速度滤波（双路）
        vel_ctrl = VEL_FILTER_ALPHA_CTRL * vel_ctrl + (1 - VEL_FILTER_ALPHA_CTRL) * dtheta
        vel_acc  = VEL_FILTER_ALPHA_ACC  * vel_acc  + (1 - VEL_FILTER_ALPHA_ACC)  * dtheta

        # 加速度
        acc_raw  = (vel_acc - prev_vel) / DT
        acc_f    = ACC_FILTER_ALPHA * acc_f + (1 - ACC_FILTER_ALPHA) * acc_raw
        acc_f    = float(np.clip(acc_f, -10.0, 10.0))
        prev_vel = vel_acc

        # Jerk
        jerk_f   = 0.5 * jerk_f + 0.5 * (acc_f - prev_acc) / DT
        jerk_f   = float(np.clip(jerk_f, -5.0, 5.0))
        jerk_n   = float(np.clip(jerk_f / JERK_NORM_SCALE, -1.0, 1.0))
        prev_acc = acc_f

        # 前馈加速度平滑
        ddtheta_buf.append(ddtheta_d)
        if len(ddtheta_buf) > DDTHETA_SMOOTH_N: ddtheta_buf.pop(0)
        ddtheta_sm = float(np.mean(ddtheta_buf))

        # 策略动作（自动适配维度）
        state  = build_state(e_pos, dtheta_d_fut - vel_ctrl, acc_f, oiac, jerk_n)
        action = policy.get_action(state)
        oiac.apply_delta(action)

        # 动态阻尼
        B_w = float(np.clip(1.0 - 0.7 * (abs(dtheta_d) / vel_d_max)**2, 0.3, 1.0))

        # 扭矩计算 + 低通滤波
        tau_raw = oiac.torque(theta, theta_d, vel_ctrl, dtheta_d, ddtheta_sm, B_w)
        tau_raw = float(np.clip(tau_raw, TORQUE_MIN, TORQUE_MAX))

        # 过载保护
        if abs(tau_raw) > TORQUE_MAX * 0.7:
            high_tau_count += 1
            if high_tau_count > 10:
                tau_raw = float(np.sign(tau_raw) * TORQUE_MAX * 0.5)
        else:
            high_tau_count = 0

        tau_f = TAU_FILTER_ALPHA * tau_f + (1 - TAU_FILTER_ALPHA) * tau_raw
        motor.send(tau_f)

        # 奖励
        e_c = float(np.clip(e_pos, -math.pi, math.pi))
        r   = (W_TRACK * math.exp(-0.5 * (e_c / SIGMA_TRACK)**2)
             + W_JERK  * math.exp(-0.5 * (jerk_n / SIGMA_JERK)**2))

        # ── 记录每步数据 ──────────────────────────────────────
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
        # ─────────────────────────────────────────────────────

        # 实时打印（每秒一次）
        if len(rewards) % SAMPLE_RATE == 0:
            print(f"    t={elapsed:.0f}s  "
                  f"err={math.degrees(e_c):+.1f}°  "
                  f"K={oiac.K:.1f}  B={oiac.B:.3f}  Kff={oiac.Kff:.2f}  "
                  f"tau={tau_f:.2f}Nm  jerk={jerk_f:.3f}  r={r:.4f}")

    motor.stop()

    if not rewards:
        return None

    # ── CSV 导出 ──────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_stem = os.path.splitext(os.path.basename(pkl_path))[0]
    csv_path = os.path.join(LOG_DIR, f"{pkl_stem}_trial{trial_num:02d}_{ts_str}.csv")

    df = pd.DataFrame({
        "t_s":       timestamps,
        "q_rad":     thetas,       # 实际关节角
        "q_des_rad": theta_ds,     # 期望关节角
        "K":         Ks,
        "B":         Bs,
        "Kff":       Kffs,
        "jerk":      jerks,        # rad/s³（物理量）
        "tau_Nm":    torques,      # 滤波后扭矩
        "reward":    rewards,
    })
    df.to_csv(csv_path, index=False)
    print(f"\n  💾 数据已保存: {csv_path}  ({len(df)} 行)")
    # ─────────────────────────────────────────────────────────

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
    print(f"\n  ■ 试验 {trial_num} 结果:")
    print(f"    平均奖励   = {result['avg_reward']:.4f}")
    print(f"    跟踪 RMSE  = {result['track_rmse']:.3f}°")
    print(f"    最大误差   = {result['max_err']:.2f}°")
    print(f"    加速度 RMS = {result['acc_rms']:.3f} rad/s²")
    print(f"    最大扭矩   = {result['max_tau']:.2f} Nm")
    print(f"    最终增益   K={result['final_K']:.2f}  "
          f"B={result['final_B']:.3f}  Kff={result['final_Kff']:.2f}")
    return result


# ─────────────────────────────────────────────────────────────
#  汇总打印
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────
def _find_pkl():
    policy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pilco_oiac_policies")
    for name in ["policy_final.pkl", "policy_latest.pkl"]:
        p = os.path.join(policy_dir, name)
        if os.path.exists(p): return p
    import glob
    found = sorted(glob.glob(os.path.join(policy_dir, "*.pkl")))
    return found[-1] if found else None


if __name__ == "__main__":
    pkl_path   = sys.argv[1] if len(sys.argv) > 1 else _find_pkl()
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_TRIALS

    if not pkl_path or not os.path.exists(pkl_path):
        print(f"❌ 找不到 PKL 文件: {pkl_path}")
        print("用法: python run_pkl_hardware.py [policy.pkl] [num_trials]")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  PILCO-OIAC 硬件运行器")
    print(f"  PKL:   {pkl_path}")
    print(f"  试验数: {num_trials}  时长: {TRIAL_DURATION_S}s/次")
    print(f"{'='*55}")

    policy = LinearPolicyNumpy(pkl_path)

    try:
        motor = Motor()
    except Exception as e:
        print(f"\n❌ 电机连接失败: {e}")
        sys.exit(1)

    stop_event = threading.Event()
    def _on_interrupt(*_):
        print("\n[中断] 正在安全停止...")
        stop_event.set()
        motor.stop()
    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    all_results = []
    for i in range(num_trials):
        if stop_event.is_set(): break
        print(f"\n  按 Enter 开始第 {i+1}/{num_trials} 次试验（Ctrl+C 退出）...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break
        res = run_trial(motor, policy, trial_num=i+1,
                        duration_s=TRIAL_DURATION_S,
                        stop_event=stop_event,
                        pkl_path=pkl_path)
        if res: all_results.append(res)

    if all_results:
        print_summary(all_results, pkl_path)

    motor.stop()
    print("程序结束，电机已安全停止。")