# My local imports (EMG sensor, filtering, interpretors, OIAC)
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Motors.DynamixelHardwareInterface import Motors

# General imports
import numpy as np
import numpy.linalg as la
import queue
import threading
import sys
import signal
import time
import math
from scipy import interpolate
import pickle
import os

SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

# Control parameters
TORQUE_MIN = -4.1  # Nm
TORQUE_MAX = 4.1   # Nm

# ILC parameters
ILC_ENABLED = True
ILC_MAX_TRIALS = 10
ILC_TRIAL_DURATION = 10.0  # seconds per trial
ILC_SAVE_PATH = "ilc_learning_data.pkl"

stop_event = threading.Event()
trial_reset_event = threading.Event()  # 用于标记trial结束

class EMGMuscleForceEstimator:
    """使用EMG信号估计肌肉力"""
    def __init__(self):
        self.bicep_force_history = []
        self.tricep_force_history = []
        self.force_penalty_history = []
        
        # EMG到力的转换系数（需要根据实际情况校准）
        self.emg_to_force_scale = 0.1
        
    def estimate_muscle_forces(self, bicep_rms, tricep_rms):
        """基于EMG RMS值估计肌肉力"""
        bicep_force = bicep_rms * self.emg_to_force_scale
        tricep_force = tricep_rms * self.emg_to_force_scale
        
        bicep_force = max(0, bicep_force)
        tricep_force = max(0, tricep_force)
        
        return bicep_force, tricep_force
    
    def calculate_force_penalty(self, bicep_force, tricep_force, q_error, control_torque):
        """基于估计的肌肉力计算惩罚"""
        error_deg = abs(math.degrees(q_error))
        
        if error_deg < 8.0:
            force_magnitude = bicep_force + tricep_force
            
            torque_force_alignment = 1.0
            if control_torque > 0 and bicep_force > tricep_force:
                torque_force_alignment = 0.5
            elif control_torque < 0 and tricep_force > bicep_force:
                torque_force_alignment = 0.5
            else:
                torque_force_alignment = 2.0
            
            force_penalty = 0.001 * force_magnitude * torque_force_alignment
        else:
            force_penalty = 0.0
        
        self.bicep_force_history.append(bicep_force)
        self.tricep_force_history.append(tricep_force)
        self.force_penalty_history.append(force_penalty)
        
        if len(self.bicep_force_history) > 100:
            self.bicep_force_history.pop(0)
            self.tricep_force_history.pop(0)
            self.force_penalty_history.pop(0)
            
        return force_penalty
    
    def get_force_statistics(self):
        """获取肌肉力统计"""
        if not self.bicep_force_history:
            return 0.0, 0.0, 0.0, 0.0
            
        avg_bicep = np.mean(self.bicep_force_history)
        avg_tricep = np.mean(self.tricep_force_history)
        max_bicep = np.max(self.bicep_force_history)
        max_tricep = np.max(self.tricep_force_history)
        
        return avg_bicep, avg_tricep, max_bicep, max_tricep
    
    def reset_history(self):
        """重置历史数据（用于新trial）"""
        self.bicep_force_history.clear()
        self.tricep_force_history.clear()
        self.force_penalty_history.clear()


# ==================== OIAC Controller (Paper Implementation) ====================

class OnlineImpedanceAdaptationController:
    """
    Online Impedance Adaptation Controller based on:
    Xiong, X., & Fang, C. (2023). An Online Impedance Adaptation Controller 
    for Decoding Skill Intelligence. Biomimetic Intelligence and Robotics, 3(2).
    
    This is the paper's original formulation adapted for hardware control.
    """
    def __init__(self, dof=1):
        self.DOF = dof
        self.k_mat = np.zeros((self.DOF, self.DOF))  # Stiffness matrix
        self.b_mat = np.zeros((self.DOF, self.DOF))  # Damping matrix
        
        # State variables
        self.q = np.zeros((self.DOF, 1))      # Real joint angle
        self.q_d = np.zeros((self.DOF, 1))    # Desired joint angle
        self.dq = np.zeros((self.DOF, 1))     # Real joint velocity
        self.dq_d = np.zeros((self.DOF, 1))   # Desired joint velocity
        
        # OIAC parameters from paper (Eq. 3)
        self.a = 0.04      # Adaptation factor numerator
        self.b = 0.001     # Adaptation factor denominator coefficient
        self.k = 0.5       # Tracking error weight
        
        # Scaling factors (hardware-specific, tune based on experiments)
        # 这些值需要根据实际硬件响应调整
        self.k_scale = 100.0   # Stiffness scaling (降低以适应硬件)
        self.b_scale = 80.0    # Damping scaling (降低以适应硬件)
        
        # Safety limits
        self.k_min = 30.0    # Minimum stiffness
        self.k_max = 150.0   # Maximum stiffness
        self.b_min = 10.0    # Minimum damping
        self.b_max = 60.0    # Maximum damping
        
        # Integral term for steady-state error
        self.integral = 0.0
        self.ki = 5.0
        self.max_integral = 15.0
        
    def gen_pos_err(self):
        """Position error (Eq. 1)"""
        return (self.q - self.q_d)
    
    def gen_vel_err(self):
        """Velocity error (Eq. 1)"""
        return (self.dq - self.dq_d)
    
    def gen_track_err(self):
        """Tracking error (Eq. 3)"""
        return (self.k * self.gen_vel_err() + self.gen_pos_err())
    
    def gen_ad_factor(self):
        """Adaptation scalar (Eq. 3)"""
        track_err_norm = la.norm(self.gen_track_err())
        return self.a / (1.0 + self.b * track_err_norm * track_err_norm)
    
    def update_impedance(self, q, q_d, dq, dq_d, dt=0.002):
        """
        Update stiffness and damping matrices using OIAC algorithm (Eq. 2)
        
        Args:
            q: Current joint position (scalar or array)
            q_d: Desired joint position (scalar or array)
            dq: Current joint velocity (scalar or array)
            dq_d: Desired joint velocity (scalar or array)
            dt: Time step for integral calculation
        
        Returns:
            k_mat: Updated stiffness matrix
            b_mat: Updated damping matrix
            integral: Integral term for steady-state error reduction
        """
        # Convert inputs to column vectors
        self.q = np.atleast_2d(np.atleast_1d(q)).T
        self.q_d = np.atleast_2d(np.atleast_1d(q_d)).T
        self.dq = np.atleast_2d(np.atleast_1d(dq)).T
        self.dq_d = np.atleast_2d(np.atleast_1d(dq_d)).T
        
        # Compute error terms
        track_err = self.gen_track_err()
        pos_err = self.gen_pos_err()
        vel_err = self.gen_vel_err()
        ad_factor = self.gen_ad_factor()
        
        # Update stiffness K and damping B using Eq. (2) - outer product formulation
        # K = k_scale * (track_err @ pos_err.T) / ad_factor
        # B = b_scale * (track_err @ vel_err.T) / ad_factor
        self.k_mat = self.k_scale * (track_err @ pos_err.T) / ad_factor
        self.b_mat = self.b_scale * (track_err @ vel_err.T) / ad_factor
        
        # Apply safety limits (element-wise clipping)
        self.k_mat = np.clip(self.k_mat, self.k_min, self.k_max)
        self.b_mat = np.clip(np.abs(self.b_mat), self.b_min, self.b_max)  # 取绝对值避免负阻尼
        
        # Integral term for steady-state error reduction
        error_scalar = float(pos_err.item())
        self.integral += error_scalar * dt
        
        # Anti-windup: only integrate when error is small
        if abs(error_scalar) < math.radians(2.0):
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        else:
            self.integral *= 0.9  # Decay when error is large
        
        return self.k_mat, self.b_mat, self.integral
    
    def reset(self):
        """重置控制器状态（用于新trial）"""
        self.integral = 0.0
        # Note: K and B matrices are not reset as they adapt online


# ==================== Enhanced ILC Controller ====================

class EnhancedILC:
    """
    增强的迭代学习控制器
    用于重复性任务的前馈学习
    """
    def __init__(self, max_trials=10, reference_length=5000):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []  # 每个trial的前馈
        self.reference_time = None
        self.reference_length = reference_length
        
        # 学习率随trial递减
        self.learning_rates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
        
        # 历史数据记录
        self.trial_errors = []
        self.trial_torques = []
        
    def update_learning(self, time_array, error_array, torque_array):
        """
        ILC学习更新
        
        参数:
            time_array: 时间序列
            error_array: 跟踪误差序列
            torque_array: 控制扭矩序列
        
        返回:
            ff: 更新后的前馈扭矩
        """
        if len(time_array) == 0 or len(error_array) == 0:
            print("[ILC] Warning: Empty data, skipping update")
            return np.zeros(self.reference_length)
        
        # 创建统一的参考时间轴
        if self.reference_time is None:
            max_time = max(time_array) if len(time_array) > 0 else ILC_TRIAL_DURATION
            self.reference_time = np.linspace(0, max_time, self.reference_length)
        
        # 对齐数据到参考时间轴
        try:
            interp_error = interpolate.interp1d(
                time_array, error_array, 
                kind='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            aligned_error = interp_error(self.reference_time)
        except Exception as e:
            print(f"[ILC] Interpolation error: {e}")
            aligned_error = np.zeros_like(self.reference_time)
        
        # Test
        lr = 0.1

        # 学习更新
        if not self.learned_feedforward:
            # 第一次trial，直接用误差初始化
            ff = np.zeros_like(aligned_error)
        else:
            # 使用上一次的前馈 + 学习项
            lr = self.learning_rates[min(self.current_trial, len(self.learning_rates)-1)]
            ff = self.learned_feedforward[-1] + lr * aligned_error
        
        # 限制前馈幅度，避免过大
        ff = np.clip(ff, -20.0, 20.0)
        
        # 平滑处理
        if len(ff) > 10:
            window_size = 11
            ff = np.convolve(ff, np.ones(window_size)/window_size, mode='same')
        
        self.learned_feedforward.append(ff)
        self.trial_errors.append(aligned_error)
        self.trial_torques.append(torque_array)
        self.current_trial += 1
        
        # 计算性能指标
        avg_error = np.mean(np.abs(aligned_error))
        max_error = np.max(np.abs(aligned_error))
        
        print(f"[ILC] Trial {self.current_trial} completed:")
        print(f"      Learning rate: {lr}")
        print(f"      Avg error: {math.degrees(avg_error)}°")
        print(f"      Max error: {math.degrees(max_error)}°")
        print(f"      Feedforward range: [{np.min(ff)}, {np.max(ff)}] Nm")
        
        return ff
    
    def get_feedforward(self, t, trial_idx=-1):
        """
        获取指定时刻的前馈扭矩
        
        参数:
            t: 当前时间
            trial_idx: trial索引，-1表示使用最新的
        
        返回:
            feedforward torque
        """
        if trial_idx < 0:
            trial_idx = len(self.learned_feedforward) - 1
            
        if trial_idx < 0 or trial_idx >= len(self.learned_feedforward):
            return 0.0
        
        if self.reference_time is None:
            return 0.0
            
        # 找到最接近的时间点
        idx = np.argmin(np.abs(self.reference_time - t))
        if idx < len(self.learned_feedforward[trial_idx]):
            return float(self.learned_feedforward[trial_idx][idx])
        return 0.0
    
    def save_learning(self, filepath):
        """保存学习数据"""
        data = {
            'learned_feedforward': self.learned_feedforward,
            'reference_time': self.reference_time,
            'trial_errors': self.trial_errors,
            'current_trial': self.current_trial
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"[ILC] Learning data saved to {filepath}")
        except Exception as e:
            print(f"[ILC] Failed to save: {e}")
    
    def load_learning(self, filepath):
        """加载学习数据"""
        if not os.path.exists(filepath):
            print(f"[ILC] No saved data found at {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.learned_feedforward = data['learned_feedforward']
            self.reference_time = data['reference_time']
            self.trial_errors = data.get('trial_errors', [])
            self.current_trial = data['current_trial']
            print(f"[ILC] Loaded {self.current_trial} trials from {filepath}")
            return True
        except Exception as e:
            print(f"[ILC] Failed to load: {e}")
            return False
    
    def reset(self):
        """重置ILC（开始新的学习过程）"""
        self.learned_feedforward.clear()
        self.trial_errors.clear()
        self.trial_torques.clear()
        self.current_trial = 0
        print("[ILC] Reset completed")


def read_EMG(EMG_sensor, raw_queue):
    """EMG读取线程

    Notes:
    - Use a parameter name that does not shadow the imported `queue` module.
    - Catch the correct exception type `queue.Full` (capital F) instead of
      calling `queue.full()` which returns a bool and causes a TypeError
      when used in an `except` clause.
    """
    while not stop_event.is_set():
        reading = EMG_sensor.read()
        try:
            raw_queue.put_nowait(reading)
        except queue.Full:
            try:
                # discard oldest and try again
                raw_queue.get_nowait()
                raw_queue.put_nowait(reading)
            except queue.Full:
                # still full after discard attempt; skip this sample
                pass
        except Exception as e:
            print(f"[reader] error: {e}", file=sys.stderr)


def send_motor_command(motor, command_queue, motor_state):
    """电机命令发送线程"""
    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            motor.sendMotorCommand(motor.motor_ids[0], motor.torq2curcom(command[0]))
            motor_state['position'] = motor.get_position()[0]
            motor_state['velocity'] = motor.get_velocity()[0]
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)


def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

# TODO: Test this implementation vs. lowpass filter on position
def sliding_window_average(data, window_size=5):
    if len(data) <= window_size:
        return np.mean(data)
    return 0.0


signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    print("=" * 60)
    print("EMG-based Paper OIAC+ILC Control System")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print(f"ILC enabled: {ILC_ENABLED}")
    if ILC_ENABLED:
        print(f"Max trials: {ILC_MAX_TRIALS}")
        print(f"Trial duration: {ILC_TRIAL_DURATION}s")
        print("\n⚠️  IMPORTANT: Please repeat the SAME movement pattern")
        print("   in each trial for effective ILC learning!")
    print("=" * 60)
    
    # 创建队列
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    command_queue = queue.Queue(maxsize=10)
    motor_state = {'position': 0, 'velocity': 0}
    position_average_queue = queue.Queue(maxsize=5)
    
    # 初始化EMG传感器
    emg = DelsysEMG()
    
    # 初始化滤波器和解释器
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    # interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, 
    #                  user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, 
                     user_name=USER_NAME, BicepEMG=True, TricepEMG=False)
    
    interpreter.set_Kp(8)
    
    # 初始化电机
    motor = Motors()
    motor.set_cont_mode(mode='cur')
    position_filter = rt_filtering(SAMPLE_RATE, 10, 0.5, 2)
    
    # 初始化控制器（使用论文版本的OIAC）
    oiac = OnlineImpedanceAdaptationController(dof=1)
    muscle_estimator = EMGMuscleForceEstimator()
    ilc = EnhancedILC(max_trials=ILC_MAX_TRIALS) if ILC_ENABLED else None
    
    # 尝试加载之前的ILC学习数据
    if ILC_ENABLED and os.path.exists(ILC_SAVE_PATH):
        user_input = input(f"\nFound saved ILC data. Load it? (y/n): ")
        if user_input.lower() == 'y':
            ilc.load_learning(ILC_SAVE_PATH)
    
    # 基础PD增益（用于前馈部分）
    Kp_base = 30.0
    Kd_base = 10.0
    
    # 电机位置转换参数
    step = 1500.0 / 140.0
    motor_center = 2550
    
    # 等待并初始化电机位置
    time.sleep(1.0)
    
    # 启动EMG传感器
    emg.start()
    
    # 启动线程
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_motor = threading.Thread(target=send_motor_command, args=(motor, command_queue, motor_state), daemon=True)
    t_emg.start()
    t_motor.start()
    print("\nEMG and motor threads started!")
    
    # ILC trial循环
    if ILC_ENABLED:
        max_trials = ILC_MAX_TRIALS
        start_trial = ilc.current_trial
    else:
        max_trials = 1
        start_trial = 0
    
    all_trial_stats = []
    
    for trial_num in range(start_trial, max_trials):
        if ILC_ENABLED:
            print(f"\n{'='*60}")
            print(f"Starting Trial {trial_num + 1}/{max_trials}")
            print(f"{'='*60}")
            print("⚠️  Please perform the SAME movement pattern as previous trials!")
            print("   This is critical for ILC learning effectiveness.")
            print("Press Enter to start trial...")
            input()
        
        # 重置trial相关的状态
        oiac.reset()
        muscle_estimator.reset_history()
        
        Bicep_RMS_queue = queue.Queue(maxsize=50)
        Tricep_RMS_queue = queue.Queue(maxsize=50)
        
        # Trial数据记录
        trial_time_log = []
        trial_error_log = []
        trial_torque_log = []
        trial_desired_angle_log = []
        trial_current_angle_log = []
        trial_bicep_force_log = []
        trial_tricep_force_log = []
        trial_k_log = []
        trial_b_log = []
        
        # 状态变量
        current_angle = math.radians(55.0)
        current_velocity = 0.0
        last_time = time.time()
        trial_start_time = time.time()
        last_desired_angle = math.radians(55.0)
        
        # 统计变量
        control_count = 0
        last_debug_time = time.time()
        last_force_debug_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Trial {trial_num + 1} - Control Loop Active")
        print(f"{'='*60}\n")
        
        try:
            while not stop_event.is_set():
                # 检查trial时间限制
                if ILC_ENABLED:
                    elapsed_time = time.time() - trial_start_time
                    if elapsed_time > ILC_TRIAL_DURATION:
                        print(f"\n[Trial {trial_num + 1}] Duration reached, ending trial...")
                        break
                
                # 获取EMG数据
                try:
                    reading = raw_data.get_nowait()
                except queue.Empty:
                    time.sleep(0.001)
                    continue
                
                current_time = time.time()
                dt = current_time - last_time
                trial_time = current_time - trial_start_time
                
                # 滤波EMG数据
                filtered_Bicep = filter_bicep.bandpass(reading[0])
                filtered_Tricep = filter_tricep.bandpass(reading[1]) if len(reading) > 1 else 0.0
                
                # 计算RMS
                try:
                    if Bicep_RMS_queue.full():
                        Bicep_RMS_queue.get_nowait()
                    Bicep_RMS_queue.put_nowait(filtered_Bicep)
                    
                    if Tricep_RMS_queue.full():
                        Tricep_RMS_queue.get_nowait()
                    Tricep_RMS_queue.put_nowait(filtered_Tricep)
                except queue.Full:
                    pass
                
                Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
                Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))
                
                # 低通滤波RMS信号
                filtered_bicep_RMS = filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))
                filtered_tricep_RMS = filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))
                
                # 计算激活度和期望角度
                #activation = interpreter.compute_activation([filtered_bicep_RMS, filtered_tricep_RMS])
                activation = interpreter.compute_activation(filtered_bicep_RMS)
                desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
                
                # TODO: Test these two commented out implementations.
                # Sliding window average
                # if position_average_queue.full():
                #     position_average_queue.get_nowait()
                # position_average_queue.put_nowait(desired_angle_deg)
                #desired_angle_deg = sliding_window_average(list(position_average_queue.queue))
                # Lowpass filter
                # desired_angle_deg = position_filter.lowpass(np.atleast_1d(desired_angle_deg))
                
                # Convert to radians
                desired_angle_rad = math.radians(desired_angle_deg)
                
                # 估计期望角速度
                desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
                last_desired_angle = desired_angle_rad
                
                # 估计当前角速度
                #current_velocity = (desired_angle_rad - current_angle) / dt if dt > 0 else 0.0
                #current_angle += current_velocity * dt
                current_velocity = motor_state['velocity']
                current_angle_deg = (motor_center - motor_state['position']) / step
                current_angle = math.radians(current_angle_deg)
                
                # ========== Paper-Based OIAC+ILC Control Law ==========
                
                position_error = desired_angle_rad - current_angle
                velocity_error = desired_velocity_rad - current_velocity
                
                # 1. OIAC: Update impedance parameters using paper formula
                K_mat, B_mat, integral = oiac.update_impedance(
                    current_angle, desired_angle_rad,
                    current_velocity, desired_velocity_rad,
                    dt
                )
                
                # 2. OIAC Feedback: Compute impedance-based torque (tau_fb)
                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])
                
                # Paper Eq. (2): tau_fb = K*e_pos + B*e_vel
                impedance_torque = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
                
                # Add integral term for steady-state error
                integral_torque = oiac.ki * integral
                
                # 3. ILC Feedforward (tau_ff)
                ff_torque = 0.0
                if ILC_ENABLED and trial_num > 0:
                    ff_torque = ilc.get_feedforward(trial_time, trial_num - 1)
                
                # 4. Total torque: tau = tau_ff + tau_fb + tau_integral
                total_torque = ff_torque + impedance_torque + integral_torque
                
                # ===== 肌肉力估计和优化 =====
                bicep_force, tricep_force = muscle_estimator.estimate_muscle_forces(
                    Bicep_RMS, Tricep_RMS
                )
                
                force_penalty = muscle_estimator.calculate_force_penalty(
                    bicep_force, tricep_force, position_error, total_torque
                )
                
                # 应用肌肉力惩罚
                final_torque = total_torque - force_penalty
                
                # 扭矩限制
                torque_clipped = np.clip(final_torque, TORQUE_MIN, TORQUE_MAX)
                
                # 记录trial数据
                trial_time_log.append(trial_time)
                trial_error_log.append(position_error)
                trial_torque_log.append(torque_clipped)
                trial_desired_angle_log.append(desired_angle_rad)
                trial_current_angle_log.append(current_angle)
                trial_bicep_force_log.append(bicep_force)
                trial_tricep_force_log.append(tricep_force)
                trial_k_log.append(float(K_mat[0, 0]))
                trial_b_log.append(float(B_mat[0, 0]))
                
                # 转换为电机位置命令（使用期望角度）
                position_motor = motor_center - int(desired_angle_deg * step)
                
                # 发送命令（只用position控制）
                try:
                    command_queue.put_nowait((torque_clipped, position_motor))
                except queue.Full:
                    try:
                        command_queue.get_nowait()
                        command_queue.put_nowait((torque_clipped, position_motor))
                    except:
                        pass
                
                # ===== 调试输出 =====
                control_count += 1
                
                if current_time - last_debug_time > 2.0:
                    error_deg = math.degrees(position_error)
                    k_val = float(K_mat[0, 0])
                    b_val = float(B_mat[0, 0])
                    print(f"t={trial_time}s | "
                          f"Desired={desired_angle_deg}° | "
                          f"Current={math.degrees(current_angle)}° | "
                          f"Error={error_deg}° | "
                          f"Torque={torque_clipped}Nm | "
                          f"FF={ff_torque}Nm | "
                          f"K={k_val} | "
                          f"B={b_val}")
                    last_debug_time = current_time
                
                if current_time - last_force_debug_time > 3.0:
                    print(f"Muscle | "
                          f"Bicep: {bicep_force:6.2f}N | "
                          f"Tricep: {tricep_force:6.2f}N | "
                          f"Penalty: {force_penalty:6.4f}Nm")
                    last_force_debug_time = current_time
                
                last_time = current_time
        
        except KeyboardInterrupt:
            print(f"\n[Trial {trial_num + 1}] Interrupted by user")
            if not ILC_ENABLED:
                break
        
        # Trial结束，统计结果
        print(f"\n{'='*60}")
        print(f"Trial {trial_num + 1} Summary")
        print(f"{'='*60}")
        
        if len(trial_error_log) > 0:
            avg_error = np.mean(np.abs(trial_error_log))
            max_error = np.max(np.abs(trial_error_log))
            avg_bicep = np.mean(trial_bicep_force_log)
            avg_tricep = np.mean(trial_tricep_force_log)
            avg_k = np.mean(trial_k_log)
            avg_b = np.mean(trial_b_log)
            
            trial_stats = {
                'trial': trial_num + 1,
                'avg_error_deg': math.degrees(avg_error),
                'max_error_deg': math.degrees(max_error),
                'avg_bicep_force': avg_bicep,
                'avg_tricep_force': avg_tricep,
                'avg_k': avg_k,
                'avg_b': avg_b,
                'control_cycles': control_count
            }
            all_trial_stats.append(trial_stats)
            
            print(f"Average tracking error: {math.degrees(avg_error)}°")
            print(f"Maximum tracking error: {math.degrees(max_error)}°")
            print(f"Average bicep force: {avg_bicep}N")
            print(f"Average tricep force: {avg_tricep}N")
            print(f"Average K: {avg_k}, Average B: {avg_b}")
            print(f"Control cycles: {control_count}")
            
            # ILC学习更新
            if ILC_ENABLED and trial_num < max_trials - 1:
                print(f"\nUpdating ILC learning...")
                ilc.update_learning(trial_time_log, trial_error_log, trial_torque_log)
                
                # 保存学习数据
                ilc.save_learning(ILC_SAVE_PATH)
                
                # 检查是否达到目标性能
                if math.degrees(avg_error) < 2.0:
                    print(f"\n🎉 Target performance achieved! Avg error < 2°")
                    user_input = input("Continue learning? (y/n): ")
                    if user_input.lower() != 'y':
                        break
        else:
            print("No data collected in this trial")
        
        # 如果不是ILC模式，只运行一次
        if not ILC_ENABLED:
            break
        
        print(f"\n{'='*60}\n")
    
    # 最终统计
    print("\n" + "="*60)
    print("FINAL STATISTICS - Paper OIAC+ILC")
    print("="*60)
    
    if ILC_ENABLED and len(all_trial_stats) > 0:
        print(f"\nCompleted {len(all_trial_stats)} trials")
        print("\nLearning Progress:")
        for stats in all_trial_stats:
            print(f"  Trial {stats['trial']}: "
                  f"Avg Error={stats['avg_error_deg']}°, "
                  f"Max Error={stats['max_error_deg']}°, "
                  f"K={stats['avg_k']}, "
                  f"B={stats['avg_b']}, "
                  f"Bicep={stats['avg_bicep_force']}N, "
                  f"Tricep={stats['avg_tricep_force']}N")
        
        if len(all_trial_stats) > 1:
            improvement = (all_trial_stats[0]['avg_error_deg'] - 
                          all_trial_stats[-1]['avg_error_deg'])
            print(f"\nError improvement: {improvement}° "
                  f"({all_trial_stats[0]['avg_error_deg']}° → "
                  f"{all_trial_stats[-1]['avg_error_deg']}°)")

    # 停止系统
    print("\n" + "="*60)
    print("SHUTTING DOWN")
    print("="*60)
    stop_event.set()
    
    t_emg.join(timeout=2.0)
    t_motor.join(timeout=2.0)
    
    emg.stop()
    motor.close()
    
    raw_data.queue.clear()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    command_queue.queue.clear()
    
    print("\nGoodbye!")