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
import select

SAMPLE_RATE = 2000  # Hz
USER_NAME = 'zichen'
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
trial_reset_event = threading.Event()

# ==================== Control Mode Definitions ====================

class ControlMode:
    """控制模式定义"""
    AAN = "assist_as_needed"  # 辅助模式
    RAN = "resist_as_needed"  # 阻力模式


class ModeManager:
    """
    模式管理器 - 根据论文图9实现AAN/RAN切换逻辑
    
    转换条件:
    1. AAN -> RAN: 用户能稳定跟踪目标（连续N秒误差<阈值）
    2. RAN -> AAN: 用户在RAN模式下表现不佳（运动幅度不足或误差过大）
    """
    def __init__(self):
        self.current_mode = ControlMode.AAN  # 默认从AAN开始
        self.mode_history = []
        
        # 切换条件参数
        self.aan_to_ran_error_threshold = math.radians(5.0)  # 5度误差阈值
        self.aan_to_ran_stable_time = 10.0  # 需要10秒稳定表现
        self.ran_to_aan_motion_threshold = math.radians(10.0)  # RAN模式最小运动幅度
        self.ran_to_aan_error_threshold = math.radians(15.0)  # RAN模式最大允许误差
        
        # 状态跟踪
        self.stable_tracking_start_time = None
        self.ran_motion_range_history = []
        self.ran_error_history = []
        
    def update_mode(self, position_error, current_angle, desired_angle, current_time):
        """
        更新控制模式
        
        参数:
            position_error: 当前位置误差（弧度）
            current_angle: 当前关节角度
            desired_angle: 期望关节角度
            current_time: 当前时间
        
        返回:
            mode_changed: 是否发生模式切换
        """
        old_mode = self.current_mode
        
        if self.current_mode == ControlMode.AAN:
            # AAN -> RAN 条件检查
            if abs(position_error) < self.aan_to_ran_error_threshold:
                if self.stable_tracking_start_time is None:
                    self.stable_tracking_start_time = current_time
                elif (current_time - self.stable_tracking_start_time) > self.aan_to_ran_stable_time:
                    self.current_mode = ControlMode.RAN
                    self.stable_tracking_start_time = None
                    print(f"\n{'='*60}")
                    print("MODE SWITCH: AAN -> RAN")
                    print("User has demonstrated stable tracking ability")
                    print(f"{'='*60}\n")
            else:
                self.stable_tracking_start_time = None
                
        elif self.current_mode == ControlMode.RAN:
            # RAN -> AAN 条件检查
            motion_range = abs(current_angle - math.radians(55.0))  # 相对于中立位置
            self.ran_motion_range_history.append(motion_range)
            self.ran_error_history.append(abs(position_error))
            
            # 保持最近5秒的历史
            if len(self.ran_motion_range_history) > 100:  # 假设50Hz控制频率
                self.ran_motion_range_history.pop(0)
                self.ran_error_history.pop(0)
            
            # 检查是否需要切回AAN
            if len(self.ran_motion_range_history) > 50:
                avg_motion = np.mean(self.ran_motion_range_history[-50:])
                avg_error = np.mean(self.ran_error_history[-50:])
                
                # 运动幅度不足或误差过大
                if (avg_motion < self.ran_to_aan_motion_threshold or 
                    avg_error > self.ran_to_aan_error_threshold):
                    self.current_mode = ControlMode.AAN
                    self.ran_motion_range_history.clear()
                    self.ran_error_history.clear()
                    print(f"\n{'='*60}")
                    print("MODE SWITCH: RAN -> AAN")
                    print(f"Avg motion: {math.degrees(avg_motion):.1f}°, "
                          f"Avg error: {math.degrees(avg_error):.1f}°")
                    print("User needs more assistance")
                    print(f"{'='*60}\n")
        
        mode_changed = (old_mode != self.current_mode)
        if mode_changed:
            self.mode_history.append({
                'time': current_time,
                'from': old_mode,
                'to': self.current_mode
            })
        
        return mode_changed
    
    def manual_switch_mode(self):
        """手动切换模式"""
        if self.current_mode == ControlMode.AAN:
            self.current_mode = ControlMode.RAN
            print("\nManually switched to RAN mode")
        else:
            self.current_mode = ControlMode.AAN
            print("\nManually switched to AAN mode")
        
        self.stable_tracking_start_time = None
        self.ran_motion_range_history.clear()
        self.ran_error_history.clear()


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


# ==================== OIAC Controller (Paper Implementation with RAN/AAN) ====================

class OnlineImpedanceAdaptationController:
    """
    Online Impedance Adaptation Controller based on:
    Xiong, X., & Fang, C. (2023). An Online Impedance Adaptation Controller 
    for Decoding Skill Intelligence. Biomimetic Intelligence and Robotics, 3(2).
    
    Enhanced with RAN/AAN mode support (Section 4.3)
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
        
        # Mode-specific scaling factors
        self.k_scale_aan = 100.0   # AAN stiffness scaling
        self.b_scale_aan = 80.0    # AAN damping scaling
        self.k_scale_ran = 150.0   # RAN stiffness scaling (higher resistance)
        self.b_scale_ran = 120.0   # RAN damping scaling (higher resistance)
        
        # Safety limits
        self.k_min = 30.0
        self.k_max_aan = 150.0
        self.k_max_ran = 250.0  # RAN allows higher stiffness
        self.b_min = 10.0
        self.b_max_aan = 60.0
        self.b_max_ran = 100.0  # RAN allows higher damping
        
        # Integral term for steady-state error (AAN only)
        self.integral = 0.0
        self.ki = 5.0
        self.max_integral = 15.0
        
        # RAN specific: fixed reference position (Paper Eq. 22)
        self.ran_reference_position = math.radians(55.0)  # Neutral elbow position
        
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
    
    def update_impedance(self, q, q_d, dq, dq_d, dt, mode):
        """
        Update stiffness and damping matrices using OIAC algorithm (Eq. 2)
        with RAN/AAN mode support
        
        Args:
            q: Current joint position (scalar or array)
            q_d: Desired joint position (scalar or array) - used in AAN, ignored in RAN
            dq: Current joint velocity (scalar or array)
            dq_d: Desired joint velocity (scalar or array) - used in AAN, ignored in RAN
            dt: Time step for integral calculation
            mode: Control mode (ControlMode.AAN or ControlMode.RAN)
        
        Returns:
            k_mat: Updated stiffness matrix
            b_mat: Updated damping matrix
            integral: Integral term for steady-state error reduction
        """
        # Convert inputs to column vectors
        self.q = np.atleast_2d(np.atleast_1d(q)).T
        self.dq = np.atleast_2d(np.atleast_1d(dq)).T
        
        if mode == ControlMode.RAN:
            # RAN mode: Fixed reference position (Paper Eq. 22: q_d = 0)
            self.q_d = np.array([[self.ran_reference_position]])
            self.dq_d = np.zeros((self.DOF, 1))  # Desired velocity is zero
            
            # Use RAN scaling parameters
            k_scale = self.k_scale_ran
            b_scale = self.b_scale_ran
            k_max = self.k_max_ran
            b_max = self.b_max_ran
            
        else:  # AAN mode
            self.q_d = np.atleast_2d(np.atleast_1d(q_d)).T
            self.dq_d = np.atleast_2d(np.atleast_1d(dq_d)).T
            
            # Use AAN scaling parameters
            k_scale = self.k_scale_aan
            b_scale = self.b_scale_aan
            k_max = self.k_max_aan
            b_max = self.b_max_aan
        
        # Compute error terms
        track_err = self.gen_track_err()
        pos_err = self.gen_pos_err()
        vel_err = self.gen_vel_err()
        ad_factor = self.gen_ad_factor()
        
        # Update stiffness K and damping B using Eq. (2) - outer product formulation
        self.k_mat = k_scale * (track_err @ pos_err.T) / ad_factor
        self.b_mat = b_scale * (track_err @ vel_err.T) / ad_factor
        
        # Apply safety limits (element-wise clipping)
        self.k_mat = np.clip(self.k_mat, self.k_min, k_max)
        self.b_mat = np.clip(np.abs(self.b_mat), self.b_min, b_max)
        
        # Integral term for steady-state error reduction (AAN only)
        if mode == ControlMode.AAN:
            error_scalar = float(pos_err.item())
            self.integral += error_scalar * dt
            
            # Anti-windup: only integrate when error is small
            if abs(error_scalar) < math.radians(2.0):
                self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
            else:
                self.integral *= 0.9  # Decay when error is large
        else:
            self.integral = 0.0  # No integral term in RAN mode
        
        return self.k_mat, self.b_mat, self.integral
    
    def reset(self):
        """重置控制器状态（用于新trial）"""
        self.integral = 0.0


# ==================== Enhanced ILC Controller ====================

class EnhancedILC:
    """
    增强的迭代学习控制器
    用于重复性任务的前馈学习
    """
    def __init__(self, max_trials=10, reference_length=5000):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []
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
        
        # 学习更新
        if not self.learned_feedforward:
            ff = np.zeros_like(aligned_error)
        else:
            lr = self.learning_rates[min(self.current_trial, len(self.learning_rates)-1)]
            ff = self.learned_feedforward[-1] + lr * aligned_error
        
        # 限制前馈幅度
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
        print(f"      Learning rate: {self.learning_rates[min(self.current_trial-1, len(self.learning_rates)-1)]:.2f}")
        print(f"      Avg error: {math.degrees(avg_error):.2f}°")
        print(f"      Max error: {math.degrees(max_error):.2f}°")
        print(f"      Feedforward range: [{np.min(ff):.2f}, {np.max(ff):.2f}] Nm")
        
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


def read_EMG(EMG_sensor, queue):
    """EMG读取线程"""
    while not stop_event.is_set():
        reading = EMG_sensor.read()
        try:
            queue.put_nowait(reading)
        except queue.Full:
            try:
                queue.get_nowait()
                queue.put_nowait(reading)
            except queue.Full:
                pass
        except Exception as e:
            print(f"[reader] error: {e}", file=sys.stderr)


def send_motor_command(motor, command_queue):
    """电机命令发送线程"""
    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            motor.sendMotorCommand(motor.motor_ids[0], command[1])
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)


def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    print("=" * 60)
    print("EMG-based Paper OIAC+ILC with RAN/AAN Control System")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print(f"ILC enabled: {ILC_ENABLED}")
    if ILC_ENABLED:
        print(f"Max trials: {ILC_MAX_TRIALS}")
        print(f"Trial duration: {ILC_TRIAL_DURATION}s")
        print("\n⚠️  IMPORTANT: Please repeat the SAME movement pattern")
        print("   in each trial for effective ILC learning!")
    print("\n📋 Control Modes:")
    print("   - AAN (Assist-as-Needed): Helps complete movements")
    print("   - RAN (Resist-as-Needed): Provides resistance training")
    print("   - Press 'm' + Enter during trial to manually switch modes")
    print("=" * 60)
    
    # 创建队列
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    command_queue = queue.Queue(maxsize=10)
    
    # 初始化EMG传感器
    emg = DelsysEMG()
    
    # 初始化滤波器和解释器
    filter = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, 
                     user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    interpreter.set_Kp(8)
    
    # 初始化电机
    motor = Motors()
    
    # 初始化控制器
    oiac = OnlineImpedanceAdaptationController(dof=1)
    muscle_estimator = EMGMuscleForceEstimator()
    mode_manager = ModeManager()  # 新增：模式管理器
    ilc = EnhancedILC(max_trials=ILC_MAX_TRIALS) if ILC_ENABLED else None
    
    # 尝试加载之前的ILC学习数据
    if ILC_ENABLED and os.path.exists(ILC_SAVE_PATH):
        user_input = input(f"\nFound saved ILC data. Load it? (y/n): ")
        if user_input.lower() == 'y':
            ilc.load_learning(ILC_SAVE_PATH)
    
    # 电机位置转换参数
    step = 1500.0 / 140.0
    motor_center = 2550
    
    # 等待并初始化电机位置
    time.sleep(1.0)
    motor.sendMotorCommand(motor.motor_ids[0], motor_center)
    time.sleep(1.0)
    
    # 启动EMG传感器
    emg.start()
    
    # 启动线程
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_motor = threading.Thread(target=send_motor_command, args=(motor, command_queue), daemon=True)
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
            print("   Press 'm' + Enter to switch AAN/RAN mode during trial")
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
        trial_mode_log = []  # 新增：记录模式变化
        
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
        print(f"Current Mode: {mode_manager.current_mode}")
        print(f"{'='*60}\n")
        
        try:
            while not stop_event.is_set():
                # 检查trial时间限制
                if ILC_ENABLED:
                    elapsed_time = time.time() - trial_start_time
                    if elapsed_time > ILC_TRIAL_DURATION:
                        print(f"\n[Trial {trial_num + 1}] Duration reached, ending trial...")
                        break
                
                # 检查手动模式切换 (非阻塞)
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip()
                    if key.lower() == 'm':
                        mode_manager.manual_switch_mode()
                        oiac.reset()  # 重置控制器状态
                
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
                filtered_Bicep = filter.bandpass(reading[0])
                filtered_Tricep = filter.bandpass(reading[1]) if len(reading) > 1 else 0.0
                
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
                filtered_bicep_RMS = filter.lowpass(np.atleast_1d(Bicep_RMS))
                filtered_tricep_RMS = filter.lowpass(np.atleast_1d(Tricep_RMS))
                
                # 计算激活度和期望角度
                activation = interpreter.compute_activation(filtered_bicep_RMS, filtered_tricep_RMS)
                desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
                desired_angle_rad = math.radians(desired_angle_deg)
                
                # 估计期望角速度
                desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
                last_desired_angle = desired_angle_rad
                
                # 估计当前角速度
                current_velocity = (desired_angle_rad - current_angle) / dt if dt > 0 else 0.0
                current_angle += current_velocity * dt
                
                # ========== Paper-Based OIAC+ILC Control Law with RAN/AAN ==========
                
                position_error = desired_angle_rad - current_angle
                velocity_error = desired_velocity_rad - current_velocity
                
                # 获取当前控制模式
                current_mode = mode_manager.current_mode
                
                # 更新模式（自动切换逻辑）
                mode_changed = mode_manager.update_mode(
                    position_error, current_angle, desired_angle_rad, current_time
                )
                
                # 如果模式改变，重置控制器
                if mode_changed:
                    oiac.reset()
                    current_mode = mode_manager.current_mode
                
                # 1. OIAC: Update impedance parameters using paper formula
                K_mat, B_mat, integral = oiac.update_impedance(
                    current_angle, desired_angle_rad,
                    current_velocity, desired_velocity_rad,
                    dt, current_mode  # 传入当前模式
                )
                
                # 2. OIAC Feedback: Compute impedance-based torque (tau_fb)
                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])
                
                # Paper Eq. (2): tau_fb = K*e_pos + B*e_vel
                impedance_torque = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
                
                # 3. Mode-specific control law
                if current_mode == ControlMode.AAN:
                    # AAN模式: 使用完整控制
                    # 注意：保持你原代码的符号约定（不加负号）
                    # 如果你的系统需要负号来辅助，可以改为: total_torque = -(...)
                    
                    # Add integral term for steady-state error
                    integral_torque = oiac.ki * integral
                    
                    # 前馈扭矩（来自ILC）
                    ff_torque = 0.0
                    if ILC_ENABLED and trial_num > 0:
                        ff_torque = ilc.get_feedforward(trial_time, trial_num - 1)
                    
                    # 总扭矩: τ = τ_ff + τ_fb + τ_integral
                    # 保持你原代码的符号约定
                    total_torque = ff_torque + impedance_torque + integral_torque#注意这段 - or not 
                    
                else:  # RAN模式
                    # RAN模式: 只用反馈控制，提供阻力 (论文公式21)
                    # τ = -τ_fb
                    # 负号表示阻力方向，抵抗用户运动
                    total_torque = -impedance_torque
                    integral_torque = 0.0
                    ff_torque = 0.0
                
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
                trial_mode_log.append(current_mode)
                
                # 转换为电机位置命令（使用期望角度）
                position_motor = motor_center - int(desired_angle_deg * step)
                
                # 发送命令
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
                    mode_str = "AAN" if current_mode == ControlMode.AAN else "RAN"
                    
                    if current_mode == ControlMode.AAN:
                        print(f"[{mode_str}] t={trial_time:.1f}s | "
                              f"Desired={desired_angle_deg:6.1f}° | "
                              f"Current={math.degrees(current_angle):6.1f}° | "
                              f"Error={error_deg:6.1f}° | "
                              f"Torque={torque_clipped:6.2f}Nm | "
                              f"FF={ff_torque:6.2f}Nm | "
                              f"K={k_val:5.1f} | "
                              f"B={b_val:5.1f}")
                    else:  # RAN
                        ref_pos_deg = math.degrees(oiac.ran_reference_position)
                        print(f"[{mode_str}] t={trial_time:.1f}s | "
                              f"Ref={ref_pos_deg:6.1f}° | "
                              f"Current={math.degrees(current_angle):6.1f}° | "
                              f"Dev={math.degrees(current_angle - oiac.ran_reference_position):6.1f}° | "
                              f"Resistance={torque_clipped:6.2f}Nm | "
                              f"K={k_val:5.1f} | "
                              f"B={b_val:5.1f}")
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
            
            # 统计模式使用情况
            aan_count = sum(1 for m in trial_mode_log if m == ControlMode.AAN)
            ran_count = sum(1 for m in trial_mode_log if m == ControlMode.RAN)
            total_count = len(trial_mode_log)
            aan_percentage = (aan_count / total_count * 100) if total_count > 0 else 0
            ran_percentage = (ran_count / total_count * 100) if total_count > 0 else 0
            
            trial_stats = {
                'trial': trial_num + 1,
                'avg_error_deg': math.degrees(avg_error),
                'max_error_deg': math.degrees(max_error),
                'avg_bicep_force': avg_bicep,
                'avg_tricep_force': avg_tricep,
                'avg_k': avg_k,
                'avg_b': avg_b,
                'control_cycles': control_count,
                'aan_percentage': aan_percentage,
                'ran_percentage': ran_percentage
            }
            all_trial_stats.append(trial_stats)
            
            print(f"Average tracking error: {math.degrees(avg_error):.2f}°")
            print(f"Maximum tracking error: {math.degrees(max_error):.2f}°")
            print(f"Average bicep force: {avg_bicep:.2f}N")
            print(f"Average tricep force: {avg_tricep:.2f}N")
            print(f"Average K: {avg_k:.1f}, Average B: {avg_b:.1f}")
            print(f"Control cycles: {control_count}")
            print(f"Mode usage: AAN={aan_percentage:.1f}%, RAN={ran_percentage:.1f}%")
            print(f"Mode switches: {len(mode_manager.mode_history)}")
            
            # 显示模式切换历史
            if mode_manager.mode_history:
                print("\nMode switch history:")
                for switch in mode_manager.mode_history:
                    print(f"  t={switch['time']-trial_start_time:.1f}s: "
                          f"{switch['from']} -> {switch['to']}")
            
            # ILC学习更新 (只在AAN模式数据上学习)
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
    print("FINAL STATISTICS - Paper OIAC+ILC with RAN/AAN")
    print("="*60)
    
    if ILC_ENABLED and len(all_trial_stats) > 0:
        print(f"\nCompleted {len(all_trial_stats)} trials")
        print("\nLearning Progress:")
        for stats in all_trial_stats:
            print(f"  Trial {stats['trial']}: "
                  f"Avg Error={stats['avg_error_deg']:.2f}°, "
                  f"Max Error={stats['max_error_deg']:.2f}°, "
                  f"K={stats['avg_k']:.1f}, "
                  f"B={stats['avg_b']:.1f}, "
                  f"AAN={stats['aan_percentage']:.0f}%, "
                  f"RAN={stats['ran_percentage']:.0f}%")
        
        if len(all_trial_stats) > 1:
            improvement = (all_trial_stats[0]['avg_error_deg'] - 
                          all_trial_stats[-1]['avg_error_deg'])
            print(f"\nError improvement: {improvement:.2f}° "
                  f"({all_trial_stats[0]['avg_error_deg']:.2f}° → "
                  f"{all_trial_stats[-1]['avg_error_deg']:.2f}°)")
        
        # 模式使用统计
        total_aan = sum(s['aan_percentage'] for s in all_trial_stats)
        total_ran = sum(s['ran_percentage'] for s in all_trial_stats)
        avg_aan = total_aan / len(all_trial_stats)
        avg_ran = total_ran / len(all_trial_stats)
        print(f"\nOverall mode usage: AAN={avg_aan:.1f}%, RAN={avg_ran:.1f}%")
    
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
    command_queue.queue.clear()
    
    print("\nGoodbye!")