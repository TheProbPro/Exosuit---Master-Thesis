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
USER_NAME = 'zichen'
ANGLE_MIN = 0
ANGLE_MAX = 140

# Control parameters
TORQUE_MIN = -4.1  # Nm
TORQUE_MAX = 4.1   # Nm

# ========== XM440-W270-R SPECIFIC CONFIGURATION ==========
# XM440-W270-R 规格参数
MOTOR_MODEL = "XM440-W270-R"
MOTOR_TORQUE_CONSTANT = 1.783  # Kt [N·m/A] (实测值，可能略有偏差)
MOTOR_STALL_TORQUE = 4.1       # N·m @ 11.1V
MOTOR_STALL_CURRENT = 2.3      # A
MOTOR_MAX_CURRENT = 2.5        # A (安全限制)
MOTOR_GEAR_RATIO = 1.0         # 无减速器

# Dynamixel 控制表地址 (XM440系列)
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102        # 2 bytes, -1193~1193 (约 -2.69A ~ 2.69A)
ADDR_GOAL_PWM = 100           # 2 bytes, -885~885
ADDR_PRESENT_CURRENT = 126    # 2 bytes
ADDR_PRESENT_POSITION = 132   # 4 bytes

# 操作模式
MODE_CURRENT_CONTROL = 0
MODE_POSITION_CONTROL = 3
MODE_PWM_CONTROL = 16

# 扭矩控制模式选择: 'current', 'pwm', 'virtual_spring'
TORQUE_CONTROL_MODE = 'current'  # 推荐使用 current

# 虚拟弹簧模式参数（如果使用 virtual_spring）
VIRTUAL_STIFFNESS = 50.0     # [N·m/rad]
POSITION_SCALE = 1500.0 / 140.0
MOTOR_CENTER = 2550

# ILC parameters
ILC_ENABLED = True
ILC_MAX_TRIALS = 10
ILC_TRIAL_DURATION = 10.0
ILC_SAVE_PATH = "ilc_learning_data_xm440_torque.pkl"

stop_event = threading.Event()


class XM440TorqueController:
    """
    XM440-W270-R 专用扭矩控制器
    支持电流控制、PWM控制和虚拟弹簧模式
    """
    def __init__(self, motor, mode='current'):
        """
        Args:
            motor: Motors对象
            mode: 'current' (推荐), 'pwm', 'virtual_spring'
        """
        self.motor = motor
        self.mode = mode
        self.motor_id = motor.motor_ids[0]
        
        # XM440 参数
        self.Kt = MOTOR_TORQUE_CONSTANT
        self.max_current = MOTOR_MAX_CURRENT
        self.gear_ratio = MOTOR_GEAR_RATIO
        
        # 电流控制单位转换: Goal Current 单位约为 2.69mA
        # Goal Current 范围: -1193 ~ 1193 对应约 -3.2A ~ 3.2A
        self.current_unit = 2.69  # mA per unit
        
        # PWM控制
        self.pwm_max = 885
        
        # 虚拟弹簧参数
        self.virtual_k = VIRTUAL_STIFFNESS
        self.position_scale = POSITION_SCALE
        self.motor_center = MOTOR_CENTER
        self.current_angle_rad = math.radians(55.0)
        
        print(f"[XM440TorqueController] Model: {MOTOR_MODEL}")
        print(f"[XM440TorqueController] Mode: {mode}")
        
        if mode == 'current':
            print(f"  Kt = {self.Kt:.3f} N·m/A")
            print(f"  Max current = {self.max_current} A")
            print(f"  Current unit = {self.current_unit} mA/unit")
            self._setup_current_control()
        elif mode == 'pwm':
            print(f"  PWM range = [-{self.pwm_max}, {self.pwm_max}]")
            self._setup_pwm_control()
        elif mode == 'virtual_spring':
            print(f"  Virtual stiffness = {self.virtual_k} N·m/rad")
            self._setup_position_control()
    
    def _setup_current_control(self):
        """设置电流控制模式"""
        try:
            # 方法1: 使用标准 Dynamixel SDK 方法
            if hasattr(self.motor, 'setOperatingMode'):
                self.motor.setOperatingMode(self.motor_id, MODE_CURRENT_CONTROL)
                print("  ✅ Operating mode set to Current Control (Mode 0)")
            elif hasattr(self.motor, 'write1ByteTxRx'):
                # 方法2: 直接写入控制表
                self.motor.write1ByteTxRx(self.motor_id, ADDR_OPERATING_MODE, MODE_CURRENT_CONTROL)
                print("  ✅ Operating mode set to Current Control (direct write)")
            
            # 使能扭矩
            if hasattr(self.motor, 'setTorqueEnable'):
                self.motor.setTorqueEnable(self.motor_id, 1)
            elif hasattr(self.motor, 'write1ByteTxRx'):
                self.motor.write1ByteTxRx(self.motor_id, ADDR_TORQUE_ENABLE, 1)
            
            print("  ✅ Torque enabled")
            
        except Exception as e:
            print(f"  ⚠️  Setup warning: {e}")
            print("  Attempting to continue with current methods...")
    
    def _setup_pwm_control(self):
        """设置PWM控制模式"""
        try:
            if hasattr(self.motor, 'setOperatingMode'):
                self.motor.setOperatingMode(self.motor_id, MODE_PWM_CONTROL)
            elif hasattr(self.motor, 'write1ByteTxRx'):
                self.motor.write1ByteTxRx(self.motor_id, ADDR_OPERATING_MODE, MODE_PWM_CONTROL)
            
            if hasattr(self.motor, 'setTorqueEnable'):
                self.motor.setTorqueEnable(self.motor_id, 1)
            elif hasattr(self.motor, 'write1ByteTxRx'):
                self.motor.write1ByteTxRx(self.motor_id, ADDR_TORQUE_ENABLE, 1)
            
            print("  ✅ PWM Control mode enabled")
        except Exception as e:
            print(f"  ⚠️  Setup warning: {e}")
    
    def _setup_position_control(self):
        """设置位置控制模式（虚拟弹簧）"""
        try:
            if hasattr(self.motor, 'setOperatingMode'):
                self.motor.setOperatingMode(self.motor_id, MODE_POSITION_CONTROL)
            elif hasattr(self.motor, 'write1ByteTxRx'):
                self.motor.write1ByteTxRx(self.motor_id, ADDR_OPERATING_MODE, MODE_POSITION_CONTROL)
            
            if hasattr(self.motor, 'setTorqueEnable'):
                self.motor.setTorqueEnable(self.motor_id, 1)
            elif hasattr(self.motor, 'write1ByteTxRx'):
                self.motor.write1ByteTxRx(self.motor_id, ADDR_TORQUE_ENABLE, 1)
            
            print("  ✅ Position Control mode enabled (virtual spring)")
        except Exception as e:
            print(f"  ⚠️  Setup warning: {e}")
    
    def send_torque_command(self, torque_nm, current_angle_rad=None):
        """
        发送扭矩命令
        
        Args:
            torque_nm: 期望扭矩 [N·m]
            current_angle_rad: 当前角度 [rad]
        
        Returns:
            实际发送的命令值
        """
        if self.mode == 'current':
            return self._torque_to_current_control(torque_nm)
        elif self.mode == 'pwm':
            return self._torque_to_pwm_control(torque_nm)
        elif self.mode == 'virtual_spring':
            return self._torque_to_virtual_spring(torque_nm, current_angle_rad)
        else:
            print(f"[XM440TorqueController] Unknown mode: {self.mode}")
            return 0
    
    def _torque_to_current_control(self, torque_nm):
        """
        扭矩 → 电流控制 (XM440专用)
        
        公式：I = τ / Kt
        XM440 Goal Current 单位: 约 2.69mA per unit
        """
        # 考虑减速比
        motor_torque = torque_nm / self.gear_ratio
        
        # 扭矩到电流 (A)
        current_a = motor_torque / self.Kt
        
        # 电流限制
        current_clipped = np.clip(current_a, -self.max_current, self.max_current)
        
        # 转换为 Goal Current 单位 (Dynamixel 内部单位)
        # Goal Current = (current_mA) / 2.69
        current_ma = current_clipped * 1000  # A -> mA
        goal_current_units = int(current_ma / self.current_unit)
        
        # Goal Current 范围限制: -1193 ~ 1193
        goal_current_units = np.clip(goal_current_units, -1193, 1193)
        
        # 发送电流命令
        try:
            # 方法1: 专用方法
            if hasattr(self.motor, 'setGoalCurrent'):
                self.motor.setGoalCurrent(self.motor_id, goal_current_units)
            elif hasattr(self.motor, 'writeGoalCurrent'):
                self.motor.writeGoalCurrent(self.motor_id, goal_current_units)
            # 方法2: 直接写入控制表 (2 bytes)
            elif hasattr(self.motor, 'write2ByteTxRx'):
                self.motor.write2ByteTxRx(self.motor_id, ADDR_GOAL_CURRENT, goal_current_units)
            # 方法3: 通用 sendMotorCommand (如果支持电流模式)
            elif hasattr(self.motor, 'sendMotorCommand'):
                self.motor.sendMotorCommand(self.motor_id, goal_current_units)
            else:
                print("[XM440TorqueController] ERROR: No current control method available!")
                return 0
                
        except Exception as e:
            print(f"[XM440TorqueController] Current control error: {e}")
            return 0
        
        return current_clipped
    
    def _torque_to_pwm_control(self, torque_nm):
        """
        扭矩 → PWM控制
        
        PWM范围: -885 ~ 885 对应满扭矩
        """
        motor_torque = torque_nm / self.gear_ratio
        
        # 线性映射: torque -> PWM
        pwm_value = int((motor_torque / MOTOR_STALL_TORQUE) * self.pwm_max)
        pwm_clipped = np.clip(pwm_value, -self.pwm_max, self.pwm_max)
        
        try:
            if hasattr(self.motor, 'setGoalPWM'):
                self.motor.setGoalPWM(self.motor_id, pwm_clipped)
            elif hasattr(self.motor, 'write2ByteTxRx'):
                self.motor.write2ByteTxRx(self.motor_id, ADDR_GOAL_PWM, pwm_clipped)
            else:
                print("[XM440TorqueController] ERROR: No PWM control method available!")
                return 0
                
        except Exception as e:
            print(f"[XM440TorqueController] PWM control error: {e}")
            return 0
        
        return pwm_clipped
    
    def _torque_to_virtual_spring(self, torque_nm, current_angle_rad):
        """
        扭矩 → 虚拟弹簧位置补偿
        
        tau = K * delta_theta
        delta_theta = tau / K
        """
        if current_angle_rad is None:
            print("[XM440TorqueController] ERROR: current_angle_rad required")
            return 0
        
        # 角度偏移
        angle_offset_rad = torque_nm / self.virtual_k
        
        # 目标角度
        self.current_angle_rad = current_angle_rad
        target_angle_rad = self.current_angle_rad + angle_offset_rad
        
        # 限制
        target_angle_rad = np.clip(target_angle_rad, 
                                   math.radians(ANGLE_MIN), 
                                   math.radians(ANGLE_MAX))
        
        # 转换为电机位置
        target_angle_deg = math.degrees(target_angle_rad)
        position_motor = self.motor_center - int(target_angle_deg * self.position_scale)
        
        # 发送位置命令
        self.motor.sendMotorCommand(self.motor_id, position_motor)
        
        return position_motor


class EMGMuscleForceEstimator:
    """使用EMG信号估计肌肉力"""
    def __init__(self):
        self.bicep_force_history = []
        self.tricep_force_history = []
        self.force_penalty_history = []
        self.emg_to_force_scale = 0.1
        
    def estimate_muscle_forces(self, bicep_rms, tricep_rms):
        bicep_force = bicep_rms * self.emg_to_force_scale
        tricep_force = tricep_rms * self.emg_to_force_scale
        
        bicep_force = max(0, bicep_force)
        tricep_force = max(0, tricep_force)
        
        return bicep_force, tricep_force
    
    def calculate_force_penalty(self, bicep_force, tricep_force, q_error, control_torque):
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
    
    def reset_history(self):
        self.bicep_force_history.clear()
        self.tricep_force_history.clear()
        self.force_penalty_history.clear()


class OnlineImpedanceAdaptationController:
    """OIAC Controller (Paper Implementation)"""
    def __init__(self, dof=1):
        self.DOF = dof
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))
        
        self.a = 0.04
        self.b = 0.001
        self.k = 0.5
        
        self.k_scale = 100.0
        self.b_scale = 80.0
        
        self.k_min = 30.0
        self.k_max = 150.0
        self.b_min = 10.0
        self.b_max = 60.0
        
        self.integral = 0.0
        self.ki = 5.0
        self.max_integral = 15.0
        
    def gen_pos_err(self):
        return (self.q - self.q_d)
    
    def gen_vel_err(self):
        return (self.dq - self.dq_d)
    
    def gen_track_err(self):
        return (self.k * self.gen_vel_err() + self.gen_pos_err())
    
    def gen_ad_factor(self):
        track_err_norm = la.norm(self.gen_track_err())
        return self.a / (1.0 + self.b * track_err_norm * track_err_norm)
    
    def update_impedance(self, q, q_d, dq, dq_d, dt=0.002):
        self.q = np.atleast_2d(np.atleast_1d(q)).T
        self.q_d = np.atleast_2d(np.atleast_1d(q_d)).T
        self.dq = np.atleast_2d(np.atleast_1d(dq)).T
        self.dq_d = np.atleast_2d(np.atleast_1d(dq_d)).T
        
        track_err = self.gen_track_err()
        pos_err = self.gen_pos_err()
        vel_err = self.gen_vel_err()
        ad_factor = self.gen_ad_factor()
        
        self.k_mat = self.k_scale * (track_err @ pos_err.T) / ad_factor
        self.b_mat = self.b_scale * (track_err @ vel_err.T) / ad_factor
        
        self.k_mat = np.clip(self.k_mat, self.k_min, self.k_max)
        self.b_mat = np.clip(np.abs(self.b_mat), self.b_min, self.b_max)
        
        error_scalar = float(pos_err.item())
        self.integral += error_scalar * dt
        
        if abs(error_scalar) < math.radians(2.0):
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        else:
            self.integral *= 0.9
        
        return self.k_mat, self.b_mat, self.integral
    
    def reset(self):
        self.integral = 0.0


class EnhancedILC:
    """Enhanced Iterative Learning Controller"""
    def __init__(self, max_trials=10, reference_length=5000):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []
        self.reference_time = None
        self.reference_length = reference_length
        self.learning_rates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
        self.trial_errors = []
        self.trial_torques = []
        
    def update_learning(self, time_array, error_array, torque_array):
        if len(time_array) == 0 or len(error_array) == 0:
            print("[ILC] Warning: Empty data, skipping update")
            return np.zeros(self.reference_length)
        
        if self.reference_time is None:
            max_time = max(time_array) if len(time_array) > 0 else ILC_TRIAL_DURATION
            self.reference_time = np.linspace(0, max_time, self.reference_length)
        
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
        
        if not self.learned_feedforward:
            ff = np.zeros_like(aligned_error)
        else:
            lr = self.learning_rates[min(self.current_trial, len(self.learning_rates)-1)]
            ff = self.learned_feedforward[-1] + lr * aligned_error
        
        ff = np.clip(ff, -20.0, 20.0)
        
        if len(ff) > 10:
            window_size = 11
            ff = np.convolve(ff, np.ones(window_size)/window_size, mode='same')
        
        self.learned_feedforward.append(ff)
        self.trial_errors.append(aligned_error)
        self.trial_torques.append(torque_array)
        self.current_trial += 1
        
        avg_error = np.mean(np.abs(aligned_error))
        max_error = np.max(np.abs(aligned_error))
        
        print(f"[ILC] Trial {self.current_trial} completed:")
        print(f"      Learning rate: {lr:.2f}")
        print(f"      Avg error: {math.degrees(avg_error):.2f}°")
        print(f"      Max error: {math.degrees(max_error):.2f}°")
        print(f"      Feedforward range: [{np.min(ff):.2f}, {np.max(ff):.2f}] Nm")
        
        return ff
    
    def get_feedforward(self, t, trial_idx=-1):
        if trial_idx < 0:
            trial_idx = len(self.learned_feedforward) - 1
            
        if trial_idx < 0 or trial_idx >= len(self.learned_feedforward):
            return 0.0
        
        if self.reference_time is None:
            return 0.0
            
        idx = np.argmin(np.abs(self.reference_time - t))
        if idx < len(self.learned_feedforward[trial_idx]):
            return float(self.learned_feedforward[trial_idx][idx])
        return 0.0
    
    def save_learning(self, filepath):
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


def send_torque_command_thread(torque_controller, command_queue):
    """扭矩命令发送线程"""
    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            torque_nm = command[0]
            current_angle = command[1] if len(command) > 1 else None
            torque_controller.send_torque_command(torque_nm, current_angle)
        except Exception as e:
            print(f"[torque send] error: {e}", file=sys.stderr)


def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    print("=" * 60)
    print(f"EMG OIAC+ILC Control - {MOTOR_MODEL} TORQUE MODE")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print(f"Torque control mode: {TORQUE_CONTROL_MODE}")
    print(f"Motor specs: Kt={MOTOR_TORQUE_CONSTANT:.3f} N·m/A, Max={MOTOR_STALL_TORQUE} N·m")
    print(f"ILC enabled: {ILC_ENABLED}")
    if ILC_ENABLED:
        print(f"Max trials: {ILC_MAX_TRIALS}")
        print(f"Trial duration: {ILC_TRIAL_DURATION}s")
        print("\n⚠️  IMPORTANT: Please repeat the SAME movement pattern!")
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
    
    # 初始化XM440专用扭矩控制器
    torque_controller = XM440TorqueController(motor, mode=TORQUE_CONTROL_MODE)
    
    # 初始化控制器
    oiac = OnlineImpedanceAdaptationController(dof=1)
    muscle_estimator = EMGMuscleForceEstimator()
    ilc = EnhancedILC(max_trials=ILC_MAX_TRIALS) if ILC_ENABLED else None
    
    # 加载ILC数据
    if ILC_ENABLED and os.path.exists(ILC_SAVE_PATH):
        user_input = input(f"\nFound saved ILC data. Load it? (y/n): ")
        if user_input.lower() == 'y':
            ilc.load_learning(ILC_SAVE_PATH)
    
    # 等待初始化
    time.sleep(1.0)
    
    # 如果是虚拟弹簧模式，初始化到中心位置
    if TORQUE_CONTROL_MODE == 'virtual_spring':
        motor.sendMotorCommand(motor.motor_ids[0], MOTOR_CENTER)
        time.sleep(1.0)
    
    # 启动EMG传感器
    emg.start()
    
    # 启动线程
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_torque = threading.Thread(target=send_torque_command_thread, 
                                args=(torque_controller, command_queue), daemon=True)
    t_emg.start()
    t_torque.start()
    print("\n✅ EMG and torque control threads started!")
    
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
            print("⚠️  Please perform the SAME movement pattern!")
            print("Press Enter to start trial...")
            input()
        
        # 重置trial状态
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
                # 检查trial时间
                if ILC_ENABLED:
                    elapsed_time = time.time() - trial_start_time
                    if elapsed_time > ILC_TRIAL_DURATION:
                        print(f"\n[Trial {trial_num + 1}] Time limit reached")
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
                
                # EMG处理
                filtered_Bicep = filter.bandpass(reading[0])
                filtered_Tricep = filter.bandpass(reading[1]) if len(reading) > 1 else 0.0
                
                # RMS计算
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
                
                # 低通滤波
                filtered_bicep_RMS = filter.lowpass(np.atleast_1d(Bicep_RMS))
                filtered_tricep_RMS = filter.lowpass(np.atleast_1d(Tricep_RMS))
                
                # 期望角度
                activation = interpreter.compute_activation(filtered_bicep_RMS, filtered_tricep_RMS)
                desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
                desired_angle_rad = math.radians(desired_angle_deg)
                
                # 期望角速度
                desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
                last_desired_angle = desired_angle_rad
                
                # 当前角速度
                current_velocity = (desired_angle_rad - current_angle) / dt if dt > 0 else 0.0
                current_angle += current_velocity * dt
                
                # ========== OIAC+ILC Control ==========
                
                position_error = desired_angle_rad - current_angle
                velocity_error = desired_velocity_rad - current_velocity
                
                # 1. OIAC
                K_mat, B_mat, integral = oiac.update_impedance(
                    current_angle, desired_angle_rad,
                    current_velocity, desired_velocity_rad,
                    dt
                )
                
                # 2. Impedance torque
                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])
                
                impedance_torque = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
                integral_torque = oiac.ki * integral
                
                # 3. ILC feedforward
                ff_torque = 0.0
                if ILC_ENABLED and trial_num > 0:
                    ff_torque = ilc.get_feedforward(trial_time, trial_num - 1)
                
                # 4. Total torque
                total_torque = ff_torque + impedance_torque + integral_torque
                
                # 5. 肌肉力优化
                bicep_force, tricep_force = muscle_estimator.estimate_muscle_forces(
                    Bicep_RMS, Tricep_RMS
                )
                
                force_penalty = muscle_estimator.calculate_force_penalty(
                    bicep_force, tricep_force, position_error, total_torque
                )
                
                final_torque = total_torque - force_penalty
                
                # 限制
                torque_clipped = np.clip(final_torque, TORQUE_MIN, TORQUE_MAX)
                
                # 记录
                trial_time_log.append(trial_time)
                trial_error_log.append(position_error)
                trial_torque_log.append(torque_clipped)
                trial_desired_angle_log.append(desired_angle_rad)
                trial_current_angle_log.append(current_angle)
                trial_bicep_force_log.append(bicep_force)
                trial_tricep_force_log.append(tricep_force)
                trial_k_log.append(float(K_mat[0, 0]))
                trial_b_log.append(float(B_mat[0, 0]))
                
                # 发送扭矩命令
                try:
                    command_queue.put_nowait((torque_clipped, current_angle))
                except queue.Full:
                    try:
                        command_queue.get_nowait()
                        command_queue.put_nowait((torque_clipped, current_angle))
                    except:
                        pass
                
                # 调试输出
                control_count += 1
                
                if current_time - last_debug_time > 2.0:
                    error_deg = math.degrees(position_error)
                    k_val = float(K_mat[0, 0])
                    b_val = float(B_mat[0, 0])
                    print(f"t={trial_time:.1f}s | "
                          f"Des={desired_angle_deg:6.1f}° | "
                          f"Cur={math.degrees(current_angle):6.1f}° | "
                          f"Err={error_deg:6.1f}° | "
                          f"τ={torque_clipped:6.2f}Nm | "
                          f"FF={ff_torque:6.2f} | "
                          f"K={k_val:5.1f} | "
                          f"B={b_val:5.1f}")
                    last_debug_time = current_time
                
                if current_time - last_force_debug_time > 3.0:
                    print(f"Muscle | "
                          f"Bi={bicep_force:6.2f}N | "
                          f"Tri={tricep_force:6.2f}N | "
                          f"Pen={force_penalty:6.4f}Nm")
                    last_force_debug_time = current_time
                
                last_time = current_time
        
        except KeyboardInterrupt:
            print(f"\n[Trial {trial_num + 1}] Interrupted")
            if not ILC_ENABLED:
                break
        
        # Trial统计
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
            
            print(f"Avg error: {math.degrees(avg_error):.2f}°")
            print(f"Max error: {math.degrees(max_error):.2f}°")
            print(f"Avg bicep: {avg_bicep:.2f}N, tricep: {avg_tricep:.2f}N")
            print(f"Avg K={avg_k:.1f}, B={avg_b:.1f}")
            print(f"Cycles: {control_count}")
            
            # ILC学习
            if ILC_ENABLED and trial_num < max_trials - 1:
                print(f"\nUpdating ILC...")
                ilc.update_learning(trial_time_log, trial_error_log, trial_torque_log)
                ilc.save_learning(ILC_SAVE_PATH)
                
                if math.degrees(avg_error) < 2.0:
                    print(f"\n🎉 Target achieved! <2°")
                    user_input = input("Continue? (y/n): ")
                    if user_input.lower() != 'y':
                        break
        else:
            print("No data collected")
        
        if not ILC_ENABLED:
            break
        
        print(f"\n{'='*60}\n")
    
    # 最终统计
    print("\n" + "="*60)
    print(f"FINAL STATS - {MOTOR_MODEL} TORQUE MODE")
    print("="*60)
    
    if ILC_ENABLED and len(all_trial_stats) > 0:
        print(f"\nCompleted {len(all_trial_stats)} trials")
        print("\nProgress:")
        for stats in all_trial_stats:
            print(f"  T{stats['trial']}: "
                  f"Err={stats['avg_error_deg']:.2f}°, "
                  f"K={stats['avg_k']:.1f}, "
                  f"B={stats['avg_b']:.1f}")
        
        if len(all_trial_stats) > 1:
            improvement = (all_trial_stats[0]['avg_error_deg'] - 
                          all_trial_stats[-1]['avg_error_deg'])
            print(f"\nImprovement: {improvement:.2f}° "
                  f"({all_trial_stats[0]['avg_error_deg']:.2f}° → "
                  f"{all_trial_stats[-1]['avg_error_deg']:.2f}°)")
    
    # 关闭
    print("\n" + "="*60)
    print("SHUTTING DOWN")
    print("="*60)
    stop_event.set()
    
    t_emg.join(timeout=2.0)
    t_torque.join(timeout=2.0)
    
    emg.stop()
    motor.close()
    
    raw_data.queue.clear()
    command_queue.queue.clear()
    
    print("\n✅ Goodbye!")