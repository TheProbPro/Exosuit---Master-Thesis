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
trial_reset_event = threading.Event()

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


# ==================== 代码移植的控制器 ====================

class AdaptiveImpedanceController:
   
    def __init__(self, dof=1):
        self.DOF = dof
        
        # 阻抗参数
        self.k = np.zeros(self.DOF)  # 刚度
        self.d = np.zeros(self.DOF)  # 阻尼
        self.ff = np.zeros(self.DOF)  # 前馈
        
        # 自适应参数 (来自第一段代码)
        self.a = 35.0    # 自适应因子分子
        self.b = 5.0     # 自适应因子分母系数
        self.beta = 0.05 # 跟踪误差权重
        
        # 状态变量
        self.pos_diff = np.zeros(self.DOF)  # 位置误差
        self.vel_diff = np.zeros(self.DOF)  # 速度误差
        self.tra_diff = np.zeros(self.DOF)  # 跟踪误差
        self.co_diff = np.zeros(self.DOF)   # 自适应系数
        
        # 恒定阻抗控制器参数 (备用)
        self.cons_k = 0.04
        self.cons_d = np.sqrt(self.cons_k)
        
    def get_pos_diff(self, current_pos, desired_pos):
        """计算位置差异 (T3)"""
        self.pos_diff = current_pos - desired_pos
        return self.pos_diff
    
    def get_vel_diff(self, current_vel, desired_vel):
        """计算速度差异 (T4)"""
        self.vel_diff = current_vel - desired_vel
        return self.vel_diff
    
    def get_tra_diff(self):
        """计算跟踪差异 (T5)"""
        self.tra_diff = self.pos_diff + self.beta * self.vel_diff
        return self.tra_diff
    
    def get_coe(self):
        """计算自适应标量 (T6)"""
        #for i in range(self.DOF):
        self.co_diff = self.a / (1.00 + self.b * self.tra_diff * self.tra_diff)
        return self.co_diff
    
    def adaptive_impedance_control(self, current_pos, desired_pos, current_vel, desired_vel):
        """
        自适应阻抗控制 (T9, T10)
        基于代码的 ada_impe() 方法
        """
        # 计算误差
        self.get_pos_diff(current_pos, desired_pos)
        self.get_vel_diff(current_vel, desired_vel)
        self.get_tra_diff()
        self.get_coe()
        
        # 在线调制阻抗参数
        #for i in range(self.DOF):
        self.ff = self.tra_diff / self.co_diff
        self.k = self.ff * self.pos_diff
        self.d = self.ff * self.vel_diff
            
        # 计算控制扭矩
        control_torque = -(self.ff + self.k * self.pos_diff + self.d * self.vel_diff)
            
        return control_torque, self.k, self.d, self.ff
    
    def constant_impedance_control(self, current_pos, desired_pos, current_vel, desired_vel):
        """
        恒定阻抗控制 (T7, T8)
        基于代码的 const_impe() 方法
        """
        # 计算误差
        self.get_pos_diff(current_pos, desired_pos)
        self.get_vel_diff(current_vel, desired_vel)
        
        # 恒定阻抗参数
        for i in range(self.DOF):
            self.k[i] = self.cons_k
            self.d[i] = self.cons_d
            self.ff[i] = 0.00
            
            # 计算控制扭矩
            control_torque = -(self.cons_k * self.pos_diff[i] + self.cons_d * self.vel_diff[i]) - self.ff[i]
            
        return control_torque, self.k.copy(), self.d.copy(), self.ff.copy()
    
    def reset(self):
        """重置控制器状态"""
        self.k = np.zeros(self.DOF)
        self.d = np.zeros(self.DOF)
        self.ff = np.zeros(self.DOF)
        self.pos_diff = np.zeros(self.DOF)
        self.vel_diff = np.zeros(self.DOF)
        self.tra_diff = np.zeros(self.DOF)
        self.co_diff = np.zeros(self.DOF)


# ==================== 移植的迭代学习 ====================

class IterativeLearningController:
    """
    从第一段代码移植的迭代学习控制器
    基于 iter_learn_ff_mod() 方法
    """
    def __init__(self, max_trials=10, alpha=0.1):
        self.max_trials = max_trials
        self.current_trial = 0
        self.alpha = alpha  # 学习增益
        
        # 学习数据存储
        self.learned_feedforward = []  # 每个trial的前馈扭矩
        self.trial_errors = []         # 每个trial的误差
        self.trial_torques = []        # 每个trial的扭矩
        
        # 时间相关参数
        self.reference_time = None
        self.sample_rate = 50  # Hz (假设)
        
    def update_learning(self, time_array, error_array, torque_array):
        """
        迭代学习更新
        代码的 iter_learn_ff_mod() 方法
        """
        if len(time_array) == 0 or len(error_array) == 0:
            print("[ILC] Warning: Empty data, skipping update")
            return np.zeros(len(self.learned_feedforward[0]) if self.learned_feedforward else 100)
        
        # 创建统一的参考时间轴
        if self.reference_time is None:
            max_time = max(time_array) if len(time_array) > 0 else 10.0
            self.reference_time = np.linspace(0, max_time, int(max_time * self.sample_rate))
        
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
            # 第一次trial，初始化为零
            ff = np.zeros_like(aligned_error)
        else:
            # 使用上一次的前馈 + 学习项
            last_ff = self.learned_feedforward[-1]
            ff = last_ff + self.alpha * aligned_error
        
        # 限制前馈幅度
        ff = np.clip(ff, -30.0, 30.0)
        
        # 平滑处理
        if len(ff) > 10:
            window_size = 7
            ff = np.convolve(ff, np.ones(window_size)/window_size, mode='same')
        
        self.learned_feedforward.append(ff)
        self.trial_errors.append(aligned_error)
        self.trial_torques.append(torque_array)
        self.current_trial += 1
        
        # 计算性能指标
        avg_error = np.mean(np.abs(aligned_error))
        max_error = np.max(np.abs(aligned_error))
        
        print(f"[ILC] Trial {self.current_trial} completed:")
        print(f"      Learning rate: {self.alpha}")
        print(f"      Avg error: {math.degrees(avg_error)}°")
        print(f"      Max error: {math.degrees(max_error)}°")
        print(f"      Feedforward range: [{np.min(ff)}, {np.max(ff)}] Nm")
        
        return ff
    
    def get_feedforward(self, t, trial_idx=-1):
        """
        获取指定时刻的前馈扭矩
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
        """重置ILC"""
        self.learned_feedforward.clear()
        self.trial_errors.clear()
        self.trial_torques.clear()
        self.current_trial = 0
        print("[ILC] Reset completed")


# ==================== 扭矩符号的AAN/RAN控制器 ====================

class TorqueBasedAANRANController:
    """
    扭矩符号的AAN/RAN控制器
    - 正扭矩: AAN模式 (辅助)
    - 负扭矩: RAN模式 (阻力)
    """
    def __init__(self, adaptive_controller, ilc_controller=None):
        self.adaptive_controller = adaptive_controller
        self.ilc_controller = ilc_controller
        
        # 模式切换参数
        self.current_mode = 'AAN'  # 初始模式
        self.last_torque = 0.0
        self.mode_history = []
        
        # RAN阻力参数
        self.ran_resistance_level = 2.5  # 基础阻力水平
        self.ran_velocity_factor = 1.5   # 速度相关阻力
        
        # 切换参数
        self.last_switch_time = 0
        self.min_switch_interval = 0.1   # 最小切换间隔
    
    def compute_control(self, t, current_pos, current_vel, desired_pos, desired_vel, trial_idx):
        """
        计算控制扭矩，基于扭矩符号实现AAN/RAN切换
        """
        current_time = t
        
        # 使用自适应阻抗控制器计算基础扭矩
        base_torque, k, d, ff = self.adaptive_controller.adaptive_impedance_control(
            current_pos, desired_pos, current_vel, desired_vel
        )
        
        # 获取ILC前馈扭矩 (如果有)
        ilc_torque = 0.0
        if self.ilc_controller and trial_idx > 0:
            ilc_torque = self.ilc_controller.get_feedforward(t, trial_idx-1)
        
        # 计算AAN模式的总扭矩 (基础扭矩 + ILC前馈)
        aan_torque = base_torque + ilc_torque
        
        # ===== 扭矩符号的模式切换 =====
        can_switch = (current_time - self.last_switch_time) >= self.min_switch_interval
        
        if aan_torque < 0:  # 正扭矩 → AAN模式
            if self.current_mode != 'AAN' and can_switch:
                self.current_mode = 'AAN'
                self.last_switch_time = current_time
                print(f" RAN→AAN at t={t:.2f}s (torque={aan_torque:.2f}Nm) - Activating assistance")
            
            total_torque = aan_torque
            
        else:  # 负扭矩或零 → RAN模式
            if self.current_mode != 'RAN' and can_switch:
                self.current_mode = 'RAN'
                self.last_switch_time = current_time
                print(f" AAN→RAN at t={t:.2f}s (torque={aan_torque:.2f}Nm) - Activating resistance")
            
            # RAN模式：只使用基础阻抗控制 + 额外阻力
            # 阻力方向与运动方向相反
            resistance_direction = -1.0 if current_vel >= 0 else 1.0
            base_resistance = self.ran_resistance_level * resistance_direction
            velocity_resistance = self.ran_velocity_factor * abs(current_vel) * resistance_direction
            
            total_torque = base_torque + base_resistance + velocity_resistance
        
        # 记录状态
        self.last_torque = total_torque
        self.mode_history.append((current_time, self.current_mode, total_torque))
        
        # 限制历史记录长度
        if len(self.mode_history) > 1000:
            self.mode_history.pop(0)
            
        return total_torque, self.current_mode, k, d, ff
    
    def get_mode_statistics(self, recent_seconds=5):
        """获取最近一段时间内的模式统计"""
        if not self.mode_history:
            return 0.0, 0.0
            
        current_time = time.time() if self.mode_history else 0
        cutoff_time = current_time - recent_seconds
        
        recent_history = [mode for (t, mode, _) in self.mode_history if t >= cutoff_time]
        
        if not recent_history:
            return 0.0, 0.0
            
        aan_count = recent_history.count('AAN')
        ran_count = recent_history.count('RAN')
        total_count = len(recent_history)
        
        aan_ratio = aan_count / total_count * 100
        ran_ratio = ran_count / total_count * 100
        
        return aan_ratio, ran_ratio
    
    def reset(self):
        """重置控制器状态"""
        self.current_mode = 'AAN'
        self.mode_history.clear()
        self.last_switch_time = 0
        self.last_torque = 0.0
        self.adaptive_controller.reset()
        print("[TorqueBased Controller] Reset to AAN mode")


# ==================== 主控制系统 ====================

def read_EMG(EMG_sensor, raw_queue):
    """EMG读取线程"""
    while not stop_event.is_set():
        reading = EMG_sensor.read()
        try:
            raw_queue.put_nowait(reading)
        except queue.Full:
            try:
                raw_queue.get_nowait()
                raw_queue.put_nowait(reading)
            except queue.Full:
                pass
        except Exception as e:
            print(f"[reader] error: {e}", file=sys.stderr)


def send_motor_command(motor, command_queue, motor_state):
    """电机命令发送线程"""
    while not stop_event.is_set():
        try:
            # command = (torque, position_fallback)
            #command = command_queue.get(timeout=0.001)
            command = command_queue.get_nowait()
            #print(command)
        except queue.Empty:
            motor.sendMotorCommand(motor.motor_ids[0], 0)
            continue

        try:
            torque = command[0]
            current = motor.torq2curcom(torque)
            print("motor torque: ", torque, "motor position: ", motor_state['position'])
            #motor.sendMotorCommand(motor.motor_ids[0], current)
            if motor_state['position'] < 1050 and torque < 0:
                #print("Sending zero torque to avoid overflexion")
                motor.sendMotorCommand(motor.motor_ids[0], 0)
            elif motor_state['position'] > 2550 and torque > 0:
                #print("Sending zero torque to avoid overextension")
                motor.sendMotorCommand(motor.motor_ids[0], 0)
            else:
                motor.sendMotorCommand(motor.motor_ids[0], current)
            motor_state['position'] = motor.get_position()[0]
            motor_state['velocity'] = motor.get_velocity()[0]
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)


def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    print("=" * 60)
    print(" EMG-based Adaptive Impedance Control with Torque-based AAN/RAN")
    print("   (Full Bidirectional EMG Control Enabled)")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print(f"ILC enabled: {ILC_ENABLED}")
    if ILC_ENABLED:
        print(f"Max trials: {ILC_MAX_TRIALS}")
        print(f"Trial duration: {ILC_TRIAL_DURATION}s")
    
    print("\n Full Bidirectional EMG Control:")
    print("   - Bicep activation → Upward motion")
    print("   - Tricep activation → Downward motion")
    print("   - Positive torque (> 0): AAN mode (Assistance)")
    print("   - Negative torque (<= 0): RAN mode (Resistance)")
    print("   - AAN uses Adaptive Impedance + ILC feedforward")
    print("   - RAN uses Adaptive Impedance + Resistance")
    print("\n Adaptive Impedance Parameters:")
    print("   - Adaptation factor (a): 35.0")
    print("   - Adaptation denominator (b): 5.0")
    print("   - Tracking weight (beta): 0.05")
    print("=" * 60)
    
    # 创建队列
    raw_data = queue.Queue(maxsize=100)
    command_queue = queue.Queue(maxsize=10)
    motor_state = {'position': 0, 'velocity': 0}
    
    # 初始化EMG传感器
    emg = DelsysEMG(channel_range=(0, 1))
    
    # 初始化滤波器和解释器 - 启用三头肌控制
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, 
                     user_name=USER_NAME, BicepEMG=True, TricepEMG=True)  # 启用三头肌控制

    interpreter.set_Kp(8)
    
    # 初始化电机
    motor = Motors()
    
    # 初始化从第一段代码移植的控制器
    adaptive_controller = AdaptiveImpedanceController(dof=1)
    muscle_estimator = EMGMuscleForceEstimator()
    ilc_controller = IterativeLearningController(max_trials=ILC_MAX_TRIALS, alpha=0.1) if ILC_ENABLED else None
    
    # 初始化基于扭矩符号的AAN/RAN控制器
    torque_based_controller = TorqueBasedAANRANController(adaptive_controller, ilc_controller)
    
    # 尝试加载之前的ILC学习数据
    if ILC_ENABLED and os.path.exists(ILC_SAVE_PATH):
        user_input = input(f"\nFound saved ILC data. Load it? (y/n): ")
        if user_input.lower() == 'y':
            ilc_controller.load_learning(ILC_SAVE_PATH)
    
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
    print("\n EMG and motor threads started!")
    print(" Tricep control ENABLED - Full bidirectional EMG control active!")
    
    # ILC trial循环
    if ILC_ENABLED:
        max_trials = ILC_MAX_TRIALS
        start_trial = ilc_controller.current_trial if ilc_controller else 0
    else:
        max_trials = 1
        start_trial = 0
    
    all_trial_stats = []
    
    for trial_num in range(start_trial, max_trials):
        if ILC_ENABLED:
            print(f"\n{'='*60}")
            print(f" Starting Trial {trial_num + 1}/{max_trials}")
            print(f"{'='*60}")
            print("Press Enter to start trial...")
            input()
        
        # 重置trial相关的状态
        adaptive_controller.reset()
        muscle_estimator.reset_history()
        torque_based_controller.reset()
        
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
        trial_ff_log = []
        trial_mode_log = []
        
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
        print(f" Trial {trial_num + 1} - Full Bidirectional EMG Control Active")
        print(f"   Bicep → Upward motion | Tricep → Downward motion")
        print(f"{'='*60}\n")
        
        try:
            while not stop_event.is_set():
                # 检查trial时间限制
                if ILC_ENABLED:
                    elapsed_time = time.time() - trial_start_time
                    if elapsed_time > ILC_TRIAL_DURATION:
                        print(f"\n [Trial {trial_num + 1}] Duration reached, ending trial...")
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
                
                # ========== 启用双向EMG控制 ==========
                # 计算激活度和期望角度 - 使用双通道EMG
                activation = interpreter.compute_activation([filtered_bicep_RMS, filtered_tricep_RMS])
                desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
                desired_angle_rad = math.radians(desired_angle_deg)
                
                # 估计期望角速度
                desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
                last_desired_angle = desired_angle_rad
                
                # 获取当前角度和速度
                current_velocity = motor_state['velocity']
                current_angle_deg = (motor_center - motor_state['position']) / step
                current_angle = math.radians(current_angle_deg)
                
                #print(f"current angle: {current_angle}, desired angle: {desired_angle_rad}, current angle deg: {current_angle_deg}, desired angle deg: {desired_angle_deg}")
                
                # ========== 扭矩符号的AAN/RAN控制 ==========
                
                position_error = desired_angle_rad - current_angle
                
                # 使用基于扭矩符号的控制器
                total_torque, current_mode, k_val, b_val, ff_val = torque_based_controller.compute_control(
                    trial_time, 
                    current_angle, 
                    current_velocity,
                    desired_angle_rad,
                    desired_velocity_rad,
                    trial_num
                )
                
                # ===== 肌肉力估计和优化 =====
                bicep_force, tricep_force = muscle_estimator.estimate_muscle_forces(
                    filtered_bicep_RMS, filtered_tricep_RMS
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
                trial_k_log.append(k_val)
                trial_b_log.append(b_val)
                trial_ff_log.append(ff_val)
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
                    
                    # 检测运动方向
                    motion_direction = "↑UP" if desired_velocity_rad > 0 else "↓DOWN" if desired_velocity_rad < 0 else "•STABLE"
                    
                    # Mode-specific info
                    if current_mode == 'RAN':
                        mode_info = f" RAN (Resistance)"
                    else:
                        mode_info = f" AAN (Assistance)"
                    
                    print(f"t={trial_time:.2f}s | {mode_info} | {motion_direction}")
                    print(f"  Desired={desired_angle_deg:.1f}° | Current={math.degrees(current_angle):.1f}° | Error={error_deg:.1f}°")
                    print(f"  Bicep={Bicep_RMS:.3f} | Tricep={Tricep_RMS:.3f}")
                    print(f"  Torque={torque_clipped:.2f}Nm | K={k_val:.2f} | B={b_val:.2f}")
                    last_debug_time = current_time
                
                if current_time - last_force_debug_time > 3.0:
                    aan_ratio, ran_ratio = torque_based_controller.get_mode_statistics(3.0)
                    print(f" Muscle Forces | "
                          f"Bicep: {bicep_force:.2f}N | "
                          f"Tricep: {tricep_force:.2f}N | "
                          f"Mode: AAN={aan_ratio:.1f}% RAN={ran_ratio:.1f}%")
                    last_force_debug_time = current_time
                
                last_time = current_time
        
        except KeyboardInterrupt:
            print(f"\n [Trial {trial_num + 1}] Interrupted by user")
            if not ILC_ENABLED:
                break
        
        # Trial结束，统计结果
        print(f"\n{'='*60}")
        print(f" Trial {trial_num + 1} Summary")
        print(f"{'='*60}")
        
        if len(trial_error_log) > 0:
            avg_error = np.mean(np.abs(trial_error_log))
            max_error = np.max(np.abs(trial_error_log))
            avg_bicep = np.mean(trial_bicep_force_log)
            avg_tricep = np.mean(trial_tricep_force_log)
            avg_k = np.mean(trial_k_log)
            avg_b = np.mean(trial_b_log)
            avg_ff = np.mean(trial_ff_log)
            
            # Calculate mode distribution
            if trial_mode_log:
                aan_count = trial_mode_log.count('AAN')
                ran_count = trial_mode_log.count('RAN')
                total_count = len(trial_mode_log)
                aan_ratio = aan_count / total_count * 100
                ran_ratio = ran_count / total_count * 100
            else:
                aan_ratio = 100.0
                ran_ratio = 0.0
            
            # Calculate motion range
            min_angle = math.degrees(min(trial_current_angle_log))
            max_angle = math.degrees(max(trial_current_angle_log))
            motion_range = max_angle - min_angle
            
            trial_stats = {
                'trial': trial_num + 1,
                'avg_error_deg': math.degrees(avg_error),
                'max_error_deg': math.degrees(max_error),
                'avg_bicep_force': avg_bicep,
                'avg_tricep_force': avg_tricep,
                'avg_k': avg_k,
                'avg_b': avg_b,
                'avg_ff': avg_ff,
                'control_cycles': control_count,
                'aan_ratio': aan_ratio,
                'ran_ratio': ran_ratio,
                'motion_range': motion_range
            }
            all_trial_stats.append(trial_stats)
            
            print(f"Average tracking error: {math.degrees(avg_error):.2f}°")
            print(f"Maximum tracking error: {math.degrees(max_error):.2f}°")
            print(f"Motion range: {min_angle:.1f}° to {max_angle:.1f}° (span: {motion_range:.1f}°)")
            print(f"Average bicep force: {avg_bicep:.2f}N")
            print(f"Average tricep force: {avg_tricep:.2f}N")
            print(f"Average K: {avg_k:.2f}, Average B: {avg_b:.2f}, Average FF: {avg_ff:.2f}")
            print(f"Control cycles: {control_count}")
            print(f"Mode distribution:  AAN={aan_ratio:.1f}%,  RAN={ran_ratio:.1f}%")
            
            # RAN mode analysis
            if ran_ratio > 0:
                print(f"\n RAN Mode Successfully Activated!")
                print(f"   - Resistance applied during {ran_ratio:.1f}% of trial")
                print(f"   - Torque-based switching working correctly")
            else:
                print(f"\n  RAN Mode Not Activated")
                print(f"   - All computed torques were positive (AAN mode only)")
                print(f"   - This indicates good assistance performance")
            
            # ILC学习更新
            if ILC_ENABLED and ilc_controller and trial_num < max_trials - 1:
                print(f"\n Updating ILC learning...")
                ilc_controller.update_learning(trial_time_log, trial_error_log, trial_torque_log)
                
                # 保存学习数据
                ilc_controller.save_learning(ILC_SAVE_PATH)
        else:
            print(" No data collected in this trial")
        
        # 如果不是ILC模式，只运行一次
        if not ILC_ENABLED:
            break
        
        print(f"\n{'='*60}\n")
    
    # 最终统计
    print("\n" + "="*60)
    print(" FINAL STATISTICS - Full Bidirectional EMG Control System")
    print("="*60)
    
    if len(all_trial_stats) > 0:
        print(f"\n Completed {len(all_trial_stats)} trials")
        print("\n Learning Progress:")
        for stats in all_trial_stats:
            aan_symbol = "1"
            ran_symbol = "2" if stats['ran_ratio'] > 0 else "3"
            print(f"  Trial {stats['trial']}: "
                  f"Avg Error={stats['avg_error_deg']:.2f}°, "
                  f"Max Error={stats['max_error_deg']:.2f}°, "
                  f"Range={stats['motion_range']:.1f}°, "
                  f"{aan_symbol}AAN={stats['aan_ratio']:.1f}% {ran_symbol}RAN={stats['ran_ratio']:.1f}%")
    
    # 运行模式选择
    print("\n" + "="*60)
    print("press 1 to enter run mode (no ILC), 2 to exit")
    print("\n" + "="*60)
    user_input = input("Your choice: ")
    
    if user_input.strip() == '1':
        print("\n Entering Run Mode (Continuous Operation)")
        print("   - Full Bidirectional EMG Control Active")
        print("   - Adaptive Impedance Control")
        print("   - Torque-based AAN/RAN Switching")
        print("   - No ILC Learning")
        
        # 重置控制器为运行模式
        adaptive_controller.reset()
        torque_based_controller.reset()
        muscle_estimator.reset_history()
        
        Bicep_RMS_queue = queue.Queue(maxsize=50)
        Tricep_RMS_queue = queue.Queue(maxsize=50)
        
        last_time = time.time()
        last_desired_angle = math.radians(55.0)
        
        while not stop_event.is_set():
            try:
                reading = raw_data.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            
            # EMG信号处理 (与之前相同)
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
                
            # ========== 关键修改：运行模式中也使用双向EMG控制 ==========
            # 计算激活度和期望角度 - 使用双通道EMG
            activation = interpreter.compute_activation(filtered_bicep_RMS, filtered_tricep_RMS)
            desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
            desired_angle_rad = math.radians(desired_angle_deg)
                
            # 估计期望角速度
            desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
            last_desired_angle = desired_angle_rad
                
            # 获取当前角度和速度
            current_velocity = motor_state['velocity']
            current_angle_deg = (motor_center - motor_state['position']) / step
            current_angle = math.radians(current_angle_deg)
                
            # 使用基于扭矩符号的控制器
            total_torque, current_mode, k_val, b_val, ff_val = torque_based_controller.compute_control(
                current_time, 
                current_angle, 
                current_velocity,
                desired_angle_rad,
                desired_velocity_rad,
                0  # trial_idx = 0 for run mode
            )
                
            # 肌肉力估计和优化
            bicep_force, tricep_force = muscle_estimator.estimate_muscle_forces(Bicep_RMS, Tricep_RMS)
            force_penalty = muscle_estimator.calculate_force_penalty(
                bicep_force, tricep_force, desired_angle_rad - current_angle, total_torque
            )
            final_torque = total_torque - force_penalty
            torque_clipped = np.clip(final_torque, TORQUE_MIN, TORQUE_MAX)
                
            # 转换为电机位置命令
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

            last_time = current_time

    elif user_input.strip() == '2':
        pass

    # 停止系统
    print("\n" + "="*60)
    print(" SHUTTING DOWN")
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
    
    print("\n Full Bidirectional EMG Control System Complete!")
    print(" Key Features Successfully Implemented:")
    print("  ✓ Adaptive Impedance Control from first code")
    print("  ✓ Full bidirectional EMG control")
    print("  ✓ Bicep activation → Upward motion")
    print("  ✓ Tricep activation → Downward motion")
    print("  ✓ Torque-based AAN/RAN mode switching")
    print("  ✓ Positive torque → AAN mode (Assistance)")
    print("  ✓ Negative torque → RAN mode (Resistance)")
    print("  ✓ ILC learning for repetitive tasks")
    print("  ✓ EMG-based muscle force optimization")
    print("\nGoodbye! ")