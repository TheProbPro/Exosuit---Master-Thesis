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


class OnlineImpedanceAdaptationController:
    """
    Online Impedance Adaptation Controller based on:
    Xiong, X., & Fang, C. (2023). An Online Impedance Adaptation Controller 
    for Decoding Skill Intelligence. Biomimetic Intelligence and Robotics, 3(2).
    """
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
    """增强的迭代学习控制器"""
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
        
        lr = 0.1
        
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
        print(f"      Learning rate: {lr}")
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
            'current_trial': self.current_trial,
            'reference_length': self.reference_length
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
            self.reference_length = data.get('reference_length', 5000)
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
            command = command_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            motor.sendMotorCommand(motor.motor_ids[0], command[1])
            motor_state['position'] = motor.get_position()[0]
            motor_state['velocity'] = motor.get_velocity()[0]
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)


def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


def run_control_loop(emg, motor, interpreter, filter_bicep, filter_tricep, 
                     raw_data, command_queue, motor_state, 
                     oiac, muscle_estimator, ilc=None, use_ilc=False):
    """
    统一的控制循环函数
    
    参数:
        use_ilc: 是否使用ILC前馈（运行模式时为True）
    """
    # 电机参数
    step = 1500.0 / 140.0
    motor_center = 2550
    
    # RMS队列
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    
    # 状态变量
    current_angle = math.radians(55.0)
    current_velocity = 0.0
    last_time = time.time()
    last_desired_angle = math.radians(55.0)
    
    # 调试输出控制
    last_debug_time = time.time()
    last_force_debug_time = time.time()
    
    # 运行模式的时间记录
    run_start_time = time.time() if use_ilc else None
    
    print(f"\n{'='*60}")
    print(f"Control Loop Active - {'RUN MODE (with ILC)' if use_ilc else 'TRAINING MODE'}")
    print(f"{'='*60}\n")
    
    while not stop_event.is_set():
        # 获取EMG数据
        try:
            reading = raw_data.get_nowait()
        except queue.Empty:
            time.sleep(0.001)
            continue
        
        current_time = time.time()
        dt = current_time - last_time
        
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
        activation = interpreter.compute_activation(filtered_bicep_RMS)
        desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
        desired_angle_rad = math.radians(desired_angle_deg)
        
        # 估计期望角速度
        desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
        last_desired_angle = desired_angle_rad
        
        # 获取当前状态
        current_velocity = motor_state['velocity']
        current_angle_deg = (motor_center - motor_state['position']) / step
        current_angle = math.radians(current_angle_deg)
        
        # 计算误差
        position_error = desired_angle_rad - current_angle
        velocity_error = desired_velocity_rad - current_velocity
        
        # OIAC: 更新阻抗参数
        K_mat, B_mat, integral = oiac.update_impedance(
            current_angle, desired_angle_rad,
            current_velocity, desired_velocity_rad,
            dt
        )
        
        # OIAC反馈扭矩
        pos_error_vec = np.array([[position_error]])
        vel_error_vec = np.array([[velocity_error]])
        impedance_torque = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
        integral_torque = oiac.ki * integral
        
        # ILC前馈（如果使用）
        ff_torque = 0.0
        if use_ilc and ilc is not None and run_start_time is not None:
            elapsed_time = current_time - run_start_time
            ff_torque = ilc.get_feedforward(elapsed_time)
        
        # 总扭矩
        total_torque = ff_torque + impedance_torque + integral_torque
        
        # 肌肉力估计和优化
        bicep_force, tricep_force = muscle_estimator.estimate_muscle_forces(
            Bicep_RMS, Tricep_RMS
        )
        force_penalty = muscle_estimator.calculate_force_penalty(
            bicep_force, tricep_force, position_error, total_torque
        )
        final_torque = total_torque - force_penalty
        
        # 扭矩限制
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
        
        # 调试输出
        if current_time - last_debug_time > 2.0:
            error_deg = math.degrees(position_error)
            k_val = float(K_mat[0, 0])
            b_val = float(B_mat[0, 0])
            mode_str = "RUN" if use_ilc else "TRAIN"
            print(f"[{mode_str}] Desired={desired_angle_deg:.1f}° | "
                  f"Current={math.degrees(current_angle):.1f}° | "
                  f"Error={error_deg:.2f}° | "
                  f"Torque={torque_clipped:.2f}Nm | "
                  f"FF={ff_torque:.2f}Nm | "
                  f"K={k_val:.1f} | B={b_val:.1f}")
            last_debug_time = current_time
        
        if current_time - last_force_debug_time > 3.0:
            print(f"[MUSCLE] Bicep: {bicep_force:.2f}N | "
                  f"Tricep: {tricep_force:.2f}N | "
                  f"Penalty: {force_penalty:.4f}Nm")
            last_force_debug_time = current_time
        
        last_time = current_time
    
    # 清理队列
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()


if __name__ == "__main__":
    print("=" * 60)
    print("EMG-based OIAC+ILC Control System")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print("=" * 60)
    
    # 创建队列
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    command_queue = queue.Queue(maxsize=10)
    motor_state = {'position': 0, 'velocity': 0}
    
    # 初始化传感器和控制器
    emg = DelsysEMG()
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, 
                     user_name=USER_NAME, BicepEMG=True, TricepEMG=False)
    interpreter.set_Kp(8)
    
    motor = Motors()
    oiac = OnlineImpedanceAdaptationController(dof=1)
    muscle_estimator = EMGMuscleForceEstimator()
    
    # 等待初始化
    time.sleep(1.0)
    
    # 启动EMG传感器
    emg.start()
    
    # 启动线程
    t_emg = threading.Thread(target=read_EMG, args=(emg, raw_data), daemon=True)
    t_motor = threading.Thread(target=send_motor_command, args=(motor, command_queue, motor_state), daemon=True)
    t_emg.start()
    t_motor.start()
    print("\nEMG and motor threads started!")
    
    # ========== 主菜单 ==========
    while not stop_event.is_set():
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Start NEW ILC Training")
        print("2. Continue EXISTING ILC Training")
        print("3. RUN with Trained ILC Data")
        print("4. Exit")
        print("="*60)
        
        choice = input("Your choice (1-4): ").strip()
        
        if choice == '4':
            print("Exiting...")
            break
        
        # ========== 选项1: 新训练 ==========
        if choice == '1':
            print("\n⚠️  Starting NEW training will overwrite existing data!")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
            
            ilc = EnhancedILC(max_trials=ILC_MAX_TRIALS)
            print(f"\n{'='*60}")
            print(f"Starting NEW ILC Training - {ILC_MAX_TRIALS} trials")
            print(f"Trial duration: {ILC_TRIAL_DURATION}s each")
            print(f"{'='*60}")
            print("⚠️  IMPORTANT: Please repeat the SAME movement pattern")
            print("   in each trial for effective ILC learning!")
            
        # ========== 选项2: 继续训练 ==========
        elif choice == '2':
            if not os.path.exists(ILC_SAVE_PATH):
                print(f"\n❌ No saved data found at {ILC_SAVE_PATH}")
                print("Please start NEW training first (option 1)")
                continue
            
            ilc = EnhancedILC(max_trials=ILC_MAX_TRIALS)
            if not ilc.load_learning(ILC_SAVE_PATH):
                print("Failed to load data, please try NEW training")
                continue
            
            remaining_trials = ILC_MAX_TRIALS - ilc.current_trial
            if remaining_trials <= 0:
                print(f"\n✅ Training already completed ({ilc.current_trial} trials)")
                print("Choose option 3 to RUN with trained data")
                continue
            
            print(f"\n{'='*60}")
            print(f"Continuing ILC Training")
            print(f"Completed: {ilc.current_trial} trials")
            print(f"Remaining: {remaining_trials} trials")
            print(f"{'='*60}")
        
        # ========== 选项3: 运行模式 ==========
        elif choice == '3':
            if not os.path.exists(ILC_SAVE_PATH):
                print(f"\n❌ No trained data found at {ILC_SAVE_PATH}")
                print("Please complete training first (option 1 or 2)")
                continue
            
            ilc = EnhancedILC(max_trials=ILC_MAX_TRIALS)
            if not ilc.load_learning(ILC_SAVE_PATH):
                print("Failed to load trained data")
                continue
            
            print(f"\n{'='*60}")
            print(f"RUN MODE - Using Trained ILC Data")
            print(f"{'='*60}")
            print(f"Loaded {ilc.current_trial} trained trials")
            print("Control: ILC Feedforward + OIAC Feedback + Muscle Optimization")
            print("\nPress Ctrl-C to stop...")
            print("="*60)
            
            input("Press Enter to start...")
            
            # 重置控制器
            oiac.reset()
            muscle_estimator.reset_history()
            
            try:
                run_control_loop(
                    emg, motor, interpreter, filter_bicep, filter_tricep,
                    raw_data, command_queue, motor_state,
                    oiac, muscle_estimator, ilc=ilc, use_ilc=True
                )
            except KeyboardInterrupt:
                print("\n\nRun mode stopped by user")
            
            continue
        
        else:
            print("Invalid choice, please try again")
            continue
        
        # ========== 训练循环（选项1或2）==========
        if choice in ['1', '2']:
            all_trial_stats = []
            start_trial = ilc.current_trial
            
            for trial_num in range(start_trial, ILC_MAX_TRIALS):
                print(f"\n{'='*60}")
                print(f"Trial {trial_num + 1}/{ILC_MAX_TRIALS}")
                print(f"{'='*60}")
                print("⚠️  Perform the SAME movement pattern as previous trials!")
                input("Press Enter to start trial...")
                
                # 重置trial状态
                oiac.reset()
                muscle_estimator.reset_history()
                
                # Trial数据记录
                trial_time_log = []
                trial_error_log = []
                trial_torque_log = []
                
                # 控制循环变量
                Bicep_RMS_queue = queue.Queue(maxsize=50)
                Tricep_RMS_queue = queue.Queue(maxsize=50)
                
                current_angle = math.radians(55.0)
                current_velocity = 0.0
                last_time = time.time()
                trial_start_time = time.time()
                last_desired_angle = math.radians(55.0)
                
                last_debug_time = time.time()
                control_count = 0
                
                step = 1500.0 / 140.0
                motor_center = 2550
                
                print(f"\n{'='*60}")
                print(f"Trial {trial_num + 1} - Control Active")
                print(f"{'='*60}\n")
                
                try:
                    while not stop_event.is_set():
                        elapsed_time = time.time() - trial_start_time
                        if elapsed_time > ILC_TRIAL_DURATION:
                            print(f"\n[Trial {trial_num + 1}] Duration reached")
                            break
                        
                        # EMG数据获取
                        try:
                            reading = raw_data.get_nowait()
                        except queue.Empty:
                            time.sleep(0.001)
                            continue
                        
                        current_time = time.time()
                        dt = current_time - last_time
                        trial_time = current_time - trial_start_time
                        
                        # 滤波
                        filtered_Bicep = filter_bicep.bandpass(reading[0])
                        filtered_Tricep = filter_tricep.bandpass(reading[1]) if len(reading) > 1 else  0.0
                        0.0
                        
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
                        
                        filtered_bicep_RMS = filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))
                        filtered_tricep_RMS = filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))
                        
                        # 计算期望角度
                        activation = interpreter.compute_activation(filtered_bicep_RMS)
                        desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
                        desired_angle_rad = math.radians(desired_angle_deg)
                        
                        desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
                        last_desired_angle = desired_angle_rad
                        
                        # 当前状态
                        current_velocity = motor_state['velocity']
                        current_angle_deg = (motor_center - motor_state['position']) / step
                        current_angle = math.radians(current_angle_deg)
                        
                        position_error = desired_angle_rad - current_angle
                        velocity_error = desired_velocity_rad - current_velocity
                        
                        # OIAC控制
                        K_mat, B_mat, integral = oiac.update_impedance(
                            current_angle, desired_angle_rad,
                            current_velocity, desired_velocity_rad,
                            dt
                        )
                        
                        pos_error_vec = np.array([[position_error]])
                        vel_error_vec = np.array([[velocity_error]])
                        impedance_torque = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
                        integral_torque = oiac.ki * integral
                        
                        # ILC前馈（如果不是第一个trial）
                        ff_torque = 0.0
                        if trial_num > 0:
                            ff_torque = ilc.get_feedforward(trial_time, trial_num - 1)
                        
                        total_torque = ff_torque + impedance_torque + integral_torque
                        
                        # 肌肉力优化
                        bicep_force, tricep_force = muscle_estimator.estimate_muscle_forces(
                            Bicep_RMS, Tricep_RMS
                        )
                        force_penalty = muscle_estimator.calculate_force_penalty(
                            bicep_force, tricep_force, position_error, total_torque
                        )
                        final_torque = total_torque - force_penalty
                        torque_clipped = np.clip(final_torque, TORQUE_MIN, TORQUE_MAX)
                        
                        # 记录数据
                        trial_time_log.append(trial_time)
                        trial_error_log.append(position_error)
                        trial_torque_log.append(torque_clipped)
                        
                        # 电机命令
                        position_motor = motor_center - int(desired_angle_deg * step)
                        try:
                            command_queue.put_nowait((torque_clipped, position_motor))
                        except queue.Full:
                            try:
                                command_queue.get_nowait()
                                command_queue.put_nowait((torque_clipped, position_motor))
                            except:
                                pass
                        
                        control_count += 1
                        
                        # 调试输出
                        if current_time - last_debug_time > 2.0:
                            error_deg = math.degrees(position_error)
                            k_val = float(K_mat[0, 0])
                            b_val = float(B_mat[0, 0])
                            print(f"t={trial_time:.1f}s | "
                                  f"Desired={desired_angle_deg:.1f}° | "
                                  f"Current={math.degrees(current_angle):.1f}° | "
                                  f"Error={error_deg:.2f}° | "
                                  f"Torque={torque_clipped:.2f}Nm | "
                                  f"FF={ff_torque:.2f}Nm | "
                                  f"K={k_val:.1f} | B={b_val:.1f}")
                            last_debug_time = current_time
                        
                        last_time = current_time
                
                except KeyboardInterrupt:
                    print(f"\n[Trial {trial_num + 1}] Interrupted by user")
                
                # Trial统计
                print(f"\n{'='*60}")
                print(f"Trial {trial_num + 1} Summary")
                print(f"{'='*60}")
                
                if len(trial_error_log) > 0:
                    avg_error = np.mean(np.abs(trial_error_log))
                    max_error = np.max(np.abs(trial_error_log))
                    avg_bicep, avg_tricep, _, _ = muscle_estimator.get_force_statistics()
                    avg_k = np.mean([float(K_mat[0, 0])])
                    avg_b = np.mean([float(B_mat[0, 0])])
                    
                    trial_stats = {
                        'trial': trial_num + 1,
                        'avg_error_deg': math.degrees(avg_error),
                        'max_error_deg': math.degrees(max_error),
                        'avg_bicep_force': avg_bicep,
                        'avg_tricep_force': avg_tricep,
                        'control_cycles': control_count
                    }
                    all_trial_stats.append(trial_stats)
                    
                    print(f"Average error: {math.degrees(avg_error):.2f}°")
                    print(f"Maximum error: {math.degrees(max_error):.2f}°")
                    print(f"Control cycles: {control_count}")
                    
                    # ILC学习更新
                    if trial_num < ILC_MAX_TRIALS - 1:
                        print(f"\nUpdating ILC learning...")
                        ilc.update_learning(trial_time_log, trial_error_log, trial_torque_log)
                        ilc.save_learning(ILC_SAVE_PATH)
                        
                        if math.degrees(avg_error) < 2.0:
                            print(f"\n🎉 Target performance achieved! Avg error < 2°")
                            continue_training = input("Continue training? (y/n): ").strip().lower()
                            if continue_training != 'y':
                                break
                    else:
                        # 最后一个trial也要更新
                        print(f"\nFinal trial - Updating ILC learning...")
                        ilc.update_learning(trial_time_log, trial_error_log, trial_torque_log)
                        ilc.save_learning(ILC_SAVE_PATH)
                else:
                    print("No data collected in this trial")
                
                # 清理队列
                Bicep_RMS_queue.queue.clear()
            
            # 训练完成统计
            print("\n" + "="*60)
            print("TRAINING COMPLETED")
            print("="*60)
            
            if len(all_trial_stats) > 0:
                print(f"\nCompleted {len(all_trial_stats)} trials")
                print("\nLearning Progress:")
                for stats in all_trial_stats:
                    print(f"  Trial {stats['trial']}: "
                          f"Avg Error={stats['avg_error_deg']:.2f}°, "
                          f"Max Error={stats['max_error_deg']:.2f}°")
                
                if len(all_trial_stats) > 1:
                    improvement = (all_trial_stats[0]['avg_error_deg'] - 
                                  all_trial_stats[-1]['avg_error_deg'])
                    print(f"\n✅ Error improvement: {improvement:.2f}° "
                          f"({all_trial_stats[0]['avg_error_deg']:.2f}° → "
                          f"{all_trial_stats[-1]['avg_error_deg']:.2f}°)")
                
                print(f"\n💾 Training data saved to: {ILC_SAVE_PATH}")
                print("You can now use option 3 to RUN with trained data!")
    
    # ========== 系统关闭 ==========
    print("\n" + "="*60)
    print("SHUTTING DOWN SYSTEM")
    print("="*60)
    stop_event.set()
    
    t_emg.join(timeout=2.0)
    t_motor.join(timeout=2.0)
    
    emg.stop()
    motor.close()
    
    raw_data.queue.clear()
    command_queue.queue.clear()
    
    print("\n✅ System shutdown complete. Goodbye!")
