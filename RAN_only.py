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

SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

# Control parameters
TORQUE_MIN = -4.1  # Nm
TORQUE_MAX = 4.1   # Nm

stop_event = threading.Event()

class RAN_OIAC_Controller:
    """
    基于OIAC的RAN模式控制器
    使用OIAC框架实现可调节的阻力辅助
    """
    def __init__(self, dof=1):
        self.DOF = dof
        
        # OIAC阻抗参数 - 初始值
        self.K = np.eye(dof) * 80.0  # 刚度矩阵
        self.B = np.eye(dof) * 20.0  # 阻尼矩阵
        
        # RAN模式参数
        self.ran_enabled = True
        self.resistance_level = 2.0  # 基础阻力水平
        self.adaptive_gain = 0.8     # 自适应增益
        
        # OIAC自适应参数
        self.alpha_k = 0.1  # 刚度自适应系数
        self.alpha_b = 0.1  # 阻尼自适应系数
        self.gamma = 0.5    # 跟踪误差权重
        
        # 状态变量
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))
        
        # 性能监控
        self.tracking_errors = []
        self.impedance_history = []
        self.resistance_torques = []
        
    def update_impedance(self, q, q_d, dq, dq_d):
        """
        基于OIAC原理更新阻抗参数
        """
        # 更新状态
        self.q = np.array([[q]])
        self.q_d = np.array([[q_d]])
        self.dq = np.array([[dq]])
        self.dq_d = np.array([[dq_d]])
        
        # 计算误差
        e = q_d - q
        de = dq_d - dq
        
        # 跟踪误差范数
        e_norm = abs(e)
        de_norm = abs(de)
        
        # OIAC自适应律 - 根据跟踪性能调整阻抗
        if e_norm > 0.1:  # 较大跟踪误差
            # 增加阻抗以提供更多稳定性
            K_update = self.alpha_k * e_norm
            B_update = self.alpha_b * de_norm
        else:
            # 小误差时，根据RAN模式调整
            if self.ran_enabled:
                # RAN模式：适度增加阻抗提供阻力
                K_update = self.alpha_k * 0.5
                B_update = self.alpha_b * 0.8
            else:
                # 正常模式：保持或降低阻抗
                K_update = -self.alpha_k * 0.2
                B_update = -self.alpha_b * 0.2
        
        # 更新阻抗参数
        self.K[0,0] = np.clip(self.K[0,0] + K_update, 50, 200)
        self.B[0,0] = np.clip(self.B[0,0] + B_update, 10, 80)
        
        # 记录阻抗变化
        self.impedance_history.append((self.K[0,0], self.B[0,0]))
        if len(self.impedance_history) > 100:
            self.impedance_history.pop(0)
            
        return self.K[0,0], self.B[0,0]
    
    def compute_ran_resistance(self, q, dq, q_d):
        """
        基于OIAC框架计算RAN阻力
        """
        if not self.ran_enabled:
            return 0.0
        
        # 基础阻力项
        base_resistance = self.resistance_level
        
        # 速度相关阻力 (使用阻尼矩阵)
        velocity_resistance = self.B[0,0] * 0.02 * abs(dq)
        
        # 位置相关阻力 (使用刚度矩阵)
        position_error = q - q_d
        position_resistance = self.K[0,0] * 0.01 * abs(position_error)
        
        # 阻力方向 (总是抵抗运动)
        resistance_direction = -1.0 if dq >= 0 else 1.0
        
        # 总阻力
        total_resistance = (base_resistance + velocity_resistance + position_resistance) * resistance_direction
        
        # 记录阻力扭矩
        self.resistance_torques.append(abs(total_resistance))
        if len(self.resistance_torques) > 100:
            self.resistance_torques.pop(0)
            
        return total_resistance
    
    def compute_control_torque(self, q, dq, q_d, dq_d):
        """
        基于OIAC计算总控制扭矩
        """
        # 更新阻抗参数
        K_val, B_val = self.update_impedance(q, q_d, dq, dq_d)
        
        # 计算反馈扭矩 (OIAC基础)
        e = q_d - q
        de = dq_d - dq
        
        feedback_torque = K_val * e + B_val * de
        
        # 计算RAN阻力扭矩
        ran_torque = self.compute_ran_resistance(q, dq, q_d)
        
        # 总扭矩
        total_torque = feedback_torque + ran_torque
        
        # 记录跟踪误差
        self.tracking_errors.append(abs(e))
        if len(self.tracking_errors) > 100:
            self.tracking_errors.pop(0)
        
        return total_torque
    
    def set_ran_parameters(self, resistance_level=None, adaptive_gain=None):
        """设置RAN参数"""
        if resistance_level is not None:
            self.resistance_level = resistance_level
        if adaptive_gain is not None:
            self.adaptive_gain = adaptive_gain
            
        print(f"🎯 RAN-OIAC Parameters Updated:")
        print(f"   Resistance level: {self.resistance_level:.1f} Nm")
        print(f"   Adaptive gain: {self.adaptive_gain:.1f}")
        print(f"   Current K: {self.K[0,0]:.1f}, B: {self.B[0,0]:.1f}")
    
    def get_performance_metrics(self):
        """获取性能指标"""
        if not self.tracking_errors:
            return 0.0, 0.0, 0.0, 0.0
            
        avg_error = np.mean(self.tracking_errors)
        avg_K = np.mean([k for k, b in self.impedance_history]) if self.impedance_history else self.K[0,0]
        avg_B = np.mean([b for k, b in self.impedance_history]) if self.impedance_history else self.B[0,0]
        avg_resistance = np.mean(self.resistance_torques) if self.resistance_torques else 0.0
        
        return math.degrees(avg_error), avg_K, avg_B, avg_resistance
    
    def enable_ran(self, enable=True):
        """启用/禁用RAN模式"""
        self.ran_enabled = enable
        status = "ENABLED" if enable else "DISABLED"
        print(f"🔧 RAN Mode {status}")


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

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 OIAC-based RAN Control System")
    print("   (Output Impedance Adaptive Control)")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print(f"Angle range: [{ANGLE_MIN}, {ANGLE_MAX}] degrees")
    print("\n🔬 OIAC-RAN Features:")
    print("   - Adaptive impedance based on tracking performance")
    print("   - RAN resistance integrated with OIAC framework")
    print("   - Real-time impedance adjustment")
    print("   - Comprehensive performance monitoring")
    print("=" * 60)
    
    # 创建队列
    raw_data = queue.Queue(maxsize=SAMPLE_RATE)
    command_queue = queue.Queue(maxsize=10)
    motor_state = {'position': 0, 'velocity': 0}
    
    # 初始化EMG传感器
    emg = DelsysEMG()
    
    # 初始化滤波器和解释器
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, 
                     user_name=USER_NAME, BicepEMG=True, TricepEMG=False)
    
    interpreter.set_Kp(8)
    
    # 初始化电机
    motor = Motors()
    motor.set_cont_mode(mode='cur')
    
    # 🔥 初始化OIAC-RAN控制器
    ran_oiac = RAN_OIAC_Controller(dof=1)
    
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
    print("\n✅ EMG and motor threads started!")
    
    # 用户选择参数
    print("\n🎯 OIAC-RAN Configuration:")
    print("1. Light RAN (K=60, B=15, R=1.0)")
    print("2. Medium RAN (K=80, B=20, R=2.0)")  
    print("3. Heavy RAN (K=100, B=25, R=3.0)")
    print("4. Custom parameters")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        ran_oiac.K[0,0] = 60.0
        ran_oiac.B[0,0] = 15.0
        ran_oiac.set_ran_parameters(resistance_level=1.0)
    elif choice == "2":
        ran_oiac.K[0,0] = 80.0
        ran_oiac.B[0,0] = 20.0
        ran_oiac.set_ran_parameters(resistance_level=2.0)
    elif choice == "3":
        ran_oiac.K[0,0] = 100.0
        ran_oiac.B[0,0] = 25.0
        ran_oiac.set_ran_parameters(resistance_level=3.0)
    elif choice == "4":
        K = float(input("Enter initial stiffness K: "))
        B = float(input("Enter initial damping B: "))
        R = float(input("Enter resistance level: "))
        ran_oiac.K[0,0] = K
        ran_oiac.B[0,0] = B
        ran_oiac.set_ran_parameters(resistance_level=R)
    else:
        print("Using default medium parameters")
        ran_oiac.set_ran_parameters(resistance_level=2.0)
    
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    
    # 数据记录
    time_log = []
    desired_angle_log = []
    current_angle_log = []
    torque_log = []
    K_log = []
    B_log = []
    velocity_log = []
    
    # 状态变量
    current_angle = math.radians(55.0)
    current_velocity = 0.0
    last_time = time.time()
    start_time = time.time()
    last_desired_angle = math.radians(55.0)
    
    # 统计变量
    control_count = 0
    last_debug_time = time.time()
    last_oiac_update = time.time()
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting OIAC-RAN Control")
    print(f"{'='*60}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while not stop_event.is_set():
            # 获取EMG数据
            try:
                reading = raw_data.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            elapsed_time = current_time - start_time
            
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
            
            # 估计当前角速度
            #current_velocity = (desired_angle_rad - current_angle) / dt if dt > 0 else 0.0
            #current_angle += current_velocity * dt
            current_velocity = motor_state['velocity']
            current_angle_deg = (motor_center - motor_state['position']) / step
            current_angle = math.radians(current_angle_deg)
            
            # ========== 🎯 OIAC-RAN控制 ==========
            control_torque = ran_oiac.compute_control_torque(
                current_angle, current_velocity,
                desired_angle_rad, desired_velocity_rad
            )
            
            # 扭矩限制
            torque_clipped = np.clip(control_torque, TORQUE_MIN, TORQUE_MAX)
            
            # 记录数据
            time_log.append(elapsed_time)
            desired_angle_log.append(desired_angle_rad)
            current_angle_log.append(current_angle)
            torque_log.append(torque_clipped)
            K_log.append(ran_oiac.K[0,0])
            B_log.append(ran_oiac.B[0,0])
            velocity_log.append(current_velocity)
            
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
            
            # ===== OIAC性能监控输出 =====
            control_count += 1
            
            if current_time - last_debug_time > 2.0:
                avg_error, avg_K, avg_B, avg_resistance = ran_oiac.get_performance_metrics()
                
                print(f"⏱️ t={elapsed_time:.1f}s | "
                      f"Desired={desired_angle_deg:.1f}° | "
                      f"Current={math.degrees(current_angle):.1f}°")
                print(f"🔧 OIAC: K={ran_oiac.K[0,0]:.1f} | B={ran_oiac.B[0,0]:.1f} | "
                      f"Torque={torque_clipped:.2f}Nm")
                print(f"📊 Avg: Error={avg_error:.2f}° | K={avg_K:.1f} | B={avg_B:.1f} | "
                      f"Resistance={avg_resistance:.2f}Nm")
                last_debug_time = current_time
            
            # OIAC参数自适应显示
            if current_time - last_oiac_update > 4.0 and len(K_log) > 10:
                K_change = K_log[-1] - K_log[-10]
                B_change = B_log[-1] - B_log[-10]
                
                if abs(K_change) > 5 or abs(B_change) > 2:
                    print(f"🔄 OIAC Adaptation: ΔK={K_change:+.1f}, ΔB={B_change:+.1f}")
                last_oiac_update = current_time
            
            last_time = current_time
    
    except KeyboardInterrupt:
        print(f"\n🛑 OIAC-RAN Control stopped by user")
    
    # 最终性能分析
    print("\n" + "="*60)
    print("📊 OIAC-RAN Performance Analysis")
    print("="*60)
    
    if len(time_log) > 0:
        total_duration = time_log[-1] if time_log else 0
        motion_range = math.degrees(max(current_angle_log) - min(current_angle_log)) if current_angle_log else 0
        
        # OIAC性能统计
        final_K = K_log[-1] if K_log else ran_oiac.K[0,0]
        final_B = B_log[-1] if B_log else ran_oiac.B[0,0]
        avg_error, avg_K, avg_B, avg_resistance = ran_oiac.get_performance_metrics()
        
        # 阻抗变化分析
        K_variation = max(K_log) - min(K_log) if K_log else 0
        B_variation = max(B_log) - min(B_log) if B_log else 0
        
        print(f"Session Duration: {total_duration:.1f} seconds")
        print(f"Motion Range: {motion_range:.1f} degrees")
        print(f"\n🔬 OIAC Performance:")
        print(f"   Final Impedance: K={final_K:.1f}, B={final_B:.1f}")
        print(f"   Average Impedance: K={avg_K:.1f}, B={avg_B:.1f}")
        print(f"   Impedance Variation: ΔK={K_variation:.1f}, ΔB={B_variation:.1f}")
        print(f"   Average Tracking Error: {avg_error:.2f}°")
        print(f"   Average Resistance: {avg_resistance:.2f} Nm")
        print(f"   Control Cycles: {control_count}")
        
        # OIAC自适应效果评估
        if K_variation > 10 or B_variation > 5:
            print(f"\n✅ Good OIAC Adaptation: Impedance parameters actively adjusted")
        else:
            print(f"\n💡 OIAC Suggestion: Consider increasing adaptation gains")
    
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
    
    print("\n🎯 OIAC-RAN Control Complete!")
    print(" Key OIAC Features Implemented:")
    print("  ✓ Adaptive impedance control based on tracking performance")
    print("  ✓ Integrated RAN resistance within OIAC framework")
    print("  ✓ Real-time impedance parameter adaptation")
    print("  ✓ Comprehensive OIAC performance monitoring")
    print("\nOIAC-RAN system shutdown successfully! 🔧")