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

class ContinuousRAN_OIAC_Controller:
    """
    持续RAN模式控制器
    在整个运动过程中都提供阻力，不依赖跟踪性能
    """
    def __init__(self, dof=1):
        self.DOF = dof
        
        # 固定阻抗参数 - 提供稳定性
        self.K = np.eye(dof) * 70.0  # 刚度矩阵
        self.B = np.eye(dof) * 160.0  # 阻尼矩阵
        
        # RAN阻力参数 - 全程保持
        self.base_resistance = 2.5  # 基础阻力水平
        self.velocity_factor = 1.5  # 速度阻力系数
        self.position_factor = 0.6  # 位置阻力系数
        
        # 状态变量
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))
        
        # 性能监控
        self.resistance_torques = []
        self.total_torques = []
        
    def compute_continuous_resistance(self, q, dq, q_d):
        """
        计算持续阻力 - 始终存在，不依赖跟踪性能
        """
        # 1. 基础阻力 (始终抵抗运动)
        base_resistance = self.base_resistance
        
        # 确定阻力方向 (总是抵抗当前运动方向)
        if dq > 0:
            resistance_direction = -1.0  # 抵抗正向运动
        elif dq < 0:
            resistance_direction = 1.0   # 抵抗负向运动
        else:
            # 静止时，抵抗可能发生的运动方向
            position_error = q - q_d
            resistance_direction = -1.0 if position_error > 0 else 1.0
        
        # 2. 速度相关阻力 (速度越快，阻力越大)
        velocity_resistance = self.velocity_factor * abs(dq)
        
        # 3. 位置相关阻力 (离中心越远，阻力越大)
        center_position = math.radians(70)  # 中心位置
        position_error = abs(q - center_position)
        position_resistance = self.position_factor * position_error
        
        # 总阻力 (始终存在)
        total_resistance = (base_resistance + velocity_resistance + position_resistance) * resistance_direction
        
        # 记录阻力
        self.resistance_torques.append(abs(total_resistance))
        if len(self.resistance_torques) > 100:
            self.resistance_torques.pop(0)
            
        return total_resistance
    
    def compute_control_torque(self, q, dq, q_d, dq_d):
        """
        计算总控制扭矩 - 始终包含RAN阻力
        """
        # 更新状态
        self.q = np.array([[q]])
        self.q_d = np.array([[q_d]])
        self.dq = np.array([[dq]])
        self.dq_d = np.array([[dq_d]])
        
        # 计算反馈扭矩 (提供稳定性)
        e = q_d - q
        de = dq_d - dq
        
        feedback_torque = self.K[0,0] * e + self.B[0,0] * de
        
        # 计算持续RAN阻力 (始终存在)
        ran_torque = self.compute_continuous_resistance(q, dq, q_d)
        
        # 总扭矩 = 反馈 + 持续阻力
        total_torque = feedback_torque + ran_torque
        
        # 记录总扭矩
        self.total_torques.append(abs(total_torque))
        if len(self.total_torques) > 100:
            self.total_torques.pop(0)
        
        return total_torque
    
    def set_resistance_parameters(self, base_resistance=None, velocity_factor=None, position_factor=None):
        """设置阻力参数"""
        if base_resistance is not None:
            self.base_resistance = base_resistance
        if velocity_factor is not None:
            self.velocity_factor = velocity_factor
        if position_factor is not None:
            self.position_factor = position_factor
            
        print(f"🎯 Continuous RAN Parameters Updated:")
        print(f"   Base resistance: {self.base_resistance:.1f} Nm")
        print(f"   Velocity factor: {self.velocity_factor:.1f}")
        print(f"   Position factor: {self.position_factor:.1f}")
        print(f"   Stiffness K: {self.K[0,0]:.1f}, Damping B: {self.B[0,0]:.1f}")
    
    def get_resistance_statistics(self):
        """获取阻力统计"""
        if not self.resistance_torques:
            return 0.0, 0.0, 0.0
            
        avg_resistance = np.mean(self.resistance_torques)
        max_resistance = np.max(self.resistance_torques)
        current_resistance = self.resistance_torques[-1] if self.resistance_torques else 0.0
        
        return avg_resistance, max_resistance, current_resistance
    
    def get_impedance_parameters(self):
        """获取阻抗参数"""
        return self.K[0,0], self.B[0,0]


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


if __name__ == "__main__":
    print("=" * 60)
    print("💪 Continuous RAN Control System")
    print("   (Resistance Throughout Entire Movement)")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    print(f"Angle range: [{ANGLE_MIN}, {ANGLE_MAX}] degrees")
    print("\n🎯 Continuous RAN Features:")
    print("   - Resistance throughout entire movement")
    print("   - Base resistance + velocity-dependent resistance")
    print("   - Position-dependent resistance")
    print("   - Always opposes movement direction")
    print("   - Stable impedance control foundation")
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
    
    # 🔥 初始化持续RAN控制器
    ran_controller = ContinuousRAN_OIAC_Controller(dof=1)
    
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
    t_motor = threading.Thread(target=send_motor_command, args=(motor, command_queue, motor_state), daemon=True)
    t_emg.start()
    t_motor.start()
    print("\n✅ EMG and motor threads started!")
    
    # 用户选择阻力水平
    print("\n🎯 Select Continuous Resistance Level:")
    print("1. Light Resistance (1.5 Nm base)")
    print("2. Medium Resistance (2.5 Nm base)")  
    print("3. Heavy Resistance (3.5 Nm base)")
    print("4. Very Heavy Resistance (5.0 Nm base)")
    print("5. Custom Resistance")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        ran_controller.set_resistance_parameters(1.5, 1.0, 0.2)
    elif choice == "2":
        ran_controller.set_resistance_parameters(2.5, 1.5, 0.3)
    elif choice == "3":
        ran_controller.set_resistance_parameters(3.5, 2.0, 0.4)
    elif choice == "4":
        ran_controller.set_resistance_parameters(5.0, 2.5, 0.5)
    elif choice == "5":
        base = float(input("Enter base resistance (Nm): "))
        vel_factor = float(input("Enter velocity factor: "))
        pos_factor = float(input("Enter position factor: "))
        ran_controller.set_resistance_parameters(base, vel_factor, pos_factor)
    else:
        print("Using default medium resistance")
        ran_controller.set_resistance_parameters(2.5, 1.5, 0.3)
    
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    
    # 数据记录
    time_log = []
    desired_angle_log = []
    current_angle_log = []
    resistance_log = []
    velocity_log = []
    total_torque_log = []
    
    # 状态变量
    current_angle = math.radians(55.0)
    current_velocity = 0.0
    last_time = time.time()
    start_time = time.time()
    last_desired_angle = math.radians(55.0)
    
    # 统计变量
    control_count = 0
    last_debug_time = time.time()
    last_stats_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Continuous RAN Control")
    print(f"{'='*60}")
    print("💡 Resistance will be applied throughout the entire movement")
    print("   The system will always oppose your motion direction")
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
            
            # 估计当前角速度和角度
            #current_velocity = (desired_angle_rad - current_angle) / dt if dt > 0 else 0.0
            #current_angle += current_velocity * dt
            current_velocity = motor_state['velocity']
            current_angle_deg = (motor_center - motor_state['position']) / step
            current_angle = math.radians(current_angle_deg)
            
            # ========== 🔥 持续RAN控制 ==========
            total_torque = ran_controller.compute_control_torque(
                current_angle, current_velocity,
                desired_angle_rad, desired_velocity_rad
            )
            
            # 扭矩限制
            torque_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)
            
            # 记录数据
            time_log.append(elapsed_time)
            desired_angle_log.append(desired_angle_rad)
            current_angle_log.append(current_angle)
            resistance_log.append(ran_controller.resistance_torques[-1] if ran_controller.resistance_torques else 0)
            velocity_log.append(current_velocity)
            total_torque_log.append(torque_clipped)
            
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
            
            # ===== 实时反馈 =====
            control_count += 1
            
            if current_time - last_debug_time > 1.5:
                avg_resistance, max_resistance, current_resistance = ran_controller.get_resistance_statistics()
                K_val, B_val = ran_controller.get_impedance_parameters()
                
                # 显示运动方向
                motion_direction = "EXTENSION" if current_velocity > 0 else "FLEXION" if current_velocity < 0 else "HOLDING"
                
                print(f"⏱️ t={elapsed_time:.1f}s | {motion_direction}")
                print(f"🎯 Desired={desired_angle_deg:.1f}° | Current={math.degrees(current_angle):.1f}°")
                print(f"💪 Resistance: {current_resistance:.2f}Nm (Avg: {avg_resistance:.2f}Nm)")
                print(f"🔧 Total Torque: {torque_clipped:.2f}Nm | Vel: {math.degrees(abs(current_velocity)):.1f}°/s")
                last_debug_time = current_time
            
            # 每5秒显示统计信息
            if current_time - last_stats_time > 5.0:
                if len(current_angle_log) > 0:
                    motion_range = math.degrees(max(current_angle_log) - min(current_angle_log))
                    avg_velocity = np.mean(np.abs(velocity_log[-100:])) if len(velocity_log) > 100 else 0.0
                    avg_resistance = np.mean(resistance_log[-100:]) if len(resistance_log) > 100 else 0.0
                    
                    print(f"\n📊 5s Statistics:")
                    print(f"   Motion range: {motion_range:.1f}°")
                    print(f"   Avg velocity: {math.degrees(avg_velocity):.1f}°/s")
                    print(f"   Avg resistance: {avg_resistance:.2f}Nm")
                    print(f"   Control cycles: {control_count}")
                last_stats_time = current_time
            
            last_time = current_time
    
    except KeyboardInterrupt:
        print(f"\n🛑 Continuous RAN Control stopped by user")
    
    # 最终统计
    print("\n" + "="*60)
    print("📊 Continuous RAN Session Statistics")
    print("="*60)
    
    if len(time_log) > 0:
        total_duration = time_log[-1] if time_log else 0
        motion_range = math.degrees(max(current_angle_log) - min(current_angle_log)) if current_angle_log else 0
        avg_resistance, max_resistance, _ = ran_controller.get_resistance_statistics()
        avg_velocity = np.mean(np.abs(velocity_log)) if velocity_log else 0
        avg_torque = np.mean(np.abs(total_torque_log)) if total_torque_log else 0
        
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Motion range: {motion_range:.1f} degrees")
        print(f"Average resistance: {avg_resistance:.2f} Nm")
        print(f"Maximum resistance: {max_resistance:.2f} Nm") 
        print(f"Average velocity: {math.degrees(avg_velocity):.1f} °/s")
        print(f"Average torque: {avg_torque:.2f} Nm")
        print(f"Total control cycles: {control_count}")
        
        # 训练效果评估
        resistance_intensity = ""
        if avg_resistance > 4.0:
            resistance_intensity = "💪 HIGH INTENSITY"
        elif avg_resistance > 2.0:
            resistance_intensity = "🔥 MEDIUM INTENSITY"
        else:
            resistance_intensity = "🌱 LIGHT INTENSITY"
            
        print(f"\n{resistance_intensity} Workout")
        
        if motion_range > 40.0 and avg_velocity > 8.0:
            print(f"🎉 Excellent! Full range with good speed.")
        elif motion_range > 25.0:
            print(f"👍 Good effort! Moderate range achieved.")
        else:
            print(f"💡 Suggestion: Try to increase your movement range for better results.")
    
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
    
    print("\n💪 Continuous RAN Training Complete!")
    print(" Key Features:")
    print("  ✓ Resistance throughout entire movement")
    print("  ✓ Always opposes motion direction") 
    print("  ✓ Velocity-dependent resistance")
    print("  ✓ Position-dependent resistance")
    print("  ✓ Real-time resistance monitoring")
    print("\nGreat workout! 🏋️♂️")