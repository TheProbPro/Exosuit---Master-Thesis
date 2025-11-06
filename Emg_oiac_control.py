# My local imports (EMG sensor, filtering, interpretors, OIAC)
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Motors.DynamixelHardwareInterface import Motors

# General imports
import numpy as np
import queue
import threading
import sys
import signal
import time
import math

SAMPLE_RATE = 2000  # Hz
USER_NAME = 'zichen'
ANGLE_MIN = 0
ANGLE_MAX = 140

# Control parameters
TORQUE_MIN = -4.1  # Nm
TORQUE_MAX = 4.1   # Nm

stop_event = threading.Event()

class EMGMuscleForceEstimator:
    """使用EMG信号估计肌肉力"""
    def __init__(self):
        self.bicep_force_history = []
        self.tricep_force_history = []
        self.force_penalty_history = []
        
        # EMG到力的转换系数（需要根据实际情况校准）
        self.emg_to_force_scale = 0.1  # 这个值需要实验调整
        
    def estimate_muscle_forces(self, bicep_rms, tricep_rms):
        """
        基于EMG RMS值估计肌肉力
        
        参数:
            bicep_rms: 肱二头肌EMG的RMS值
            tricep_rms: 肱三头肌EMG的RMS值
        
        返回:
            bicep_force, tricep_force: 估计的肌肉力 (N)
        """
        # 简单的线性模型：力 = EMG_RMS * 系数
        # 实际应用中可能需要更复杂的模型（考虑肌肉长度、速度等）
        bicep_force = bicep_rms * self.emg_to_force_scale
        tricep_force = tricep_rms * self.emg_to_force_scale
        
        # 确保力不为负
        bicep_force = max(0, bicep_force)
        tricep_force = max(0, tricep_force)
        
        return bicep_force, tricep_force
    
    def calculate_force_penalty(self, bicep_force, tricep_force, q_error, control_torque):
        """
        基于估计的肌肉力计算惩罚
        目标：最小化肌肉激活，同时保持轨迹跟踪
        """
        error_deg = abs(math.degrees(q_error))
        
        # 只有当跟踪误差较小时才应用肌肉力惩罚
        # 大误差时优先考虑跟踪精度
        if error_deg < 8.0:
            # 总肌肉激活力
            force_magnitude = bicep_force + tricep_force
            
            # 检查控制扭矩与肌肉力的一致性
            torque_force_alignment = 1.0
            
            # 如果控制扭矩要求弯曲（正扭矩），二头肌应该发力
            if control_torque > 0 and bicep_force > tricep_force:
                torque_force_alignment = 0.5  # 一致，减小惩罚
            # 如果控制扭矩要求伸展（负扭矩），三头肌应该发力
            elif control_torque < 0 and tricep_force > bicep_force:
                torque_force_alignment = 0.5  # 一致，减小惩罚
            else:
                torque_force_alignment = 2.0  # 不一致，增加惩罚
            
            # 惩罚系数可以调整
            force_penalty = 0.001 * force_magnitude * torque_force_alignment
        else:
            force_penalty = 0.0  # 大误差时专注于跟踪精度
        
        # 记录历史
        self.bicep_force_history.append(bicep_force)
        self.tricep_force_history.append(tricep_force)
        self.force_penalty_history.append(force_penalty)
        
        # 保持历史长度
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


class RobustAdaptiveImpedanceController:
    """鲁棒自适应阻抗控制器"""
    def __init__(self, dof=1):
        self.DOF = dof
        self.k_mat = np.zeros((self.DOF, self.DOF))
        self.b_mat = np.zeros((self.DOF, self.DOF))
        
        # 阻抗参数范围
        self.K_MIN = 30.0
        self.K_MAX = 100.0
        self.B_MIN = 10.0
        self.B_MAX = 30.0
        
        self.error_history = []
        self.control_history = []
        
        # 积分项用于消除稳态误差
        self.integral = 0.0
        self.ki = 5.0
        self.max_integral = 15.0
        
        # 上一次的状态（用于计算速度）
        self.last_q = None
        self.last_time = None

    def update_impedance(self, q, q_d, dq, dq_d, t):
        """
        自适应阻抗调整
        
        参数:
            q: 当前角度 (rad)
            q_d: 期望角度 (rad)
            dq: 当前角速度 (rad/s)
            dq_d: 期望角速度 (rad/s)
            t: 当前时间 (s)
        """
        error = q_d - q
        derror = dq_d - dq
        
        error_mag_deg = abs(math.degrees(error))
        
        # 记录历史
        self.error_history.append(error_mag_deg)
        if len(self.error_history) > 80:
            self.error_history.pop(0)
        
        avg_error = np.mean(self.error_history) if self.error_history else error_mag_deg
        
        # 积分项更新
        dt = 0.002  # 假设固定时间步长
        self.integral += error * dt
        
        # 只在小误差时积分，避免积分饱和
        if abs(error) < math.radians(2.0):
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        else:
            self.integral *= 0.9  # 大误差时抑制积分
        
        # 基于平均误差的阻抗调整策略
        if avg_error > 12.0:
            k_val = 80.0
            b_val = 20.0
        elif avg_error > 8.0:
            k_val = 60.0
            b_val = 16.0
        elif avg_error > 5.0:
            k_val = 45.0
            b_val = 13.0
        elif avg_error > 3.0:
            k_val = 35.0
            b_val = 11.0
        else:
            k_val = 30.0
            b_val = 9.0
        
        # 基于速度误差调整阻尼
        velocity_factor = min(abs(derror) * 2.0, 8.0)
        b_val += velocity_factor
        
        # 限制在范围内
        k_val = np.clip(k_val, self.K_MIN, self.K_MAX)
        b_val = np.clip(b_val, self.B_MIN, self.B_MAX)
        
        self.k_mat = np.array([[k_val]])
        self.b_mat = np.array([[b_val]])
        
        return self.k_mat, self.b_mat, self.integral


def read_EMG(EMG_sensor, queue):
    """EMG读取线程"""
    while not stop_event.is_set():
        reading = EMG_sensor.read()
        try:
            queue.put_nowait(reading)
        except queue.Full:
            try:
                queue.get_nowait()  # 丢弃最旧数据
                queue.put_nowait(reading)
            except queue.Full:
                pass
        except Exception as e:
            print(f"[reader] error: {e}", file=sys.stderr)


def send_motor_command(motor, command_queue):
    """电机命令发送线程"""
    while not stop_event.is_set():
        try:
            # command = (torque, position_fallback)
            command = command_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            # 如果电机支持扭矩控制，使用command[0]
            # 如果只支持位置控制，使用command[1]
            # 这里假设使用位置控制作为后备
            motor.sendMotorCommand(motor.motor_ids[0], command[1])
        except Exception as e:
            print(f"[motor send] error: {e}", file=sys.stderr)


def handle_sigint(sig, frame):
    """Ctrl-C处理"""
    print("\nShutdown signal received...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    print("=== EMG-based OIAC Control System ===")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}] Nm")
    
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
    oiac = RobustAdaptiveImpedanceController(dof=1)
    muscle_estimator = EMGMuscleForceEstimator()
    
    # 基础PD增益
    Kp_base = 30.0
    Kd_base = 10.0
    
    # 电机位置转换参数
    step = 1500.0 / 140.0  # 角度到电机位置的转换
    motor_center = 2550    # 电机中心位置
    
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
    print("EMG and motor threads started!")
    
    # 控制变量初始化
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    
    # 状态变量
    current_angle = math.radians(55.0)  # 初始角度估计
    current_velocity = 0.0
    last_time = time.time()
    last_desired_angle = math.radians(55.0)
    
    # 统计变量
    control_count = 0
    last_debug_time = time.time()
    last_force_debug_time = time.time()
    
    print("\n=== Starting OIAC Control Loop ===")
    print("Press Ctrl-C to stop\n")
    
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
            
            # 计算激活度
            activation = interpreter.compute_activation(filtered_bicep_RMS, filtered_tricep_RMS)
            
            # 计算期望角度（从激活度）
            desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])
            desired_angle_rad = math.radians(desired_angle_deg)
            
            # 估计期望角速度（简单差分）
            desired_velocity_rad = (desired_angle_rad - last_desired_angle) / dt if dt > 0 else 0.0
            last_desired_angle = desired_angle_rad
            
            # 估计当前角速度（从电机反馈或估计）
            # 注意：这里假设我们有角度反馈，如果没有需要用电机编码器
            # 临时方案：用控制输出估计
            current_velocity = (desired_angle_rad - current_angle) / dt if dt > 0 else 0.0
            
            # 更新当前角度估计（实际应用中应该从电机编码器读取）
            current_angle += current_velocity * dt
            
            # ===== OIAC控制 =====
            # 位置和速度误差
            position_error = desired_angle_rad - current_angle
            velocity_error = desired_velocity_rad - current_velocity
            
            # 更新阻抗参数
            K_mat, B_mat, integral = oiac.update_impedance(
                current_angle, desired_angle_rad,
                current_velocity, desired_velocity_rad,
                current_time
            )
            
            # 基础PD控制
            pd_torque = (Kp_base * position_error + 
                        Kd_base * velocity_error + 
                        oiac.ki * integral)
            
            # 阻抗反馈
            pos_error_vec = np.array([[position_error]])
            vel_error_vec = np.array([[velocity_error]])
            impedance_torque = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
            
            # 总扭矩（在应用肌肉力惩罚之前）
            total_torque = pd_torque + impedance_torque
            
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
            
            # 转换为电机位置命令（后备方案）
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
            
            # 每2秒打印一次控制状态
            if current_time - last_debug_time > 2.0:
                error_deg = math.degrees(position_error)
                print(f"t={current_time:.1f}s | "
                      f"Desired={desired_angle_deg:6.1f}° | "
                      f"Current={math.degrees(current_angle):6.1f}° | "
                      f"Error={error_deg:6.1f}° | "
                      f"Torque={torque_clipped:6.2f}Nm | "
                      f"K={K_mat[0,0]:5.1f} | "
                      f"B={B_mat[0,0]:5.1f}")
                last_debug_time = current_time
            
            # 每3秒打印一次肌肉力信息
            if current_time - last_force_debug_time > 3.0:
                print(f"Muscle Forces | "
                      f"Bicep: {bicep_force:6.2f}N | "
                      f"Tricep: {tricep_force:6.2f}N | "
                      f"Penalty: {force_penalty:6.4f}Nm")
                last_force_debug_time = current_time
            
            last_time = current_time
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    
    finally:
        # 停止系统
        print("\n=== Shutting down ===")
        stop_event.set()
        
        # 等待线程结束
        t_emg.join(timeout=2.0)
        t_motor.join(timeout=2.0)
        
        # 停止EMG和电机
        emg.stop()
        motor.close()
        
        # 清空队列
        raw_data.queue.clear()
        Bicep_RMS_queue.queue.clear()
        Tricep_RMS_queue.queue.clear()
        command_queue.queue.clear()
        
        # 打印最终统计
        print("\n=== Final Statistics ===")
        avg_bicep, avg_tricep, max_bicep, max_tricep = muscle_estimator.get_force_statistics()
        print(f"Muscle forces:")
        print(f"  Bicep - Avg: {avg_bicep:.2f}N, Max: {max_bicep:.2f}N")
        print(f"  Tricep - Avg: {avg_tricep:.2f}N, Max: {max_tricep:.2f}N")
        print(f"Total control cycles: {control_count}")
        
        print("\nGoodbye!")