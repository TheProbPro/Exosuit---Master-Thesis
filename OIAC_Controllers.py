import numpy as np
import numpy.linalg as la
import math
from scipy import interpolate
import time

class ControlMode:
    """控制模式定义"""
    AAN = "assist_as_needed"  # 辅助模式
    RAN = "resist_as_needed"  # 阻力模式

class ada_imp_con():
    def __init__(self, dof):
        self.DOF = dof # degree of freedom

        self.k_mat = np.asmatrix(np.zeros((self.DOF, self.DOF))) # stiffness matrix
        self.b_mat = np.asmatrix(np.zeros((self.DOF, self.DOF))) # damping matrix

        self.ff_tau_mat = np.asmatrix(np.zeros((self.DOF, 1))) # feedforward torque matrix
        self.fb_tau_mat = np.asmatrix(np.zeros((self.DOF, 1))) # feedback torque matrix

        self.q = np.asmatrix(np.zeros((self.DOF, 1))) # position matrix
        self.q_d = np.asmatrix(np.zeros((self.DOF, 1))) # velocity matrix
        self.dq = np.asmatrix(np.zeros((self.DOF, 1))) # acceleration matrix
        self.dq_d = np.asmatrix(np.zeros((self.DOF, 1))) # desired velocity matrix

        # self.a = 1000.0#0.2 #0.0001
        # self.b = 0.00001#5.0#10.0
        # self.k = 0.00001 #0.0005 #0.05
        self.a = 0.6#0.2 #0.0001
        self.b = 0.001#5.0#10.0
        self.k = 0.001 #0.000001
        

    def update_impedance(self, q, q_d, dq, dq_d):
        # copy inputs
        self.q = np.asmatrix(np.copy(q))
        self.q_d = np.asmatrix(np.copy(q_d))
        self.dq = np.asmatrix(np.copy(dq))
        self.dq_d = np.asmatrix(np.copy(dq_d))
        #Update matrices
        self.k_mat = (self.gen_track_error() * self.gen_pos_error().T)/self.gen_ad_factor()
        self.b_mat = (self.gen_track_error() * self.gen_vel_error().T)/self.gen_ad_factor()
        return self.k_mat, self.b_mat
    
    def gen_pos_error(self):
        return self.q - self.q_d
    
    def gen_vel_error(self):
        return self.dq - self.dq_d
    
    def gen_track_error(self):
        return (self.k * self.gen_vel_error() + self.gen_pos_error())
    
    def gen_ad_factor(self):
        return self.a/(1.0 + self.b * la.norm(self.gen_track_error()) * la.norm(self.gen_track_error()))
    
    def calc_tau_fb(self):
        # self.fb_tau_mat = - self.k_mat * self.gen_pos_error() - self.b_mat * self.gen_vel_error()
        self.fb_tau_mat = self.k_mat * self.gen_pos_error() + self.b_mat * self.gen_vel_error()
        return self.fb_tau_mat
    
    def calc_tau_ff(self):
        self.ff_tau_mat = self.gen_ad_factor() * self.gen_track_error()
        return self.ff_tau_mat
    
    def get_tau(self):
        return self.calc_tau_fb() + self.calc_tau_ff()


class ILCv1():
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
        self.ILC_TRIAL_DURATION = 10.0  # seconds per trial
        
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
            max_time = max(time_array) if len(time_array) > 0 else self.ILC_TRIAL_DURATION
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
    
class ILCv2():
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
    
    # def save_learning(self, filepath):
    #     """保存学习数据"""
    #     data = {
    #         'learned_feedforward': self.learned_feedforward,
    #         'reference_time': self.reference_time,
    #         'trial_errors': self.trial_errors,
    #         'current_trial': self.current_trial
    #     }
    #     try:
    #         with open(filepath, 'wb') as f:
    #             pickle.dump(data, f)
    #         print(f"[ILC] Learning data saved to {filepath}")
    #     except Exception as e:
    #         print(f"[ILC] Failed to save: {e}")
    
    # def load_learning(self, filepath):
    #     """加载学习数据"""
    #     if not os.path.exists(filepath):
    #         print(f"[ILC] No saved data found at {filepath}")
    #         return False
            
    #     try:
    #         with open(filepath, 'rb') as f:
    #             data = pickle.load(f)
    #         self.learned_feedforward = data['learned_feedforward']
    #         self.reference_time = data['reference_time']
    #         self.trial_errors = data.get('trial_errors', [])
    #         self.current_trial = data['current_trial']
    #         print(f"[ILC] Loaded {self.current_trial} trials from {filepath}")
    #         return True
    #     except Exception as e:
    #         print(f"[ILC] Failed to load: {e}")
    #         return False
    
    def reset(self):
        """重置ILC"""
        self.learned_feedforward.clear()
        self.trial_errors.clear()
        self.trial_torques.clear()
        self.current_trial = 0
        print("[ILC] Reset completed")

class ModeControllerThreshold():
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

class ModeControllerUpDown():
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