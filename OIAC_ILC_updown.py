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
        self.DOF = dof  # degree of freedom

        self.k_mat = np.asmatrix(np.zeros((self.DOF, self.DOF)))  # stiffness matrix
        self.b_mat = np.asmatrix(np.zeros((self.DOF, self.DOF)))  # damping matrix

        self.ff_tau_mat = np.asmatrix(np.zeros((self.DOF, 1)))  # feedforward torque matrix
        self.fb_tau_mat = np.asmatrix(np.zeros((self.DOF, 1)))  # feedback torque matrix

        self.q = np.asmatrix(np.zeros((self.DOF, 1)))  # position matrix
        self.q_d = np.asmatrix(np.zeros((self.DOF, 1)))  # velocity matrix
        self.dq = np.asmatrix(np.zeros((self.DOF, 1)))  # acceleration matrix
        self.dq_d = np.asmatrix(np.zeros((self.DOF, 1)))  # desired velocity matrix

        self.a = 10  # 0.2 #0.0001
        self.b = 0.0001  # 5.0#10.0
        self.k = 0.5  # 0.0005 #0.05
        # self.a = 0.2#0.2 #0.0001
        # self.b = 5.0#5.0#10.0
        # self.k = 0.005 #0.000001

    def update_impedance(self, q, q_d, dq, dq_d):
        # copy inputs
        self.q = np.asmatrix(np.copy(q))
        self.q_d = np.asmatrix(np.copy(q_d))
        self.dq = np.asmatrix(np.copy(dq))
        self.dq_d = np.asmatrix(np.copy(dq_d))
        # Update matrices
        self.k_mat = (self.gen_track_error() * self.gen_pos_error().T) / self.gen_ad_factor()
        self.b_mat = (self.gen_track_error() * self.gen_vel_error().T) / self.gen_ad_factor()
        return self.k_mat, self.b_mat

    def gen_pos_error(self):
        return self.q - self.q_d

    def gen_vel_error(self):
        return self.dq - self.dq_d

    def gen_track_error(self):
        return (self.k * self.gen_vel_error() + self.gen_pos_error())

    def gen_ad_factor(self):
        return self.a / (1.0 + self.b * la.norm(self.gen_track_error()) * la.norm(self.gen_track_error()))

    def calc_tau_fb(self):
        # self.fb_tau_mat = - self.k_mat * self.gen_pos_error() - self.b_mat * self.gen_vel_error()
        self.fb_tau_mat = self.k_mat * self.gen_pos_error() + self.b_mat * self.gen_vel_error()
        return self.fb_tau_mat

    def calc_tau_ff(self):
        self.ff_tau_mat = self.gen_ad_factor() * self.gen_track_error()
        return self.ff_tau_mat

    def get_tau(self):
        return self.calc_tau_fb() + self.calc_tau_ff()

    def reset(self):
        """重置控制器状态"""
        self.k_mat = np.asmatrix(np.zeros((self.DOF, self.DOF)))
        self.b_mat = np.asmatrix(np.zeros((self.DOF, self.DOF)))
        self.ff_tau_mat = np.asmatrix(np.zeros((self.DOF, 1)))
        self.fb_tau_mat = np.asmatrix(np.zeros((self.DOF, 1)))
        self.q = np.asmatrix(np.zeros((self.DOF, 1)))
        self.q_d = np.asmatrix(np.zeros((self.DOF, 1)))
        self.dq = np.asmatrix(np.zeros((self.DOF, 1)))
        self.dq_d = np.asmatrix(np.zeros((self.DOF, 1)))


class ILC():
    def __init__(self, max_trials=10, trial_duration=10.0, frequency=20.9, lr=0.1):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []
        self.trial_duration = trial_duration
        self.frequency = frequency
        self.error_history = []
        self.lr = lr  # learning rate
        self.ff_iterator = 0

    def update_learning(self, error_array):
        # reset iterator for feedforward retrieval
        self.ff_iterator = 0

        if len(error_array) == 0:
            print("[ILC] Warning: Empty error array, skipping update")
            return np.zeros(int(self.trial_duration * self.frequency))

        # Convert error_array to numpy array if it isn't already
        error_array = np.array(error_array)

        # Ensure error array is the same length as previous trials
        expected_len = int(self.trial_duration * self.frequency)
        assert len(error_array) == expected_len

        # Update learning
        if not self.learned_feedforward:
            ff = np.zeros_like(error_array)
        else:
            ff = self.learned_feedforward[-1] + self.lr * error_array

        # save learned feedforward and error history and iterate trial count
        self.learned_feedforward.append(ff)
        self.error_history.append(error_array)
        self.current_trial += 1

        # Performance metrics
        avg_error = np.mean(np.abs(error_array))
        max_error = np.max(np.abs(error_array))
        print(f"[ILC] Trial {self.current_trial} completed:")
        print(f"      Learning rate: {self.lr}")
        print(f"      Avg error: {math.degrees(avg_error)}°")
        print(f"      Max error: {math.degrees(max_error)}°")
        return ff

    def get_feedforward(self, trial_idx=-1):
        # retrieve feedforward value at current iterator position
        if not self.learned_feedforward:
            return 0.0
        ff_value = self.learned_feedforward[trial_idx][self.ff_iterator]
        # increment iterator
        self.ff_iterator += 1
        return ff_value


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

    def compute_control(self, t, current_pos, current_vel, desired_pos, desired_vel, trial_idx=0):
        """
        计算控制扭矩，基于扭矩符号实现AAN/RAN切换
        """
        current_time = t

        # 使用自适应阻抗控制器计算基础扭矩
        # 更新阻抗参数
        q = np.array([[current_pos]])
        q_d = np.array([[desired_pos]])
        dq = np.array([[current_vel]])
        dq_d = np.array([[desired_vel]])
        
        k_mat, b_mat = self.adaptive_controller.update_impedance(q, q_d, dq, dq_d)
        base_torque = float(self.adaptive_controller.get_tau()[0, 0])
        
        k = float(k_mat[0, 0])
        d = float(b_mat[0, 0])
        ff = float(self.adaptive_controller.calc_tau_ff()[0, 0])

        # 获取ILC前馈扭矩 (如果有)
        ilc_torque = 0.0
        if self.ilc_controller and trial_idx > 0:
            ilc_torque = self.ilc_controller.get_feedforward(trial_idx - 1)

        # 计算AAN模式的总扭矩 (基础扭矩 + ILC前馈)
        aan_torque = base_torque + ilc_torque

        # ===== 扭矩符号的模式切换 =====
        can_switch = (current_time - self.last_switch_time) >= self.min_switch_interval

        if aan_torque < 0:  # 负扭矩 → AAN模式 (辅助)
            if self.current_mode != 'AAN' and can_switch:
                self.current_mode = 'AAN'
                self.last_switch_time = current_time
                print(f" RAN→AAN at t={t:.2f}s (torque={aan_torque:.2f}Nm) - Activating assistance")

            total_torque = aan_torque

        else:  # 正扭矩或零 → RAN模式 (阻力)
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

        current_time = self.mode_history[-1][0] if self.mode_history else 0
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


# # ===== 使用示例 =====
# if __name__ == "__main__":
#     # 初始化控制器
#     dof = 1  # 单自由度
#     adaptive_controller = ada_imp_con(dof)
#     ilc_controller = ILC(max_trials=10, trial_duration=10.0, frequency=20.0, lr=0.1)
#     mode_controller = ModeControllerUpDown(adaptive_controller, ilc_controller)

#     print("控制器初始化完成")
#     print(f"初始模式: {mode_controller.current_mode}")
    
#     # 模拟控制循环示例
#     dt = 0.05  # 时间步长
#     t = 0.0
#     trial_idx = 0
    
#     # 模拟运动
#     for i in range(100):
#         # 示例输入
#         current_pos = 0.1 * np.sin(2 * np.pi * 0.5 * t)
#         current_vel = 0.1 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
#         desired_pos = 0.0
#         desired_vel = 0.0
        
#         # 计算控制扭矩
#         torque, mode, k, d, ff = mode_controller.compute_control(
#             t, current_pos, current_vel, desired_pos, desired_vel, trial_idx
#         )
        
#         if i % 20 == 0:  # 每秒打印一次
#             print(f"t={t:.2f}s: 模式={mode}, 扭矩={torque:.3f}Nm, K={k:.3f}, D={d:.3f}")
        
#         t += dt
    
#     # 获取模式统计
#     aan_ratio, ran_ratio = mode_controller.get_mode_statistics(recent_seconds=5)
#     print(f"\n最近5秒模式统计: AAN={aan_ratio:.1f}%, RAN={ran_ratio:.1f}%")