import mujoco, mujoco.viewer
import time, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import interpolate

# ---------------------------
# Configuration and System Diagnostics
# ---------------------------
MODEL_PATH = "mergedCopy.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Detailed system diagnostics
print("=== System Detailed Diagnostics ===")
print(f"Model joint count: {model.njnt}")
print(f"Model degrees of freedom: {model.nv}")
print(f"Actuator count: {model.nu}")

# Find all joint information
for i in range(model.njnt):
    jnt_type = model.jnt_type[i]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i] if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE else -1
    print(f"Joint {i} ({jnt_name}): type={jnt_type}, qpos_addr={qpos_adr}, dof_addr={dof_adr}")

# Find elbow joint
joint_name = "el_x"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
if joint_id == -1:
    print(f"Error: Joint '{joint_name}' not found")
    print("Available joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"  {name}")
    exit()
    
qpos_adr = model.jnt_qposadr[joint_id]
dof_adr = model.jnt_dofadr[joint_id]

print(f"\nTarget joint '{joint_name}':")
print(f"  ID: {joint_id}, qpos address: {qpos_adr}, dof address: {dof_adr}")
print(f"  Joint range: {model.jnt_range[joint_id]} radians")
print(f"  Joint range: {np.degrees(model.jnt_range[joint_id])} degrees")

# Find actuators
print(f"\nActuator configuration:")
actuator_found = False
for i in range(model.nu):
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  Actuator {i} ({act_name}): controlled joint={jnt_name}, ctrlrange={model.actuator_ctrlrange[i]}, gear={model.actuator_gear[i]}")
    
    if jnt_name == joint_name:
        act_idx = i
        actuator_found = True
        print(f"  -> Found elbow joint actuator: index={i}")

if not actuator_found:
    print("Warning: No dedicated elbow joint actuator found, using first actuator")
    act_idx = 0

# ==================== Physical System Test ====================

def physical_system_test():
    """Test physical system response"""
    print("\n=== Physical System Response Test ===")
    
    data.qpos[qpos_adr] = math.radians(55.0)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    
    initial_pos = data.qpos[qpos_adr]
    print(f"Initial position: {math.degrees(initial_pos):.1f}°")
    
    print("\nGravity effect test:")
    for i in range(50):
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            pos_deg = math.degrees(data.qpos[qpos_adr])
            print(f"  Step {i}: position={pos_deg:6.1f}°")
    
    data.qpos[qpos_adr] = math.radians(55.0)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    
    print("\nTorque response test:")
    test_torques = [-20, -10, -5, 0, 5, 10, 20]
    for torque in test_torques:
        data.ctrl[act_idx] = torque
        positions = []
        for step in range(20):
            mujoco.mj_step(model, data)
            if step % 4 == 0:
                positions.append(math.degrees(data.qpos[qpos_adr]))
        
        print(f"  Torque={torque:3}Nm: position changes {positions}")
        
        data.qpos[qpos_adr] = math.radians(55.0)
        data.qvel[dof_adr] = 0.0
        mujoco.mj_forward(model, data)

physical_system_test()

# ==================== True RAN-Optimized OIAC Controller ====================

class TrueRANOptimizedOIAC:
    """
    True RAN-optimized controller - RAN mode provides resistance without position tracking
    """
    def __init__(self, dof):
        self.DOF = dof
        # Moderate initial impedance
        self.k_mat = np.eye(dof) * 60.0
        self.b_mat = np.eye(dof) * 15.0
        
        # State variables
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))
        
        # Conservative parameters for stability
        self.a = 0.1#0.05
        self.b = 0.005#0.01
        self.k = 0.2
        
        # Reasonable impedance ranges
        self.k_min = 20.0
        self.k_max = 200.0
        self.b_min = 8.0
        self.b_max = 80.0
        
    def gen_pos_err(self):
        return (self.q - self.q_d)
    
    def gen_vel_err(self):
        return (self.dq - self.dq_d)
    
    def gen_track_err(self):
        return (self.k * self.gen_vel_err() + self.gen_pos_err())
    
    def gen_ad_factor(self):
        track_err_norm = la.norm(self.gen_track_err())
        denominator = max(1.0 + self.b * track_err_norm * track_err_norm, 0.1)
        return self.a / denominator
    
    def update_impedance(self, q, q_d, dq, dq_d, mode='AAN'):
        self.q = np.atleast_2d(np.atleast_1d(q)).T
        self.q_d = np.atleast_2d(np.atleast_1d(q_d)).T
        self.dq = np.atleast_2d(np.atleast_1d(dq)).T
        self.dq_d = np.atleast_2d(np.atleast_1d(dq_d)).T
        
        track_err = self.gen_track_err()
        pos_err = self.gen_pos_err()
        vel_err = self.gen_vel_err()
        ad_factor = max(self.gen_ad_factor(), 0.001)
        
        # Moderate scaling for both modes
        k_scale = 150.0
        b_scale = 100.0
        
        k_update = k_scale * (track_err @ pos_err.T) / ad_factor
        b_update = b_scale * (track_err @ vel_err.T) / ad_factor
        
        # Smooth adaptation
        alpha = 0.9
        self.k_mat = alpha * self.k_mat + (1 - alpha) * np.clip(k_update, self.k_min, self.k_max)
        self.b_mat = alpha * self.b_mat + (1 - alpha) * np.clip(b_update, self.b_min, self.b_max)
        
        return self.k_mat, self.b_mat

# ==================== Enhanced ILC Controller ====================

class EnhancedILC:
    def __init__(self, max_trials=10, target_length=2000):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []
        self.reference_time = None
        self.learning_rates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03]
        
    def update_learning(self, time_array, error_array, torque_array):
        """Enhanced learning update"""
        if self.reference_time is None:
            max_time = min(10.0, max(time_array))
            self.reference_time = np.linspace(0, max_time, len(time_array))
        
        # Align data
        interp_error = interpolate.interp1d(time_array, error_array, bounds_error=False, fill_value=0.0)
        aligned_error = interp_error(self.reference_time)
        
        if not self.learned_feedforward:
            ff = np.zeros_like(aligned_error)
        else:
            lr = self.learning_rates[min(self.current_trial, len(self.learning_rates)-1)]
            ff = self.learned_feedforward[-1] + lr * aligned_error
        
        # Limit feedforward magnitude
        ff = np.clip(ff, -30.0, 30.0)
        
        # Smoothing
        if len(ff) > 10:
            ff = np.convolve(ff, np.ones(7)/7.0, mode='same')
        
        self.learned_feedforward.append(ff)
        self.current_trial += 1
        
        print(f"ILC: Trial {self.current_trial}, Learning rate={self.learning_rates[min(self.current_trial-1, len(self.learning_rates)-1)]:.2f}, Feedforward range[{np.min(ff):.1f}, {np.max(ff):.1f}]")
        
        return ff
    
    def get_feedforward(self, t, trial_idx):
        """Get feedforward torque"""
        if trial_idx < 0 or trial_idx >= len(self.learned_feedforward):
            return 0.0
            
        idx = np.argmin(np.abs(self.reference_time - t))
        if idx < len(self.learned_feedforward[trial_idx]):
            return float(self.learned_feedforward[trial_idx][idx])
        return 0.0

# ==================== True RAN Multifunctional Controller ====================

class TrueRANMultifunctionalController:
    """
    True RAN multifunctional controller - RAN provides resistance during AAN trajectory
    RAN mode uses only OIAC control (no ILC feedforward)
    """
    def __init__(self, oiac, ilc):
        self.oiac = oiac
        self.ilc = ilc
        self.current_mode = 'AAN'
        
        # RAN activation parameters - activate RAN during good tracking
        self.error_threshold_aan_to_ran = math.radians(3.0)
        self.error_threshold_ran_to_aan = math.radians(7.0)
        
        # RAN resistance parameters
        self.ran_resistance_level = 2.5  # Base resistance level
        self.ran_velocity_factor = 1.5   # Velocity-dependent resistance
        
        # Mode history
        self.mode_history = []
        
        # RAN state
        self.ran_start_time = 0
        self.last_switch_time = 0
        self.min_switch_interval = 0.5  # Minimum time between switches
        
    def compute_control(self, t, q, qdot, trial_idx):
        """Compute control with true RAN resistance during trajectory"""
        current_time = t
        
        # Always use AAN trajectory for desired position
        q_des = target_angle_rad(t)
        dq_des = target_velocity_rad(t)
        error = q_des - q
        
        # Check if we can switch modes
        can_switch = (current_time - self.last_switch_time) >= self.min_switch_interval
        
        # Update impedance parameters
        K_mat, B_mat = self.oiac.update_impedance(q, q_des, qdot, dq_des, self.current_mode)
        
        # Compute base feedback torque
        pos_error_vec = np.array([[error]])
        vel_error_vec = np.array([[dq_des - qdot]])
        tau_fb = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
        
        # Mode-specific control logic
        if self.current_mode == 'AAN':
            # AAN mode: normal trajectory tracking with ILC feedforward
            tau_ff = self.ilc.get_feedforward(t, trial_idx-1) if trial_idx > 0 else 0.0
            total_torque = tau_ff + tau_fb
            
            # AAN → RAN: Activate resistance when tracking is good
            if can_switch and abs(error) < self.error_threshold_aan_to_ran:
                self.current_mode = 'RAN'
                self.ran_start_time = current_time
                self.last_switch_time = current_time
                print(f"🔄 AAN→RAN at t={t:.1f}s - Activating resistance during motion")
                
        else:
            # RAN mode: Only OIAC control (no ILC feedforward) + resistance
            # Calculate resistance torque (always opposes motion)
            resistance_direction = -1.0 if qdot >= 0 else 1.0
            base_resistance = self.ran_resistance_level * resistance_direction
            velocity_resistance = self.ran_velocity_factor * abs(qdot) * resistance_direction
            
            # Total RAN torque: OIAC feedback only (no feedforward) + resistance
            total_torque = tau_fb + base_resistance + velocity_resistance
            
            # RAN → AAN: Deactivate resistance if tracking deteriorates
            if can_switch and abs(error) > self.error_threshold_ran_to_aan:
                self.current_mode = 'AAN'
                self.last_switch_time = current_time
                print(f"🔄 RAN→AAN at t={t:.1f}s - Deactivating resistance")
        
        # Record mode
        self.mode_history.append(self.current_mode)
        
        return total_torque, q_des, error, self.current_mode

# ==================== Control Parameters ====================

# Moderate torque limits
TORQUE_MIN = -4.1
TORQUE_MAX = 4.1

# Desired trajectory
AMP_DEG = 15.0
FREQ = 0.16

def target_angle_rad(t):
    phase = 2 * np.pi * FREQ * t
    angle_rad = math.radians(55.0) + math.radians(AMP_DEG) * math.sin(phase)
    return float(angle_rad)

def target_velocity_rad(t):
    phase = 2 * np.pi * FREQ * t
    return math.radians(AMP_DEG) * 2 * np.pi * FREQ * math.cos(phase)

# ==================== Main Control Loop ====================

print(f"\n=== Starting True RAN Multifunctional Control ===")
print(f"AAN Trajectory: {AMP_DEG}° @ {FREQ}Hz")
print(f"RAN Mode: Resistance DURING Trajectory (OIAC only, no ILC)")
print(f"Mode Switching: AAN→RAN when error < 3.0°, RAN→AAN when error > 7.0°")
print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}]Nm")
print(f"RAN Resistance: Base={2.5}Nm + {1.5}*velocity")

# Instantiate controllers
oiac = TrueRANOptimizedOIAC(dof=1)
ilc = EnhancedILC(max_trials=10)
multi_controller = TrueRANMultifunctionalController(oiac, ilc)

all_avg_errors = []
all_max_errors = []
all_k_values = []
all_b_values = []
all_mode_distributions = []

for trial in range(ilc.max_trials):
    print(f"\n=== Trial {trial+1}/{ilc.max_trials} ===")
    
    # Reset state
    data.qpos[qpos_adr] = math.radians(55.0)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    
    # Reset controllers
    oiac = TrueRANOptimizedOIAC(dof=1)
    multi_controller.oiac = oiac
    multi_controller.current_mode = 'AAN'
    multi_controller.mode_history = []
    multi_controller.last_switch_time = 0

    time_log = []
    q_log = []
    q_des_log = []
    torque_log = []
    error_log = []
    k_log = []
    b_log = []
    mode_log = []

    t0 = time.time()
    last_debug_time = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            if t > 12.0:
                break

            # Get current state
            q = float(data.qpos[qpos_adr])
            qdot = float(data.qvel[dof_adr])
            
            # Compute control
            torque, q_des, error, current_mode = multi_controller.compute_control(t, q, qdot, trial)
            
            # Saturation
            torque_clipped = max(TORQUE_MIN, min(torque, TORQUE_MAX))
            
            # Apply control
            data.ctrl[act_idx] = torque_clipped

            # Record data
            time_log.append(t)
            q_log.append(q)
            q_des_log.append(q_des)
            torque_log.append(torque_clipped)
            error_log.append(error)
            mode_log.append(current_mode)
            k_log.append(float(oiac.k_mat[0, 0]))
            b_log.append(float(oiac.b_mat[0, 0]))

            # Debug output
            if t - last_debug_time > 1.5:
                actual_deg = math.degrees(q)
                desired_deg = math.degrees(q_des)
                error_deg = math.degrees(error)
                velocity_deg = math.degrees(qdot)
                k_val = float(oiac.k_mat[0, 0])
                b_val = float(oiac.b_mat[0, 0])
                
                mode_info = f"Mode={current_mode}"
                if current_mode == 'RAN':
                    mode_info += " (Resistance ON - OIAC only)"
                else:
                    mode_info += " (ILC + OIAC)"
                
                print(f"t={t:.1f}s: {mode_info}, Angle={actual_deg:6.1f}°, Desired={desired_deg:6.1f}°, Error={error_deg:5.1f}°, Vel={velocity_deg:5.1f}°/s, Torque={torque_clipped:6.1f}Nm")
                last_debug_time = t

            mujoco.mj_step(model, data)
            viewer.sync()

    # Trial results
    avg_error = np.mean(np.abs(error_log))
    max_error = np.max(np.abs(error_log))
    avg_k = np.mean(k_log)
    avg_b = np.mean(b_log)
    
    # Calculate mode distribution
    aan_count = mode_log.count('AAN')
    ran_count = mode_log.count('RAN')
    total_count = len(mode_log)
    aan_ratio = aan_count / total_count * 100
    ran_ratio = ran_count / total_count * 100
    
    # Calculate motion range
    min_angle = math.degrees(min(q_log))
    max_angle = math.degrees(max(q_log))
    motion_range = max_angle - min_angle
    
    all_avg_errors.append(avg_error)
    all_max_errors.append(max_error)
    all_k_values.append(avg_k)
    all_b_values.append(avg_b)
    all_mode_distributions.append((aan_ratio, ran_ratio))
    
    print(f"Trial results:")
    print(f"  Average error: {math.degrees(avg_error):.2f}°")
    print(f"  Maximum error: {math.degrees(max_error):.2f}°")
    print(f"  Motion range: {min_angle:.1f}° to {max_angle:.1f}° (span: {motion_range:.1f}°)")
    print(f"  Average K: {avg_k:.1f}, Average B: {avg_b:.1f}")
    print(f"  Mode distribution: AAN={aan_ratio:.1f}%, RAN={ran_ratio:.1f}%")

    # Check if we're achieving the desired motion range
    if motion_range < 20.0:
        print(f"⚠️  Warning: Motion range too small ({motion_range:.1f}°), expected ~30°")
        if trial == 0:
            print("   This is normal for first trial - ILC needs learning")
    
    # ILC learning update (only for AAN mode)
    if trial < ilc.max_trials - 1:
        ilc.update_learning(time_log, error_log, torque_log)
    
    # # Early stopping check
    # if math.degrees(avg_error) < 3.0 and motion_range > 25.0 and trial >= 2:
    #     print(f"🎉 Excellent performance! Good tracking + full motion range")
    #     break

# Result analysis
print(f"\n=== True RAN Multifunctional Control Final Results ===")
print(f"Completed trials: {len(all_avg_errors)}")
print(f"Final average error: {math.degrees(all_avg_errors[-1]):.2f}°")
print(f"Final maximum error: {math.degrees(all_max_errors[-1]):.2f}°")

# Calculate final motion range
final_min_angle = math.degrees(min(q_log))
final_max_angle = math.degrees(max(q_log))
final_motion_range = final_max_angle - final_min_angle
print(f"Final motion range: {final_min_angle:.1f}° to {final_max_angle:.1f}° (span: {final_motion_range:.1f}°)")

if len(all_avg_errors) > 1:
    improvement = math.degrees(all_avg_errors[0] - all_avg_errors[-1])
    print(f"Error improvement: {improvement:.2f}°")

final_aan_ratio, final_ran_ratio = all_mode_distributions[-1]
print(f"Final mode distribution: AAN={final_aan_ratio:.1f}%, RAN={final_ran_ratio:.1f}%")

# Enhanced Visualization
plt.figure(figsize=(16, 12))

# Learning progress with motion range
plt.subplot(3, 2, 1)
trials = range(1, len(all_avg_errors)+1)
avg_errors_deg = [math.degrees(e) for e in all_avg_errors]
max_errors_deg = [math.degrees(e) for e in all_max_errors]

plt.plot(trials, avg_errors_deg, 'o-', linewidth=2, label='Average Error')
plt.plot(trials, max_errors_deg, 's-', linewidth=2, label='Maximum Error')
plt.axhline(y=3.0, color='r', linestyle='--', label='AAN→RAN Threshold (3.0°)')
plt.axhline(y=7.0, color='g', linestyle='--', label='RAN→AAN Threshold (7.0°)')
plt.xlabel('Trial Number')
plt.ylabel('Error (°)')
plt.legend()
plt.grid(True)
plt.title('Learning Progress with True RAN Switching')

# Trajectory tracking with mode coloring
plt.subplot(3, 2, 2)
plt.plot(time_log, [math.degrees(q) for q in q_log], label='Actual Angle', linewidth=2)
plt.plot(time_log, [math.degrees(q) for q in q_des_log], '--', label='Desired Angle', linewidth=2)

# Color background based on mode
if mode_log:
    current_mode = mode_log[0]
    start_idx = 0
    for i, mode in enumerate(mode_log):
        if mode != current_mode:
            color = 'lightblue' if current_mode == 'AAN' else 'lightcoral'
            plt.axvspan(time_log[start_idx], time_log[i], alpha=0.3, color=color, 
                       label='AAN Mode' if current_mode == 'AAN' and start_idx == 0 else "" or
                             'RAN Mode' if current_mode == 'RAN' and start_idx == 0 else "")
            current_mode = mode
            start_idx = i

    # Add the last segment
    color = 'lightblue' if current_mode == 'AAN' else 'lightcoral'
    plt.axvspan(time_log[start_idx], time_log[-1], alpha=0.3, color=color)

plt.xlabel('Time (s)')
plt.ylabel('Angle (°)')
plt.legend()
plt.grid(True)
plt.title('Trajectory Tracking with True RAN Resistance\n(Blue=AAN=ILC+OIAC, Red=RAN=OIAC only)')

# Control torque
plt.subplot(3, 2, 3)
plt.plot(time_log, torque_log, 'orange', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Control Torque (Nm)')
plt.grid(True)
plt.title('Control Torque (Resistance in RAN Mode)')

# Tracking error with mode thresholds
plt.subplot(3, 2, 4)
plt.plot(time_log, [math.degrees(e) for e in error_log], 'purple', linewidth=2)
plt.axhline(y=3.0, color='r', linestyle='--', label='AAN→RAN Threshold')
plt.axhline(y=7.0, color='g', linestyle='--', label='RAN→AAN Threshold')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Tracking Error (°)')
plt.legend()
plt.grid(True)
plt.title('Tracking Error with True RAN Thresholds')

# Mode distribution over trials
plt.subplot(3, 2, 5)
trials = range(1, len(all_mode_distributions) + 1)
aan_ratios = [dist[0] for dist in all_mode_distributions]
ran_ratios = [dist[1] for dist in all_mode_distributions]

plt.plot(trials, aan_ratios, 'bo-', linewidth=2, label='AAN Mode %')
plt.plot(trials, ran_ratios, 'ro-', linewidth=2, label='RAN Mode %')
plt.xlabel('Trial Number')
plt.ylabel('Mode Distribution (%)')
plt.legend()
plt.grid(True)
plt.title('Mode Distribution Over Trials')

# Impedance parameters over trials
plt.subplot(3, 2, 6)
plt.plot(trials, all_k_values, 'g^-', linewidth=2, label='Average K')
plt.plot(trials, all_b_values, 'mv-', linewidth=2, label='Average B')
plt.xlabel('Trial Number')
plt.ylabel('Impedance Parameters')
plt.legend()
plt.grid(True)
plt.title('Adaptive Impedance Parameters Over Trials')

plt.tight_layout()
plt.show()

print("=== True RAN Multifunctional Control Completed ===")

# ==================== Professional AAN vs RAN Analysis ====================

print(f"\n=== Professional AAN vs RAN Analysis ===")

# 计算速度误差 (e_dot)
e_dot_log = []
for i in range(len(time_log)):
    q_des = target_angle_rad(time_log[i])
    dq_des = target_velocity_rad(time_log[i])
    e_dot = dq_des - data.qvel[dof_adr] if i < len(q_log) else 0
    e_dot_log.append(e_dot)

# 分离AAN和RAN模式的数据
aan_indices = [i for i, mode in enumerate(mode_log) if mode == 'AAN']
ran_indices = [i for i, mode in enumerate(mode_log) if mode == 'RAN']

print(f"AAN mode data points: {len(aan_indices)}")
print(f"RAN mode data points: {len(ran_indices)}")

# 转换为度数和度数/秒的单位
torque_log = np.array(torque_log)
k_log = np.array(k_log)
b_log = np.array(b_log)
error_deg_log = np.degrees(np.array(error_log))
e_dot_deg_log = np.degrees(np.array(e_dot_log))

# 创建专业风格的图表
plt.style.use('seaborn-v0_8-whitegrid')

# 图1: AAN模式专业图表
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

if aan_indices:
    aan_time = [time_log[i] for i in aan_indices]
    aan_torque = torque_log[aan_indices]
    aan_k = k_log[aan_indices]
    aan_b = b_log[aan_indices]
    aan_error = error_deg_log[aan_indices]
    aan_e_dot = e_dot_deg_log[aan_indices]
    
    # 控制力矩
    ax1.plot(aan_time, aan_torque, 'b-', linewidth=2, label='Torque')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_title('AAN Mode: Control Torque', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 阻抗参数
    ax2.plot(aan_time, aan_k, 'g-', linewidth=2, label='K')
    ax2.plot(aan_time, aan_b, 'r-', linewidth=2, label='B')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Impedance Parameters')
    ax2.set_title('AAN Mode: Adaptive Impedance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 位置误差
    ax3.plot(aan_time, aan_error, 'purple', linewidth=2, label='e')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='AAN→RAN Threshold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (°)')
    ax3.set_title('AAN Mode: Position Error (e)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 速度误差
    ax4.plot(aan_time, aan_e_dot, 'orange', linewidth=2, label='e_dot')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity Error (°/s)')
    ax4.set_title('AAN Mode: Velocity Error (e_dot)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

plt.tight_layout()
plt.show()

# 图2: RAN模式专业图表
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

if ran_indices:
    ran_time = [time_log[i] for i in ran_indices]
    ran_torque = torque_log[ran_indices]
    ran_k = k_log[ran_indices]
    ran_b = b_log[ran_indices]
    ran_error = error_deg_log[ran_indices]
    ran_e_dot = e_dot_deg_log[ran_indices]
    
    # 控制力矩
    ax1.plot(ran_time, ran_torque, 'r-', linewidth=2, label='Torque')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_title('RAN Mode: Control Torque', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 阻抗参数
    ax2.plot(ran_time, ran_k, 'g-', linewidth=2, label='K')
    ax2.plot(ran_time, ran_b, 'r-', linewidth=2, label='B')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Impedance Parameters')
    ax2.set_title('RAN Mode: Adaptive Impedance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 位置误差
    ax3.plot(ran_time, ran_error, 'purple', linewidth=2, label='e')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=7.0, color='green', linestyle='--', alpha=0.7, label='RAN→AAN Threshold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (°)')
    ax3.set_title('RAN Mode: Position Error (e)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 速度误差
    ax4.plot(ran_time, ran_e_dot, 'orange', linewidth=2, label='e_dot')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity Error (°/s)')
    ax4.set_title('RAN Mode: Velocity Error (e_dot)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

plt.tight_layout()
plt.show()

# 统计信息表格
if aan_indices and ran_indices:
    print(f"\n=== Mode Comparison Statistics ===")
    
    aan_stats = {
        'Duration (s)': f"{aan_time[-1]-aan_time[0]:.1f}",
        'Avg Torque': f"{np.mean(aan_torque):.2f} Nm",
        'Avg K': f"{np.mean(aan_k):.1f}",
        'Avg B': f"{np.mean(aan_b):.1f}",
        'Avg |e|': f"{np.mean(np.abs(aan_error)):.2f}°",
        'Avg |e_dot|': f"{np.mean(np.abs(aan_e_dot)):.2f}°/s",
        'Max |e|': f"{np.max(np.abs(aan_error)):.2f}°"
    }
    
    ran_stats = {
        'Duration (s)': f"{ran_time[-1]-ran_time[0]:.1f}",
        'Avg Torque': f"{np.mean(ran_torque):.2f} Nm",
        'Avg K': f"{np.mean(ran_k):.1f}",
        'Avg B': f"{np.mean(ran_b):.1f}",
        'Avg |e|': f"{np.mean(np.abs(ran_error)):.2f}°",
        'Avg |e_dot|': f"{np.mean(np.abs(ran_e_dot)):.2f}°/s",
        'Max |e|': f"{np.max(np.abs(ran_error)):.2f}°"
    }
    
    print("\nAAN Mode Statistics:")
    for key, value in aan_stats.items():
        print(f"  {key}: {value}")
    
    print("\nRAN Mode Statistics:")
    for key, value in ran_stats.items():
        print(f"  {key}: {value}")
    
    # 创建对比统计图
    fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['Avg Torque', 'Avg K', 'Avg B', 'Avg |e|', 'Avg |e_dot|']
    aan_values = [float(aan_stats[m].split()[0]) for m in metrics]
    ran_values = [float(ran_stats[m].split()[0]) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, aan_values, width, label='AAN Mode', color='blue', alpha=0.7)
    ax.bar(x + width/2, ran_values, width, label='RAN Mode', color='red', alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('AAN vs RAN Mode Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()