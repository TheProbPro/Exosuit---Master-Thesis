# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from OIAC_Controllers import ada_imp_con, ILCv1, ILCv2, ModeControllerThreshold, ModeControllerUpDown, ControlMode

# General imports
import numpy as np
import threading
import signal
import time
import matplotlib.pyplot as plt

# General configuration parameters
SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = 0
ANGLE_MAX = 140

TORQUE_MIN = -4.1
TORQUE_MAX = 4.1

stop_event = threading.Event()

def sine_position(step, speed=0.05, min_val=0, max_val=140):
    """
    Returns a smooth sine-wave value between min_val and max_val.
    
    Parameters:
        step (int): Increasing integer input 1, 2, 3, ...
        speed (float): Smaller = smoother & slower oscillation. Default 0.05.
        min_val (float): Minimum value of the oscillation.
        max_val (float): Maximum value of the oscillation.
    """
    amplitude = (max_val - min_val) / 2
    offset = min_val + amplitude
    x = step * speed
    return amplitude * math.sin(x) + offset

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    plot_position = []
    plot_desired_position = []
    plot_error = []
    plot_torque = []

    motor = Motors(port="COM4")

    # Wait a moment before starting
    time.sleep(1.0)
    print("Motor command threads started!")

    # Filter and interpret the raw data
    joint_torque = 0.0
    last_desired_angle = 0
    i = 0
    OIAC = ada_imp_con(1) # 1 degree of freedom
    #TODO: test the two mode controllers with iterative learning on exo
    ILC_controller = ILCv1(max_trials=10)
    # ILC_controller = ILCv2(max_trials=10, alpha=0.1)
    mode_controller = ModeControllerThreshold()
    # mode_controller = ModeControllerUpDown(OIAC, ILC_controller)

    # Run trial
    all_trial_stats = []
    trial_num = 10

    for trial in range(trial_num):
        print("Press enter to start trial")
        input()
        trial_start_time = time.time()
        last_time = time.time()
        trial_time_log = []
        trial_error_log = []
        trial_torque_log = []
        trial_desired_angle_log = []
        trial_current_angle_log = []
        trial_k_log = []
        trial_b_log = []
        trial_ff_log = []
        trial_mode_log = []

        try:
            while not stop_event.is_set():
                elapsed_time = time.time() - trial_start_time
                if elapsed_time > 10:  # Each trial lasts 10 seconds
                    break
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                desired_angle_deg = sine_position(i, speed=0.1)
                desired_angle_rad = math.radians(desired_angle_deg)
                desired_velocity = (desired_angle_deg - last_desired_angle) / dt if dt > 0 else 0.0
                last_desired_angle = desired_angle_deg
                # step = 1500/140
                step = (2280-1145)/140
                motor_pos = motor.get_position()
                # current_angle_deg = (2550 - motor_pos[0]) / step
                current_angle_deg = (2280 - motor_pos[0]) / step
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity()
                position_error = desired_angle_rad - current_angle_rad
                velocity_error = desired_velocity - current_velocity

                current_mode = mode_controller.current_mode
                mode_changed = mode_controller.update_mode(position_error, current_angle_rad, desired_angle_rad, current_time)

                if mode_changed:
                    current_mode = mode_controller.current_mode

                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity)

                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0

                if current_mode == ControlMode.AAN:
                    integral = position_error * dt
                    ff_torque = 0.0
                    if position_error < math.radians(1):
                        integral = np.clip(integral, -15, 15)  # Anti-windup
                    else:
                        integral *= 0.9
                    integral_torque = 5.0 * integral
                    if trial_num > 0:
                        ff_torque = ILC_controller.get_feedforward(elapsed_time, trial_num - 1)
                    total_torque = tau_fb + integral_torque + ff_torque
                else:
                    total_torque = tau_fb
                    integral_torque = 0.0
                    ff_torque = 0.0
                
                torque_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)

                trial_time_log.append(10)
                trial_error_log.append(position_error)
                trial_torque_log.append(torque_clipped)
                trial_desired_angle_log.append(desired_angle_deg)
                trial_current_angle_log.append(current_angle_deg)
                trial_k_log.append(K_mat[0,0])
                trial_b_log.append(B_mat[0,0])
                trial_ff_log.append(ff_torque)
                trial_mode_log.append(current_mode)

                current = motor.torq2curcom(torque_clipped)
                if motor_pos < 1050 and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                elif motor_pos > 2550 and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                else:
                    motor.sendMotorCommand(motor.motor_ids[0], current)

                time.sleep(0.005)  # Sleep briefly to yield CPU
                i += 1

        except Exception as e:
            print(f"Exception during trial: {e}")
        
        finally:
            motor.sendMotorCommand(motor.motor_ids[0], 0)

        if len(trial_error_log) > 0:
            avg_error = np.mean(np.abs(trial_error_log))
            max_error = np.max(np.abs(trial_error_log))
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
                'avg_k': avg_k,
                'avg_b': avg_b,
                'aan_percentage': aan_percentage,
                'ran_percentage': ran_percentage
            }
            all_trial_stats.append(trial_stats)
            
            print(f"Average tracking error: {math.degrees(avg_error):.2f}°")
            print(f"Maximum tracking error: {math.degrees(max_error):.2f}°")
            print(f"Average K: {avg_k:.8f}, Average B: {avg_b:.8f}")
            print(f"Mode usage: AAN={aan_percentage:.1f}%, RAN={ran_percentage:.1f}%")
            print(f"Mode switches: {len(mode_controller.mode_history)}")
            
            # 显示模式切换历史
            if mode_controller.mode_history:
                print("\nMode switch history:")
                for switch in mode_controller.mode_history:
                    print(f"  t={switch['time']-trial_start_time:.1f}s: "
                          f"{switch['from']} -> {switch['to']}")

            if trial_num < 10:
                ILCv1.update_learning(trial_time_log, trial_error_log, trial_torque_log)
            
        else:
            print("No data recorded for this trial.")

    

    # TODO: Implement other up down controller
    # TODO: Implement threshold controller underneath
    i = 0
    last_time = time.time()
    loop_timer = time.time()
    #while not stop_event.is_set():
    try:
        while not stop_event.is_set():
                elapsed_time = time.time() - trial_start_time
                if elapsed_time > 10:  # Each trial lasts 10 seconds
                    break
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                desired_angle_deg = sine_position(i, speed=0.1)
                desired_angle_rad = math.radians(desired_angle_deg)
                desired_velocity = (desired_angle_deg - last_desired_angle) / dt if dt > 0 else 0.0
                last_desired_angle = desired_angle_deg
                # step = 1500/140
                step = (2280-1145)/140
                motor_pos = motor.get_position()
                # current_angle_deg = (2550 - motor_pos[0]) / step
                current_angle_deg = (2280 - motor_pos[0]) / step
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity()
                position_error = desired_angle_rad - current_angle_rad
                velocity_error = desired_velocity - current_velocity

                current_mode = mode_controller.current_mode
                mode_changed = mode_controller.update_mode(position_error, current_angle_rad, desired_angle_rad, current_time)

                if mode_changed:
                    current_mode = mode_controller.current_mode

                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity)

                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0

                if current_mode == ControlMode.AAN:
                    integral = position_error * dt
                    ff_torque = 0.0
                    if position_error < math.radians(1):
                        integral = np.clip(integral, -15, 15)  # Anti-windup
                    else:
                        integral *= 0.9
                    integral_torque = 5.0 * integral
                    if trial_num > 0:
                        ff_torque = ILC_controller.get_feedforward(elapsed_time, trial_num - 1)
                    total_torque = tau_fb + integral_torque + ff_torque
                else:
                    total_torque = tau_fb
                    integral_torque = 0.0
                    ff_torque = 0.0
                
                torque_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)

                current = motor.torq2curcom(torque_clipped)
                if motor_pos < 1050 and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                elif motor_pos > 2550 and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                else:
                    motor.sendMotorCommand(motor.motor_ids[0], current)

                time.sleep(0.005)  # Sleep briefly to yield CPU
                i += 1
        while last_time - loop_timer < 10:  # Run for 10 seconds
            current_time = time.time()
            dt = current_time - last_time

            desired_angle_deg = sine_position(i, speed=0.1)
            desired_angle_rad = math.radians(desired_angle_deg)
            desired_velocity = (desired_angle_deg - last_desired_angle) / dt if dt > 0 else 0.0
            last_desired_angle = desired_angle_deg
            # step = 1500/140
            step = (2280-1145)/140
            motor_pos = motor.get_position()
            # current_angle_deg = (2550 - motor_pos) / step
            current_angle_deg = (2280 - motor_pos[0]) / step
            current_angle_rad = math.radians(current_angle_deg)
            current_velocity = motor.get_velocity()
            plot_position.append(current_angle_deg)
            plot_error.append(current_angle_deg - desired_angle_deg)
            plot_desired_position.append(desired_angle_deg)

            # print(f"current_angle_deg: {current_angle_deg}, desired_angle_deg: {desired_angle_deg}, error: {current_angle_deg - desired_angle_deg}, current_velocity: {current_velocity}, desired_velocity: {desired_velocity}")

            # OIAC online impedance adaptation
            K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity) #TODO: is this correct?
            # K_ma, B_mat = OIAC.update_impedance(current_angle_deg, desired_angle_deg, current_velocity, desired_velocity)
            tau_fb = OIAC.calc_tau_fb()[0,0] # TODO: This might have to swap sign
            #print(f"tau_fb: {tau_fb}")
            total_torque = 0.0

            if current_mode == ControlMode.AAN:
                integral = position_error * dt
                ff_torque = 0.0
                if position_error < math.radians(1):
                    integral = np.clip(integral, -15, 15)  # Anti-windup
                else:
                    integral *= 0.9
                integral_torque = 5.0 * integral
                if trial_num > 0:
                    ff_torque = ILC_controller.get_feedforward(elapsed_time, trial_num - 1)
                total_torque = tau_fb + integral_torque + ff_torque
            else:
                total_torque = tau_fb
                integral_torque = 0.0
                ff_torque = 0.0

            tau_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)
            plot_torque.append(float(tau_clipped))

            current = motor.torq2curcom(tau_clipped)
            #print("motor torque: ", tau_clipped, "motor position: ", motor_pos)
            
            if motor_pos < 1050 and tau_clipped < 0:
                motor.sendMotorCommand(motor.motor_ids[0], 0)
            elif motor_pos > 2550 and tau_clipped > 0:
                motor.sendMotorCommand(motor.motor_ids[0], 0)
            else:
                motor.sendMotorCommand(motor.motor_ids[0], current)

            time.sleep(0.005)  # Sleep briefly to yield CPU
            i += 1
            
            last_time = current_time
    except Exception as e:
        print(f"Exception during final run: {e}")

    # Stop EMG reading thread and EMG sensor
    motor.sendMotorCommand(motor.motor_ids[0], 0)
    print("Shutting down...")
    stop_event.set()
    motor.close()
    # empty all queues
    
    print("plotting data...")
    #plot desired and actual position in one graph and error in another graph
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(plot_position, label='Actual Position (deg)')
    plt.plot(plot_desired_position, label='Desired Position (deg)', linestyle='--')
    plt.title('Actual vs Desired Position')
    plt.xlabel('Time Steps')
    plt.ylabel('Position (deg)')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(plot_error, label='Position Error (deg)', color='red')
    plt.title('Position Error Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Error (deg)')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(plot_torque, label='Applied Torque (Nm)', color='green')
    plt.title('Applied Torque Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("Goodbye!")
# if __name__ == "__main__":
#     # Test sin wave
#     plot = []

#     last_time = time.time()

#     for i in range(1000):
#         current_time = time.time()
#         dt = current_time - last_time
#         last_time = current_time
#         value = sine_position(i)
#         plot.append(value)

#     plt.plot(plot)
#     plt.title("Sine Wave Test")
#     plt.show()