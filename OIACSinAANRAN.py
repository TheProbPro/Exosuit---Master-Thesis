# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from OIAC_Controllers import ada_imp_con, ILC, ILCv1, ILCv2, ModeControllerThreshold, ModeControllerUpDown, ControlMode

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

MOTOR_POS_MIN = 2280
MOTOR_POS_MAX = 1145

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

def sine_velocity(step, speed=0.05, min_val=0, max_val=140):
    """
    Returns the velocity of a sine-wave oscillation.

    Parameters:
        step (int): Increasing integer input 1, 2, 3, ...
        speed (float): Frequency scaling (same as position function).
        min_val (float): Minimum position value.
        max_val (float): Maximum position value.
    """
    amplitude = (max_val - min_val) / 2
    x = step * speed
    return amplitude * speed * math.cos(x)

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
    ILC_controller = ILC()
    # ILC_controller = ILCv1(max_trials=10)
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
        trial_fb_log = []

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
                desired_velocity_rad = sine_velocity(i, speed=0.1)
                last_desired_angle = desired_angle_deg
                step = (MOTOR_POS_MIN - MOTOR_POS_MAX)/140
                motor_pos = motor.get_position()[0]
                current_angle_deg = (MOTOR_POS_MIN - motor_pos) / step
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity()[0]
                position_error = desired_angle_rad - current_angle_rad
                velocity_error = desired_velocity_rad - current_velocity

                current_mode = mode_controller.current_mode
                mode_changed = mode_controller.update_mode(position_error, current_angle_rad, desired_angle_rad, current_time)

                if mode_changed:
                    current_mode = mode_controller.current_mode

                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity_rad)

                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0

                if current_mode == ControlMode.AAN:
                    #integral = position_error * dt
                    ff_torque = 0.0
                    #if position_error < math.radians(1):
                    #    integral = np.clip(integral, -15, 15)  # Anti-windup
                    #else:
                    #    integral *= 0.9
                    #integral_torque = 5.0 * integral
                    if trial > 0:
                        ff_torque = ILC_controller.get_feedforward()
                        #print(f"ff torque from ILC: {ff_torque}")
                    #total_torque = tau_fb + integral_torque + ff_torque
                    total_torque = tau_fb + ff_torque
                    #print(f"ff torque: {ff_torque},\n tau_fb: {tau_fb},\n total: {total_torque}")
                else:
                    total_torque = tau_fb
                    #integral_torque = 0.0
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
                trial_fb_log.append(tau_fb)
                trial_mode_log.append(current_mode)

                current = motor.torq2curcom(torque_clipped)
                if motor_pos < MOTOR_POS_MAX and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                elif motor_pos > MOTOR_POS_MIN and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                else:
                    motor.sendMotorCommand(motor.motor_ids[0], current)

                #time.sleep(0.001)  # Sleep briefly to yield CPU
                i += 1

        except Exception as e:
            print(f"Exception during trial: {e}")
        
        finally:
            motor.sendMotorCommand(motor.motor_ids[0], 0)
            i = 0
        
        print("max ff torque this trial: ", np.max(np.abs(trial_ff_log)), "Nm, and max fb torque: ", np.max(np.abs(trial_fb_log)), "Nm")
        print(f"Trial error log size: {len(trial_error_log)}")
        #plot fb, ff, tau
        plt.figure(figsize=(10, 6))

        plt.subplot(3,1,1)
        plt.plot(trial_fb_log, label='Feedback Torque (Nm)')  # 修正变量名
        plt.title(f'Trial {trial + 1} Feedback Torque')
        plt.xlabel('Time Steps')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid()

        plt.subplot(3,1,2)
        plt.plot(trial_ff_log, label='Feedforward Torque (Nm)', color='orange')
        plt.title(f'Trial {trial + 1} Feedforward Torque')
        plt.xlabel('Time Steps')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid()

        plt.subplot(3,1,3)
        plt.plot(trial_torque_log, label='Total Applied Torque (Nm)')
        plt.title(f'Trial {trial + 1} Control Torque')
        plt.xlabel('Time Steps')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.grid()

        plt.tight_layout()  # 添加这行使子图布局更紧凑
        plt.show()

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
                'trial': trial + 1,
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

            if trial < trial_num:
                # ILC_controller.update_learning(trial_time_log, trial_error_log, trial_torque_log)
                ILC_controller.update_learning(trial_error_log)


            # 在绘图之前添加诊断信息  在这里开始 
            
            if len(trial_error_log) > 0:
                # 诊断信息
                print(f"\n=== 诊断信息 Trial {trial + 1} ===")
                print(f"trial_fb_log 长度: {len(trial_fb_log)}")
                print(f"trial_ff_log 长度: {len(trial_ff_log)}")
                print(f"trial_torque_log 长度: {len(trial_torque_log)}")
                
                print(f"\ntrial_fb_log 范围: [{min(trial_fb_log):.4f}, {max(trial_fb_log):.4f}]")
                print(f"trial_ff_log 范围: [{min(trial_ff_log):.4f}, {max(trial_ff_log):.4f}]")
                print(f"trial_torque_log 范围: [{min(trial_torque_log):.4f}, {max(trial_torque_log):.4f}]")
                
                print(f"\ntrial_fb_log 前5个值: {trial_fb_log[:5]}")
                print(f"trial_ff_log 前5个值: {trial_ff_log[:5]}")
                
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
                    'trial': trial + 1,
                    'avg_error_deg': math.degrees(avg_error),
                    'max_error_deg': math.degrees(max_error),
                    'avg_k': avg_k,
                    'avg_b': avg_b,
                    'aan_percentage': aan_percentage,
                    'ran_percentage': ran_percentage
                }
                all_trial_stats.append(trial_stats)
                
                print(f"\nAverage tracking error: {math.degrees(avg_error):.2f}°")
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
                           
            else:
                print("No data recorded for this trial.")
            
        # else:
        #     print("No data recorded for this trial.")

    

    # TODO: Implement other up down controller
    # TODO: Implement threshold controller underneath
    print("Press enter to run final trial with learned feedforward")
    input()
    i = 0
    last_time = time.time()
    loop_timer = time.time()
    trial_start_time = time.time()
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
                plot_desired_position.append(desired_angle_deg)
                desired_angle_rad = math.radians(desired_angle_deg)
                #desired_velocity = (desired_angle_deg - last_desired_angle) / dt if dt > 0 else 0.0
                #desired_velocity_rad = math.radians(desired_velocity)
                desired_velocity_rad = sine_velocity(i, speed=0.1)
                last_desired_angle = desired_angle_deg
                # step = 1500/140
                step = (MOTOR_POS_MIN - MOTOR_POS_MAX)/140
                motor_pos = motor.get_position()[0]
                # current_angle_deg = (2550 - motor_pos[0]) / step
                current_angle_deg = (MOTOR_POS_MIN - motor_pos) / step
                plot_position.append(current_angle_deg)
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity()[0]
                position_error = desired_angle_rad - current_angle_rad
                plot_error.append(math.degrees(position_error))
                velocity_error = desired_velocity_rad - current_velocity

                current_mode = mode_controller.current_mode
                mode_changed = mode_controller.update_mode(position_error, current_angle_rad, desired_angle_rad, current_time)

                if mode_changed:
                    current_mode = mode_controller.current_mode

                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity_rad)

                pos_error_vec = np.array([[position_error]])
                vel_error_vec = np.array([[velocity_error]])

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0

                if current_mode == ControlMode.AAN:
                    integral = position_error * dt
                    ff_torque = ILC_controller.get_feedforward()
                    if position_error < math.radians(1):
                        integral = np.clip(integral, -15, 15)  # Anti-windup
                    else:
                        integral *= 0.9
                    integral_torque = 5.0 * integral
                    total_torque = tau_fb + integral_torque + ff_torque
                else:
                    total_torque = tau_fb
                    integral_torque = 0.0
                    ff_torque = 0.0
                
                torque_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)
                plot_torque.append(torque_clipped)

                current = motor.torq2curcom(torque_clipped)
                if motor_pos < MOTOR_POS_MAX and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                elif motor_pos > MOTOR_POS_MIN and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                else:
                    motor.sendMotorCommand(motor.motor_ids[0], current)

                time.sleep(0.005)  # Sleep briefly to yield CPU
                i += 1
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