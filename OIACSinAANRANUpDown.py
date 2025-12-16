# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from OIAC_Controllers import ada_imp_con, ILC, ILCv1, ILCv2, ModeControllerThreshold, ModeControllerUpDown, ModeControllerUpDownv1, ControlMode

# General imports
import numpy as np
import threading
import signal
import time
import matplotlib.pyplot as plt

# General configuration parameters
SAMPLE_RATE = 166.7  # Hz
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
    plot_control_mode = []

    motor = Motors(port="COM4")

    # Wait a moment before starting
    time.sleep(1.0)
    print("Motor command threads started!")

    # Filter and interpret the raw data
    joint_torque = 0.0
    last_desired_angle = 0
    i = 0
    OIAC = ada_imp_con(1) # 1 degree of freedom
    ILC_controller = ILC()
    mode_controller = ModeControllerUpDownv1()

    # Run trial
    all_trial_stats = []
    trial_num = 10
    SIN_SPEED = 2

    for trial in range(trial_num):
        print("Press enter to start trial")
        input()
        trial_start_time = time.time()
        last_time = time.time()
        trial_error_log = []
        trial_torque_log = []
        trial_k_log = []
        trial_b_log = []
        trial_ff_log = []
        trial_mode_log = []
        trial_fb_log = []
        trial_desired_position = []

        mode_controller.reset()

        try:
            while not stop_event.is_set():
                current_time = time.time()
                elapsed_time = current_time - trial_start_time
                if elapsed_time > 10:  # Each trial lasts 10 seconds
                    break
                dt = current_time - last_time
                last_time = current_time
                desired_angle_deg = sine_position(elapsed_time, speed=SIN_SPEED)
                trial_desired_position.append(desired_angle_deg)
                desired_angle_rad = math.radians(desired_angle_deg)
                desired_velocity_deg = sine_velocity(elapsed_time, speed=SIN_SPEED)
                desired_velocity_rad = math.radians(desired_velocity_deg)
                last_desired_angle = desired_angle_deg
                step = (MOTOR_POS_MIN - MOTOR_POS_MAX)/140
                motor_pos = motor.get_position()[0]
                current_angle_deg = (MOTOR_POS_MIN - motor_pos) / step
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity()[0]
                position_error = current_angle_rad - desired_angle_rad
                velocity_error = desired_velocity_rad - current_velocity

                current_mode = mode_controller.current_mode

                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity_rad)

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0

                if current_mode == ControlMode.AAN:
                    ff_torque = 0.0
                    if trial > 0:
                        ff_torque = ILC_controller.get_feedforward()
                    total_torque = tau_fb + ff_torque
                else:
                    total_torque = tau_fb
                    ff_torque = 0.0
                torque_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)

                current_mode = mode_controller.update_control_mode(torque_clipped, elapsed_time)

                trial_error_log.append(position_error)
                trial_torque_log.append(torque_clipped)
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

                i += 1

        except Exception as e:
            print(f"Exception during trial: {e}")
        
        finally:
            motor.sendMotorCommand(motor.motor_ids[0], 0)
            i = 0
        
        print("max ff torque this trial: ", np.max(np.abs(trial_ff_log)), "Nm, and max fb torque: ", np.max(np.abs(trial_fb_log)), "Nm")
        print(f"Trial error log size: {len(trial_error_log)}")
        #plot fb, ff, tau
        # plt.figure(figsize=(10, 6))

        # plt.subplot(4,1,1)
        # plt.plot(trial_desired_position, label='Desired Position (deg)')
        # plt.title(f'Trial {trial + 1} Desired Position')
        # plt.xlabel('Time Steps')
        # plt.ylabel('Position (deg)')
        # plt.legend()
        # plt.grid()

        # plt.subplot(4,1,2)
        # plt.plot(trial_fb_log, label='Feedback Torque (Nm)')  # 修正变量名
        # plt.title(f'Trial {trial + 1} Feedback Torque')
        # plt.xlabel('Time Steps')
        # plt.ylabel('Torque (Nm)')
        # plt.legend()
        # plt.grid()

        # plt.subplot(4,1,3)
        # plt.plot(trial_ff_log, label='Feedforward Torque (Nm)', color='orange')
        # plt.title(f'Trial {trial + 1} Feedforward Torque')
        # plt.xlabel('Time Steps')
        # plt.ylabel('Torque (Nm)')
        # plt.legend()
        # plt.grid()

        # plt.subplot(4,1,4)
        # plt.plot(trial_torque_log, label='Total Applied Torque (Nm)')
        # plt.title(f'Trial {trial + 1} Control Torque')
        # plt.xlabel('Time Steps')
        # plt.ylabel('Torque (Nm)')
        # plt.legend()
        # plt.grid()

        # plt.tight_layout()  # 添加这行使子图布局更紧凑
        # plt.show()

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
                ILC_controller.update_learning(trial_error_log)

            # 在绘图之前添加诊断信息  在这里开始 
            
            # if len(trial_error_log) > 0:
            #     # 诊断信息
            #     print(f"\n=== 诊断信息 Trial {trial + 1} ===")
            #     print(f"trial_fb_log 长度: {len(trial_fb_log)}")
            #     print(f"trial_ff_log 长度: {len(trial_ff_log)}")
            #     print(f"trial_torque_log 长度: {len(trial_torque_log)}")
                
            #     print(f"\ntrial_fb_log 范围: [{min(trial_fb_log):.4f}, {max(trial_fb_log):.4f}]")
            #     print(f"trial_ff_log 范围: [{min(trial_ff_log):.4f}, {max(trial_ff_log):.4f}]")
            #     print(f"trial_torque_log 范围: [{min(trial_torque_log):.4f}, {max(trial_torque_log):.4f}]")
                
            #     print(f"\ntrial_fb_log 前5个值: {trial_fb_log[:5]}")
            #     print(f"trial_ff_log 前5个值: {trial_ff_log[:5]}")
                
            #     avg_error = np.mean(np.abs(trial_error_log))
            #     max_error = np.max(np.abs(trial_error_log))
            #     avg_k = np.mean(trial_k_log)
            #     avg_b = np.mean(trial_b_log)
                
            #     # 统计模式使用情况
            #     aan_count = sum(1 for m in trial_mode_log if m == ControlMode.AAN)
            #     ran_count = sum(1 for m in trial_mode_log if m == ControlMode.RAN)
            #     total_count = len(trial_mode_log)
            #     aan_percentage = (aan_count / total_count * 100) if total_count > 0 else 0
            #     ran_percentage = (ran_count / total_count * 100) if total_count > 0 else 0
                
            #     trial_stats = {
            #         'trial': trial + 1,
            #         'avg_error_deg': math.degrees(avg_error),
            #         'max_error_deg': math.degrees(max_error),
            #         'avg_k': avg_k,
            #         'avg_b': avg_b,
            #         'aan_percentage': aan_percentage,
            #         'ran_percentage': ran_percentage
            #     }
            #     all_trial_stats.append(trial_stats)
                
            #     print(f"\nAverage tracking error: {math.degrees(avg_error):.2f}°")
            #     print(f"Maximum tracking error: {math.degrees(max_error):.2f}°")
            #     print(f"Average K: {avg_k:.8f}, Average B: {avg_b:.8f}")
            #     print(f"Mode usage: AAN={aan_percentage:.1f}%, RAN={ran_percentage:.1f}%")
            #     print(f"Mode switches: {len(mode_controller.mode_history)}")
                
            #     # 显示模式切换历史
            #     if mode_controller.mode_history:
            #         print("\nMode switch history:")
            #         for switch in mode_controller.mode_history:
            #             print(f"  t={switch['time']-trial_start_time:.1f}s: "
            #                 f"{switch['from']} -> {switch['to']}")
                           
        else:
            print("No data recorded for this trial.")
            
    
    # =========================== Free run ===========================
    mode_controller.reset()
    print("Press enter to run final trial with learned feedforward")
    input()
    i = 0
    last_time = time.time()
    loop_timer = time.time()
    trial_start_time = time.time()
    plot_ff_torque = []
    plot_fb_torque = []
    plot_total_torque = []
    try:
        while not stop_event.is_set():
                current_time = time.time()
                elapsed_time = current_time - trial_start_time
                if elapsed_time > 10:  # Each trial lasts 10 seconds
                    break
                
                dt = current_time - last_time
                last_time = current_time
                desired_angle_deg = sine_position(elapsed_time, speed=SIN_SPEED)
                plot_desired_position.append(desired_angle_deg)
                desired_angle_rad = math.radians(desired_angle_deg)
                desired_velocity_deg = sine_velocity(elapsed_time, speed=SIN_SPEED)
                desired_velocity_rad = math.radians(desired_velocity_deg)
                last_desired_angle = desired_angle_deg
                step = (MOTOR_POS_MIN - MOTOR_POS_MAX)/140
                motor_pos = motor.get_position()[0]
                current_angle_deg = (MOTOR_POS_MIN - motor_pos) / step
                plot_position.append(current_angle_deg)
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity()[0]
                position_error = current_angle_rad - desired_angle_rad
                plot_error.append(math.degrees(position_error))
                velocity_error = desired_velocity_rad - current_velocity

                current_mode = mode_controller.current_mode
                
                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity_rad)

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0
                plot_fb_torque.append(tau_fb)

                if current_mode == ControlMode.AAN:
                    ff_torque = ILC_controller.get_feedforward()
                    plot_ff_torque.append(ff_torque)
                    total_torque = tau_fb + ff_torque
                    plot_total_torque.append(total_torque)
                else:
                    total_torque = tau_fb
                    ff_torque = 0.0
                    plot_ff_torque.append(ff_torque)
                
                torque_clipped = np.clip(total_torque, TORQUE_MIN, TORQUE_MAX)

                current_mode = mode_controller.update_control_mode(torque_clipped, elapsed_time)
                plot_control_mode.append(current_mode)
                plot_torque.append(torque_clipped)

                current = motor.torq2curcom(torque_clipped)
                if motor_pos < MOTOR_POS_MAX and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                elif motor_pos > MOTOR_POS_MIN and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                else:
                    motor.sendMotorCommand(motor.motor_ids[0], current)

                i += 1
    except Exception as e:
        print(f"Exception during final run: {e}")

    # Stop EMG reading thread and EMG sensor
    motor.sendMotorCommand(motor.motor_ids[0], 0)
    print("Shutting down...")
    stop_event.set()
    motor.close()
    # empty all queues
    
    # calculate for plotting
    #Create a time vector for the 167Hz control loop
    time_vector = np.linspace(0, len(plot_position)/SAMPLE_RATE, len(plot_position))

    # Calculate jerk
    plot_jerk = []
    last_acc = 0.0
    dt = 1.0 / SAMPLE_RATE
    plot_jerk.append(0.0)  # Jerk at first point is zero
    for j in range(1, len(plot_position)-1):
        vel_prev = (plot_position[j] - plot_position[j-1]) / dt
        vel_next = (plot_position[j+1] - plot_position[j]) / dt
        acc = (vel_next - vel_prev) / dt
        jerk = (acc - last_acc) / dt  # Assuming previous acceleration is 0 for simplicity
        last_acc = acc
        plot_jerk.append(jerk)
    plot_jerk.append(0.0)  # Jerk at last point is zero
    
    #Plotting
    print("plotting data...")
    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    start = 0
    current_mode = plot_control_mode[0]

    for i in range(1, len(plot_control_mode)):
        # When mode changes, close the previous segment
        if plot_control_mode[i] != current_mode:
            color = 'lightblue' if current_mode == ControlMode.AAN else 'lightcoral'
            plt.axvspan(time_vector[start], time_vector[i], color=color, alpha=0.3)

            # start new segment
            current_mode = plot_control_mode[i]
            start = i

    # Shade the final segment ONCE
    color = 'lightblue' if current_mode == ControlMode.AAN else 'lightcoral'
    plt.axvspan(time_vector[start], time_vector[-1], color=color, alpha=0.8)

    plt.plot(time_vector, plot_position, label='Actual Position (deg)')
    plt.plot(time_vector, plot_desired_position, label='Desired Position (deg)', linestyle='--')
    plt.title('Actual vs Desired Position (Blue = AAN, Red = RAN)')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [deg]')
    plt.xlim(0, 10)
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(time_vector, plot_error, label='Position Error (deg)', color='red')
    plt.title('Position Error Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Error (deg)')
    plt.xlim(0, 10)
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.ylim(-4.5, 4.5)
    plt.plot(time_vector, plot_fb_torque, label='Feedback Torque (Nm)', color='orange')
    plt.plot(time_vector, plot_ff_torque, label='Feedforward Torque (Nm)', color='green')
    plt.plot(time_vector, plot_torque, label='Total Applied Torque (Nm)', color='purple')
    plt.title('Feedback and Feedforward Torque Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque (Nm)')
    plt.xlim(0, 10)
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(time_vector, plot_jerk, label='Jerk (deg/s³)', color='blue')
    plt.title('Jerk Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Jerk (deg/s³)')
    plt.xlim(0, 10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # #plot desired and actual position in one graph and error in another graph
    # plt.figure(figsize=(12, 6))
    # plt.subplot(5, 1, 1)
    # plt.plot(plot_position, label='Actual Position (deg)')
    # plt.plot(plot_desired_position, label='Desired Position (deg)', linestyle='--')
    # plt.title('Actual vs Desired Position')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Position (deg)')
    # plt.legend()
    # plt.grid()
    # plt.subplot(5, 1, 2)
    # plt.plot(plot_error, label='Position Error (deg)', color='red')
    # plt.title('Position Error Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Error (deg)')
    # plt.legend()
    # plt.grid()
    # plt.subplot(5, 1, 3)
    # plt.plot(plot_fb_torque, label='Feedback Torque (Nm)', color='orange')
    # plt.plot(plot_ff_torque, label='Feedforward Torque (Nm)', color='green')
    # plt.title('Feedback and Feedforward Torque Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Torque (Nm)')
    # plt.legend()
    # plt.grid()
    # plt.subplot(5, 1, 4)
    # plt.plot(plot_total_torque, label='Total Applied Torque (Nm)', color='purple')
    # plt.title('Total Applied Torque Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Torque (Nm)')
    # plt.legend()
    # plt.grid()
    # plt.subplot(5, 1, 5)
    # plt.plot(plot_torque, label='Applied Torque (Nm)', color='green')
    # plt.title('Applied Torque Over Time')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Torque (Nm)')
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    print("Goodbye!")
