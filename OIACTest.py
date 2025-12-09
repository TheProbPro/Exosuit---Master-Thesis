# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from OIAC_Controllers import ada_imp_con

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

    last_time = time.time()
    loop_timer = time.time()
    #while not stop_event.is_set():
    while last_time - loop_timer < 10:  # Run for 10 seconds
        current_time = time.time()
        dt = current_time - last_time

        desired_angle_deg = sine_position(i, speed=0.1)
        desired_angle_rad = math.radians(desired_angle_deg)
        desired_velocity = (desired_angle_deg - last_desired_angle) / dt if dt > 0 else 0.0
        last_desired_angle = desired_angle_deg
        step = 1500/140
        current_velocity = motor.get_velocity()
        motor_pos = motor.get_position()
        current_angle_deg = (2550 - motor_pos) / step
        current_angle_rad = math.radians(current_angle_deg)
        plot_position.append(current_angle_deg)
        plot_error.append(current_angle_deg - desired_angle_deg)
        plot_desired_position.append(desired_angle_deg)

        # print(f"current_angle_deg: {current_angle_deg}, desired_angle_deg: {desired_angle_deg}, error: {current_angle_deg - desired_angle_deg}, current_velocity: {current_velocity}, desired_velocity: {desired_velocity}")

        # OIAC online impedance adaptation
        K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity) #TODO: is this correct?
        # K_ma, B_mat = OIAC.update_impedance(current_angle_deg, desired_angle_deg, current_velocity, desired_velocity)
        tau_fb = OIAC.calc_tau_fb()[0,0] # TODO: This might have to swap sign
        #print(f"tau_fb: {tau_fb}")
        tau_clipped = np.clip(tau_fb, TORQUE_MIN, TORQUE_MAX)
        plot_torque.append(float(tau_clipped))
        
        position_motor = 2550 - int(desired_angle_deg*step)

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