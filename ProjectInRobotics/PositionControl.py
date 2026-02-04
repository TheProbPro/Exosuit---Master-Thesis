# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Controllers.PID import PID

# General imports
import numpy as np
import threading
import signal
import time
import matplotlib.pyplot as plt
import queue
import pandas as pd
from pathlib import Path

import traceback


import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

#########################################################
# This script requires motors to be in pos control mode #
#########################################################

# SavePath
# USER_NAME = 'VictorBNielsen'
# USER_NAME = 'ZicehnWang'
USER_NAME = 'Cao'
SAVE_PATH = Path(f"C:/Users/nvigg/Documents/GitHub/Exosuit---Master-Thesis/Outputs/{Path(__file__).stem}/{USER_NAME}/")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# General configuration parameters
SAMPLE_RATE = 166.7  # Hz
EMG_SAMPLE_RATE = 2000  # Hz
ANGLE_MIN = math.radians(0)
ANGLE_MAX = math.radians(140)

MOTOR_POS_MIN = 2280
MOTOR_POS_MAX = 1145

stop_event = threading.Event()

def read_EMG(raw_queue):
    """EMG读取线程"""
    # Initialize filters
    filter_bicep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    emg = DelsysEMG(channel_range=(0,1))
    emg.start()

    time.sleep(1.0)
    
    while not stop_event.is_set():
        reading = emg.read()

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        desired_angle_deg = math.degrees(interpreter.compute_angle(activation[0], activation[1]))

        try:
            raw_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            raw_queue.get_nowait()
            raw_queue.put_nowait(desired_angle_deg)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    plot_position = []
    plot_desired_position = []
    plot_error = []

    # Test
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=SAMPLE_RATE, lp_cutoff=3, order=2)

    EMG_queue = queue.Queue(maxsize=5)

    motor = Motors(port="COM4", baudrate=4500000)

    # Wait a moment before starting
    time.sleep(1.0)
    print("Motor command threads started!")

    emg_thread = threading.Thread(target=read_EMG, args=(EMG_queue,), daemon=True)
    emg_thread.start()
    time.sleep(1.0)

    # PID controller for position control
    controller = PID(Kp=2.0, Ki=0.5, Kd=0.1, output_limits=(0, 140)) 

    # Filter and interpret the raw data
    last_desired_angle = 0
            
    # ====================== Free run ==========================

    print("Press enter to run position control for 10 seconds...")
    input()
    last_time = time.time()
    trial_start_time = time.time()
    previous_position = 70.0
    desired_angle_deg = 70.0  # Start at middle position
    try:
        while not stop_event.is_set():
                current_time = time.time()
                elapsed_time = current_time - trial_start_time
                if elapsed_time > 10:  # Each trial lasts 10 seconds
                    break
                
                dt = current_time - last_time
                last_time = current_time
                
                try:
                    desired_angle_deg = desired_angle_lowpass.lowpass(np.atleast_1d(EMG_queue.get_nowait()))[0]
                except queue.Empty:
                    pass

                plot_desired_position.append(desired_angle_deg)
                desired_angle_rad = math.radians(desired_angle_deg)
                desired_velocity_deg = (desired_angle_deg - previous_position) / dt
                previous_position = desired_angle_deg
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

                # TODO: Use PID controller
                control_output = controller.compute(desired_angle_deg, current_angle_deg, dt)
                control_output = np.clip(control_output, 0.0, 140.0)

                # Convert position to motor steps
                new_motor_pos = MOTOR_POS_MIN - (control_output * step)
                
                # TODO: This needs to just send desired position, not torque!
                motor.sendMotorCommand(motor.motor_ids[0], new_motor_pos)

    except Exception as e:
        print(f"Exception during final run: {e}")

    # Stop EMG reading thread and EMG sensor
    motor.sendMotorCommand(motor.motor_ids[0], 0)
    print("Shutting down...")
    stop_event.set()
    motor.close()
    
    # calculate for plotting
    #Create a time vector for the 167Hz control loop
    time_vector = np.linspace(0, 10, len(plot_position))

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

    # Plotting
    print("plotting data...")
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6), constrained_layout=True)

    # Subplot 1: Actual vs Desired Position with shaded control modes
    axs[0].plot(time_vector, plot_position, label='Actual', linewidth=1.2)
    axs[0].plot(time_vector, plot_desired_position, label='Desired', linestyle='--', linewidth=1.2)
    axs[0].set_title('Actual vs Desired Position')
    axs[0].set_ylabel('Position (deg)')
    axs[0].legend()
    axs[0].grid()

    # Subplot 2: Position Error
    axs[1].plot(time_vector, plot_error, color='red', linewidth=1.2)
    axs[1].set_ylabel('Error (deg)')
    axs[1].grid()

    # Subplot 3: Jerk
    axs[2].plot(time_vector, plot_jerk, linewidth=1.2)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Jerk (deg/s³)')
    axs[2].grid()
    axs[2].set_xlim(0, 10)

    plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    })

    plt.show()

    # TODO: Save data that needs to be plotted later.
    FILE_NAME = f"Final_Trial_Data.csv"
    df = pd.DataFrame({
        'Time_s': time_vector,
        'Actual_Position_deg': plot_position,
        'Desired_Position_deg': plot_desired_position,
        'Position_Error_deg': plot_error,
        'Jerk_deg_per_s3': plot_jerk
    })
    df.to_csv(SAVE_PATH / FILE_NAME, index=False)

    print("Goodbye!")
