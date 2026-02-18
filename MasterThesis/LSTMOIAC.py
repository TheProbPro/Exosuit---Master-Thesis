# My local imports (EMG sensor, filtering, interpretors, OIAC)
import math
from Motors.DynamixelHardwareInterface import Motors
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Controllers.OIAC_Controllers import ada_imp_con
from AdaptiveEmbodiedControlSystems.LSTM import LSTMModel

# General imports
import numpy as np
import threading
import signal
import time
import matplotlib.pyplot as plt
import queue
import pandas as pd
import torch

import traceback

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# General configuration parameters
SAMPLE_RATE = 135 #166.7  # Hz TODO: This needs to be measured again, since baudrate has been updated to 4.5Mbps
EMG_SAMPLE_RATE = 2000  # Hz
USER_NAME = 'VictorBNielsen'
ANGLE_MIN = math.radians(0)
ANGLE_MAX = math.radians(140)

# TODO: Exosuit motor can apply torques of up to 10.6 Nm, but we limit it temporarely for safety
TORQUE_MAX = 8.2
TORQUE_MIN = -TORQUE_MAX

stop_event = threading.Event()

def read_EMG(raw_queue):
    """EMG读取线程"""
    # Initialize filters
    filter_bicep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    LSTM_queue = queue.Queue(maxsize=25)

    emg = DelsysEMG(channel_range=(0,1))
    emg.start()

    time.sleep(1.0)

    # Initialize the model
    Model_Save_Path = "Outputs/models/LSTM/Windowed_LSTM.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=1, hidden_size=64, output_size=1, num_layers=1, batch_first=True)
    model.load_state_dict(torch.load(Model_Save_Path, map_location=torch.device(device)))
    model.eval()
    
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

        # Predict the desired angle using the LSTM model to ensure AAN control mode
        try:
            LSTM_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            LSTM_queue.get_nowait()
            LSTM_queue.put_nowait(desired_angle_deg)

        pred_angle = None
        if LSTM_queue.full():
            x = torch.tensor(np.array(LSTM_queue.queue(), dtype=np.float32)).to(device).view(1, 25, 1)  # Shape: (1, seq_len, 1)
            pred = model(x)
            pred_angle = pred.squeeze().item()

        if pred_angle is not None:
            try:
                raw_queue.put_nowait(pred_angle)
            except queue.Full:
                raw_queue.get_nowait()
                raw_queue.put_nowait(pred_angle)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    LSTM_queue.queue.clear()

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

    # Load MOTOR_POS min and max from calibration file if available
    df = pd.read_csv(f'Calib/Users/{USER_NAME}/motor_calib_data.csv')
    MOTOR_POS_MIN = df['Extended'][0]
    MOTOR_POS_MAX = df['Flexed'][0]
    print(f"Loaded motor calibration data: MOTOR_POS_MIN={MOTOR_POS_MIN}, MOTOR_POS_MAX={MOTOR_POS_MAX}")

    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=SAMPLE_RATE, lp_cutoff=3, order=2)

    EMG_queue = queue.Queue(maxsize=5)

    motor = Motors(port="COM3", baudrate=4500000)

    # Wait a moment before starting
    time.sleep(1.0)

    emg_thread = threading.Thread(target=read_EMG, args=(EMG_queue,), daemon=True)
    emg_thread.start()
    time.sleep(1.0)

    # Filter and interpret the raw data
    joint_torque = 0.0
    last_desired_angle = 0
    i = 0
    OIAC = ada_imp_con(1) # 1 degree of freedom

    # Run trial
    all_trial_stats = []
    trial_num = 10

    # ====================== Free run ==========================

    print("Press enter to run final trial with learned feedforward")
    input()
    i = 0
    last_time = time.time()
    trial_start_time = time.time()
    plot_ff_torque = []
    plot_fb_torque = []
    plot_total_torque = []
    previous_position = 70.0
    desired_angle_deg = 70.0  # Start at middle position
    desired_angle_lowpass.reset()
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
                step = (MOTOR_POS_MAX - MOTOR_POS_MIN)/140
                motor_pos = motor.get_position(motor_id=motor.motor_ids[0])
                current_angle_deg = (motor_pos - MOTOR_POS_MIN) / step
                plot_position.append(current_angle_deg)
                current_angle_rad = math.radians(current_angle_deg)
                current_velocity = motor.get_velocity(motor_id=motor.motor_ids[0])
                position_error = current_angle_rad - desired_angle_rad
                plot_error.append(math.degrees(position_error))
                velocity_error = desired_velocity_rad - current_velocity

                K_mat, B_mat = OIAC.update_impedance(current_angle_rad, desired_angle_rad, current_velocity, desired_velocity_rad)

                tau_fb = OIAC.calc_tau_fb()[0,0]
                total_torque = 0.0
                plot_fb_torque.append(tau_fb)
                
                torque_clipped = -np.clip(tau_fb, TORQUE_MIN, TORQUE_MAX)
                plot_torque.append(torque_clipped)

                current = motor.torq2curcom(torque_clipped)
                if motor_pos > MOTOR_POS_MAX and torque_clipped > 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                    print("Motor at MAX position, stopping positive torque: {}".format(torque_clipped))
                elif motor_pos < MOTOR_POS_MIN and torque_clipped < 0:
                    motor.sendMotorCommand(motor.motor_ids[0], 0)
                    print("Motor at MIN position, stopping negative torque: {}".format(torque_clipped))
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
    
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 6), constrained_layout=True)

    # Subplot 1: Actual vs Desired Position with shaded control modes
    start = 0
    current_mode = plot_control_mode[0]

    axs[0].plot(time_vector, plot_position, label='Actual', linewidth=1.2)
    axs[0].plot(time_vector, plot_desired_position, label='Desired', linestyle='--', linewidth=1.2)
    axs[0].set_title('Actual vs Desired Position (Blue = AAN, Red = RAN)')
    axs[0].set_ylabel('Position (deg)')
    axs[0].legend()
    axs[0].grid()

    # Subplot 2: Position Error
    axs[1].plot(time_vector, plot_error, color='red', linewidth=1.2)
    axs[1].set_ylabel('Error (deg)')
    axs[1].grid()

    # Subplot 3: Feedback and Feedforward Torque
    axs[2].plot(time_vector, plot_fb_torque, label='Feedback', linewidth=1.2)
    axs[2].plot(time_vector, plot_ff_torque, label='Feedforward', linewidth=1.2)
    axs[2].plot(time_vector, plot_torque, label='Total', linewidth=1.4)
    axs[2].set_ylabel('Torque (Nm)')
    axs[2].set_ylim(-4.5, 4.5)
    axs[2].legend()
    axs[2].grid()

    # Subplot 4: Jerk
    axs[3].plot(time_vector, plot_jerk, linewidth=1.2)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Jerk (deg/s³)')
    axs[3].grid()
    axs[3].set_xlim(0, 10)

    plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    })

    plt.show()

    print("Goodbye!")
