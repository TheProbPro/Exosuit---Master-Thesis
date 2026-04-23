import os

from Sensors.EMGSensor import DelsysEMGIMU, DelsysEMG
from SignalProcessing.IMUProcessing import IMUProcessing
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import queue
import signal
from ahrs.filters import Madgwick
import pandas as pd
import math

stop_event = threading.Event()

FS = 2000

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

USER_NAME = 'VictorBNielsen'
SAVE_PATH = "Outputs/IMU_EMG_MoCap_Test/"

# This script is to test the data acquisition and processing together with mocap for validation. It will require 3 emg sensors one on bicep one on tricep and one on forearm.


# def Read_EMG(raw_queue):
#     # Initialize filters
#     filter_bicep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
#     filter_tricep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
#     net_a_lowpass = rt_desired_Angle_lowpass(EMG_SAMPLE_RATE, lp_cutoff=2, order=2)
#     interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
#     Bicep_RMS_queue = queue.Queue(maxsize=50)
#     Tricep_RMS_queue = queue.Queue(maxsize=50)

#     # Initialize EMG
#     emg = DelsysEMG(channel_range=(0,2), samples_per_read=1, host='localhost', emg_units='mV')
#     emg.start()

#     time.sleep(1.0)
    
#     while not stop_event.is_set():
#         reading = emg.read()
#         timestamp = time.time()

#         filtered_bicep = filter_bicep.bandpass(reading[0])
#         filtered_tricep = filter_tricep.bandpass(reading[1])

#         if Bicep_RMS_queue.full():
#             Bicep_RMS_queue.get()
#         Bicep_RMS_queue.put(filtered_bicep)
#         if Tricep_RMS_queue.full():
#             Tricep_RMS_queue.get()
#         Tricep_RMS_queue.put(filtered_tricep)

#         Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
#         Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

#         filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
#         filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

#         # Compute net activation
#         activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
#         net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
#         filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

#         # Compute desired trajectory using optimizer

#         # desired_angle_deg = math.degrees(interpreter.compute_angle(activation[0], activation[1]))

#         try:
#             raw_queue.put_nowait((filtered_net_a, timestamp))
#         except queue.Full:
#             raw_queue.get_nowait()
#             raw_queue.put_nowait((filtered_net_a, timestamp))

#     Bicep_RMS_queue.queue.clear()
#     Tricep_RMS_queue.queue.clear()

# #TODO: Maybe make some functions that can process the IMU and the EMG data

# def handle_sigint(sig, frame):
#     stop_event.set()
# signal.signal(signal.SIGINT, handle_sigint)

# if __name__ == "__main__":
#     # Create output queues
#     emg_queue = queue.Queue(maxsize=5)

#     # Create EMG-IMU sensor instance and start it
#     t_emg = threading.Thread(target=Read_EMG, args=(emg_queue,))
#     t_emg.start()

    
#     #Initialize lowpass filter for desired angle
#     desired_angle_filter = rt_desired_Angle_lowpass(166.7, lp_cutoff=3, order=2)

#     # 10 second loop to read and process data
#     print("Press enter to start 10 seconds of data acquisition and processing...")
#     input()
#     emg_angle_list = []
#     start_time = time.time()
#     while time.time() - start_time < 10.0:
#         try:
#             emg_data, emg_timestamp = emg_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue

        
#         # Process EMG data to get desired angle
#         emg_angle_list.append((desired_angle_filter.lowpass(np.atleast_1d(emg_data)), emg_timestamp))

#     stop_event.set()
#     t_emg.join()
    
#     # Save the angles and timestamps to .csv for later analysis
#     emg_df = pd.DataFrame(emg_angle_list, columns=['Desired_Angle', 'Timestamp'])

#     if not os.path.exists(SAVE_PATH):
#         os.makedirs(SAVE_PATH)

#     emg_df.to_csv(os.path.join(SAVE_PATH, 'emg_desired_angles.csv'), index=False)

#     # Plot the angles over time for visual inspection
#     # plt.figure(figsize=(12, 6))
#     # plt.subplot(2, 1, 1)
#     # plt.plot(imu_df['Timestamp'], imu_df['Elbow_Angle'], label='IMU Elbow Angle (degrees)')
#     # plt.xlabel('Time (s)')
#     # plt.ylabel('Angle (degrees)')
#     # plt.title('Elbow Angle from IMU Data')
#     # plt.legend()
#     # plt.subplot(2, 1, 2)
#     # plt.plot(emg_df['Timestamp'], emg_df['Desired_Angle'], label='EMG Desired Angle (degrees)', color='orange')
#     # plt.xlabel('Time (s)')
#     # plt.ylabel('Desired Angle (degrees)')
#     # plt.title('Desired Elbow Angle from EMG Data')
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.show()

if __name__ == "__main__":
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=2, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)

    # Initialize queues for EMG data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    # Initialize EMG sensors
    emg = DelsysEMG(channel_range=(0,1))
    emg.start()

    time.sleep(1.0)  # Allow some time for the EMG to start and gather data

    print("EMG initialized")

    # TODO: Add more tests if needed

    q = 0
    v = 0
    v_emg = 0
    timestamps = []
    desired_angles = []
    desired_angles.append(q)
    print("Press Enter to start test 9: EMG to position with optimization 5")
    input()
    start_time = time.time()
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        reading = emg.read()
        timestamp = time.time()

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
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        delta_t = 1/FS

        optimized_angle, v, acc = optimizer_6(filtered_net_a, v, delta_t, desired_angles[-1], THETA_MIN, THETA_MAX, b=10.0, k=np.pi*10.0*2)
        
        desired_angles.append(optimized_angle)
        timestamps.append(timestamp)
        
    # remove the initial angle from the optimized angles lists
    desired_angles.remove(desired_angles[0])
    print(f"Operating frequency: {len(desired_angles) / 10:.2f} Hz")

    # Create a DataFrame and save to CSV
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    emg_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Desired_Angle': desired_angles
    })
    emg_df.to_csv(os.path.join(SAVE_PATH, 'desired_angles.csv'), index=False)    