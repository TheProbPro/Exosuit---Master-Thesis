# local imports
import math

from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer
from ProjectInRobotics.pDMP.pDMP_functions import pDMP

# global imports
import numpy as np
import queue
import os
import threading
import sys
import signal
import time
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt

# General configuration parameters
FS = 2000  # Hz
# FILE_NAME = "Outputs/RecordedEMG/TrainLSTM.csv"
FILE_NAME = "Outputs/RecordedEMG/TestLSTM.csv"
USER_NAME = 'VictorBNielsen'

# FILE_NAME = f"Outputs/RecordedEMG/EMGData.csv"

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

# RECORDING_DURATION = 90  # seconds
RECORDING_DURATION = 60  # seconds
# RECORDING_DURATION = 20  # seconds, for testing
stop_event = threading.Event()

# def read_EMG(raw_queue):
#     # Initialize filters
#     filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
#     filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
#     interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
#     Bicep_RMS_queue = queue.Queue(maxsize=50)
#     Tricep_RMS_queue = queue.Queue(maxsize=50)

#     emg = DelsysEMG(channel_range=(0,1))
#     emg.start()

#     time.sleep(1.0)
    
#     while not stop_event.is_set():
#         reading = emg.read()

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

#         activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
#         desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])

#         try:
#             raw_queue.put_nowait(desired_angle_deg)
#         except queue.Full:
#             raw_queue.get_nowait()
#             raw_queue.put_nowait(desired_angle_deg)
        
#     emg.stop()
#     Bicep_RMS_queue.queue.clear()
#     Tricep_RMS_queue.queue.clear()

raw_bicep_emg = []
raw_tricep_emg = []
processed_bicep_emg = []
processed_tricep_emg = []
bicep_activation = []
tricep_activation = []
net_activation = []
filtered_net_activation = []

if __name__ == "__main__":
    write_array = []
    
    # Initialize EMG logic
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=2, order=2)

    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)

    # Initialize queues for EMG data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    dt = 1/FS

    phi = 0
    tau = 0.5
    DMP = pDMP(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    DMP.set_output_limits(THETA_MIN, THETA_MAX, squash_gain=1.0)
    DMP.set_output_state(np.array([0.0]))
    y_old = 0
    dy_old = 0
    start_time = time.time()
    while time.time() - start_time < 3:  # Teach for 3 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        phi += 2*np.pi * dt/tau #16*np.pi * dt/tau
        y = np.array([0])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))
        DMP.learn(y, dy, ddy)
        DMP.integration()

        # old values	
        y_old = y
        dy_old = dy
            
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

    emg = DelsysEMG(channel_range=(0,1))
    emg.start()
    # time.sleep(1.0)  # Allow some time for the EMG to start and gather data

    print("EMG initialized")

    # ==========================================================================================================================

    recorded_Samples = 0

    q = 0
    v = 0
    v_emg = 0
    desired_angles = []
    desired_angles.append(q)
    print("Starting acquisition...")
    TIME = time.time()
    while time.time() - TIME < RECORDING_DURATION:
        print(f"elapsed time: {time.time() - TIME:.2f} seconds", end='\r')
        # Read EMG data
        reading = emg.read()

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        raw_bicep_emg.append(reading[0])
        raw_tricep_emg.append(reading[1])

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

        processed_bicep_emg.append(filtered_bicep_rms)
        processed_tricep_emg.append(filtered_tricep_rms)

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        bicep_activation.append(activation[0])
        tricep_activation.append(activation[1])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        net_activation.append(net_a)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        filtered_net_activation.append(filtered_net_a)

        # TODO: For LSTM teaching data
        # optimized_angle, v, acc = optimizer_6(filtered_net_a, v, dt, desired_angles[-1], THETA_MIN, THETA_MAX, b=10.0, k=np.pi*10.0*2)
        # optimized_angle = optimize_2(np.pi*0.9, filtered_net_a, dt, desired_angles[-1], THETA_MIN, THETA_MAX)

        v = np.pi/50 #np.pi/22
        # v = np.pi/2
        for a in activation:
            DMP.set_phase(np.array([phi]))
            DMP.set_period(np.array([tau]))

            U = np.asarray([a*v])  # EMG activation as input
            DMP.update(U)
            DMP.integration()
            x, dx, ph, ta = DMP.get_state()
            optimized_angle = x[0]

        desired_angles.append(optimized_angle)
        write_array.append(optimized_angle)

        recorded_Samples += 1

    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()        

    #create a time vector spanning from 0 to recording duration with the same length as the EMG data
    time_vector = np.linspace(0, RECORDING_DURATION, len(raw_bicep_emg))

    plt.figure(figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.plot(time_vector, write_array)
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (rad)")
    plt.title("Recorded EMG Data")
    plt.subplot(2,1,2)
    plt.plot(time_vector, filtered_net_activation, label='Net Activation')
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.title("Filtered Net Activation")
    plt.legend()
    plt.show()

    # df = pd.DataFrame({
    #     "time": time_vector,
    #     'raw_bicep_emg': raw_bicep_emg,
    #     'raw_tricep_emg': raw_tricep_emg,
    #     'processed_bicep_emg': processed_bicep_emg,
    #     'processed_tricep_emg': processed_tricep_emg,
    #     'bicep_activation': bicep_activation,
    #     'tricep_activation': tricep_activation,
    #     'net_activation': net_activation,
    #     'filtered_net_activation': filtered_net_activation
    # })
    # if not os.path.exists("Outputs/RecordedEMG"):
    #     os.makedirs("Outputs/RecordedEMG")
    # df.to_csv(FILE_NAME, index=False)

    # For LSTM teaching data
    # Write the array to the .csv file
    # Create CSV file and write
    header = ['emg_pos']
    with open(FILE_NAME, mode='w', newline='', buffering=1) as file:
        csv.writer(file).writerow(header)
        csv.writer(file).writerows([[x] for x in write_array])

    print(f"len of write_array: {len(write_array)}, frequency {(len(write_array)/RECORDING_DURATION):.2f} Hz")
    print(f"Recording finished! Recorded {recorded_Samples} samples over {RECORDING_DURATION} seconds.")
    print(f"Data saved to {FILE_NAME}")
    print("EMG stopped!")