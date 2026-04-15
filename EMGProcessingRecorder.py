# local imports
import math

from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer

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

# General configuration parameters
FS = 2000  # Hz
FILE_NAME = "Outputs/RecordedEMG/Processing.csv"
USER_NAME = 'VictorBNielsen'

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

RECORDING_DURATION = 10  # seconds

stop_event = threading.Event()

Raw_Bicep_emg = []
Raw_Tricep_emg = []
Bandpassed_Bicep_emg = []
Bandpassed_Tricep_emg = []
RMS_Bicep_EMG = []
RMS_Tricep_EMG = []
Lowpassed_RMS_Bicep_EMG = []
Lowpassed_RMS_Tricep_EMG = []
Raw_Bicep_Activation = []
Raw_Tricep_Activation = []
Net_Activation = []
Filtered_Net_Activation = []

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

    emg = DelsysEMG(channel_range=(0,1))
    emg.start()
    time.sleep(1.0)  # Allow some time for the EMG to start and gather data

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

        Raw_Bicep_emg.append(reading[0])
        Raw_Tricep_emg.append(reading[1])

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        Bandpassed_Bicep_emg.append(filtered_bicep)
        Bandpassed_Tricep_emg.append(filtered_tricep)

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        RMS_Bicep_EMG.append(Bicep_RMS)
        RMS_Tricep_EMG.append(Tricep_RMS)

        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        Lowpassed_RMS_Bicep_EMG.append(filtered_bicep_rms)
        Lowpassed_RMS_Tricep_EMG.append(filtered_tricep_rms)

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        Raw_Bicep_Activation.append(activation[0])
        Raw_Tricep_Activation.append(activation[1])
        Net_Activation.append(net_a)
        Filtered_Net_Activation.append(filtered_net_a)

        recorded_Samples += 1

    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()        

    

    # Write the array to the .csv file
    # Create CSV file and write
    header = [
        'raw_bicep_emg', 'raw_tricep_emg',
        'bandpassed_bicep', 'bandpassed_tricep',
        'rms_bicep', 'rms_tricep',
        'lowpassed_rms_bicep', 'lowpassed_rms_tricep',
        'bicep_activation', 'tricep_activation',
        'net_activation', 'filtered_net_activation'
    ]

    with open(FILE_NAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(zip(
            Raw_Bicep_emg, Raw_Tricep_emg,
            Bandpassed_Bicep_emg, Bandpassed_Tricep_emg,
            RMS_Bicep_EMG, RMS_Tricep_EMG,
            Lowpassed_RMS_Bicep_EMG, Lowpassed_RMS_Tricep_EMG,
            Raw_Bicep_Activation, Raw_Tricep_Activation,
            Net_Activation, Filtered_Net_Activation
        ))

    print(f"len of write_array: {len(write_array)}, frequency {(len(write_array)/RECORDING_DURATION):.2f} Hz")
    print(f"Recording finished! Recorded {recorded_Samples} samples over {RECORDING_DURATION} seconds.")
    print(f"Data saved to {FILE_NAME}")
    print("EMG stopped!")