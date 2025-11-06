# local imports
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC

# global imports
import numpy as np
import queue
import os
import threading
import sys
import signal
import time
import csv

# General configuration parameters
SAMPLE_RATE = 2000  # Hz
FILE_NAME = "Outputs/RecordedEMG/EMG_Recording_LSTM_Test.csv"
USER_NAME = 'VictorBNielsen'
RECORDING_DURATION = 60  # seconds
ANGLE_MIN = 0
ANGLE_MAX = 140



if __name__ == "__main__":
    # Create CSV file and write
    header = ['Processed EMG', 'Muscle Activation', 'Position']
    need_header = (not os.path.exists(FILE_NAME)) or os.path.getsize(FILE_NAME) == 0
    if need_header:
        with open(FILE_NAME, mode='w', newline='') as file:
            csv.writer(file).writerow(header)
    
    # Create EMG sensor instance
    emg = DelsysEMG()

    # Initialize filtering and interpretors
    filter = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    interpreter = PMC(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=False)
    interpreter.set_Kp(8)

    emg.start()
    print("EMG started!")

    Bicep_RMS_queue = queue.Queue(maxsize=50)
    recorded_Samples = 0
    TIME = time.time()
    while time.time() - TIME < RECORDING_DURATION:
        # Read EMG data
        reading = emg.read()
        # Filter data
        filtered_reading = filter.bandpass(reading[0])  # Bicep channel

        # Calculate RMS
        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_reading.item())
        rms_value = 0.0
        if not Bicep_RMS_queue.full():
            rms_value = 0.0
        else:
            rms_value = filter.RMS(list(Bicep_RMS_queue.queue))

        # Rectify RMS signal with 3 Hz low-pass filter
        filtered_RMS = filter.lowpass(np.atleast_1d(rms_value)).item()

        if recorded_Samples < 200:
            print(f"Sample {recorded_Samples}: rms_value = {rms_value}, filtered_RMS = {filtered_RMS}")

        # Compute activation and position
        activation = interpreter.compute_activation(filtered_RMS)
        position = interpreter.compute_angle(activation[0], activation[1])
        
        # Write line to CSV file
        with open(FILE_NAME, mode='a', newline='', buffering=1) as file:
            csv.writer(file).writerow([filtered_RMS, activation[0], position])

        recorded_Samples += 1

    emg.stop()
    print(f"Recording finished! Recorded {recorded_Samples} samples over {RECORDING_DURATION} seconds.")
    print(f"Data saved to {FILE_NAME}")
    print("EMG stopped!")

    # Plot data saved in the .CSV file
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(FILE_NAME)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df['Processed EMG'], label='Processed EMG')
    plt.title('Processed EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(df['Muscle Activation'], label='Muscle Activation', color='orange')
    plt.title('Muscle Activation')
    plt.xlabel('Samples')
    plt.ylabel('Activation Level')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(df['Position'], label='Computed Position', color='green')
    plt.title('Computed Joint Position from EMG Signal')
    plt.xlabel('Samples')
    plt.ylabel('Position (degrees)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Plotting finished!")