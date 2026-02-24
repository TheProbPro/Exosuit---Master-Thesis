# local imports
import math

from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
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
import math

# General configuration parameters
SAMPLE_RATE = 2000  # Hz
# FILE_NAME = "Outputs/RecordedEMG/TrainLSTM.csv"
FILE_NAME = "Outputs/RecordedEMG/TestLSTM.csv"
USER_NAME = 'VictorBNielsen'
RECORDING_DURATION = 60  # seconds
ANGLE_MIN = 0
ANGLE_MAX = 140

stop_event = threading.Event()

def read_EMG(raw_queue):
    """EMG读取线程"""
    # Initialize filters
    filter_bicep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(SAMPLE_RATE, 450, 20, 2)
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
        desired_angle_deg = interpreter.compute_angle(activation[0], activation[1])

        try:
            raw_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            raw_queue.get_nowait()
            raw_queue.put_nowait(desired_angle_deg)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

if __name__ == "__main__":
    # Create CSV file and write
    header = ['emg_pos']
    need_header = (not os.path.exists(FILE_NAME)) or os.path.getsize(FILE_NAME) == 0
    if need_header:
        with open(FILE_NAME, mode='w', newline='') as file:
            csv.writer(file).writerow(header)

    write_array = []
    
    recorded_Samples = 0
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=167, lp_cutoff=3, order=2)
    EMG_queue = queue.Queue(maxsize=5)

    emg_thread = threading.Thread(target=read_EMG, args=(EMG_queue,), daemon=True)
    emg_thread.start()
    time.sleep(1.0)

    TIME = time.time()
    while time.time() - TIME < RECORDING_DURATION:
        start_time = time.time()
        try:
            desired_angle_deg = desired_angle_lowpass.lowpass(np.atleast_1d(EMG_queue.get_nowait()))[0]
        except queue.Empty:
            continue
    
        # # Write line to CSV file
        # with open(FILE_NAME, mode='a', newline='', buffering=1) as file:
        #     csv.writer(file).writerow([desired_angle_deg])
        write_array.append(desired_angle_deg)

        recorded_Samples += 1

        processing_time = time.time() - start_time
        sleep_time = max(0, (1.0 / 167) - processing_time)
        time.sleep(sleep_time)

    stop_event.set()
    emg_thread.join(timeout=1.0)

    # Write the array to the .csv file
    with open(FILE_NAME, mode='a', newline='', buffering=1) as file:
        csv.writer(file).writerows([[x] for x in write_array])

    print(f"Recording finished! Recorded {recorded_Samples} samples over {RECORDING_DURATION} seconds.")
    print(f"Data saved to {FILE_NAME}")
    print("EMG stopped!")