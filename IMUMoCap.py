'''
The EMG processing tests will consist of both prediction and optimization algorithms.
The prediction algorithms should be tested both before and after the optimization algorithms, and both at the same time.
So this means:
 EMG -> optimization -> prediction
 EMG -> prediction -> optimization
 EMG -> prediction -> optimization -> prediction
'''
# Custom includes
from Sensors.EMGSensor import DelsysEMGIMU
from SignalProcessing.IMUProcessing import IMUProcessing
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_IMU_optimizer, EMG_IMU_optimizer_2
# from AdaptiveEmbodiedControlSystems.ESN import ESN
# from AdaptiveEmbodiedControlSystems.LSTM import LSTM
from ProjectInRobotics.pDMP.pDMP_functions import pDMP, pDMPCoupling1, pDMPOmega

# TODO: add includes
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
import queue
import time
import signal
import math
import pandas as pd
import os

'''
Tests: firstly just EMG no IMU, then test the best performing ones with both EMG and IMU.
'''

# Define global parameters
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'

USERNAME = "VictorBNielsen"

n_Sensors = 3

EMG_FS = 1259  # EMG sampling frequency (Hz)
IMU_FS = 148  # IMU sampling frequency (Hz) TODO: SET THIS LATER

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

SAVE_PATH = "Outputs/IMUMocap3/"

stop_event = threading.Event()

def Read_IMU(Sensor, output_queue):
    while not stop_event.is_set():
        data = Sensor.read_imu()
        timestamp = time.time()
        try:
            output_queue.put_nowait((data, timestamp))
        except queue.Full:
            output_queue.get_nowait()  # discard oldest
            output_queue.put_nowait((data, timestamp))

def Read_EMG(Sensor, emg_activation_queue):
    while not stop_event.is_set():
        reading = Sensor.read_emg()
        timestamp = time.time()
        try:
            emg_activation_queue.put_nowait((reading, timestamp))
        except queue.Full:
            emg_activation_queue.get_nowait()
            emg_activation_queue.put_nowait((reading, timestamp))


# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)


# Define main
if __name__ == "__main__":
    # Define EMG queues
    emg_activation_queue = queue.Queue(maxsize=100)
    imu_queue = queue.Queue(maxsize=100)

    imuProcessor = IMUProcessing()

    # Create and start the EMG thread
    emg_imu = DelsysEMGIMU(emg_channel_range=(0,n_Sensors-1), imu_channel_range=(0,(9*n_Sensors)-1), emg_samples_per_read=1, imu_samples_per_read=1, host='localhost', emg_units='mV')
    emg_imu.start()
    time.sleep(1.0)

    emg_thread = threading.Thread(target=Read_EMG, args=(emg_imu, emg_activation_queue))
    imu_thread = threading.Thread(target=Read_IMU, args=(emg_imu, imu_queue))
    emg_thread.start()
    imu_thread.start()
    time.sleep(1.0)  # Allow some time for the EMG thread to start and gather data

    # Prepare IMU
    print("Press enter to start 1 second of data acquisition for gyroscope bias and angle zeroing calculation...")
    # input()
    imu_data_for_bias = []
    start = time.time()
    while time.time() - start < 1.0:
        try:
            imu_data, timestamp = imu_queue.get_nowait()
            # extract the first 9 and last 9 indexes for the upper and lower arm respectively, and append to the list for bias calculation
            first_sensor = imu_data[:9]
            last_sensor = imu_data[-9:]
            imu_data_for_bias.append(np.concatenate((first_sensor, last_sensor)))
        except queue.Empty:
            continue
    
    imulist = list(imu_data_for_bias)
    gyro_bias_upper, gyro_bias_lower = imuProcessor.calculate_bias(imulist)
    print(f"Gyroscope bias for upper arm: {gyro_bias_upper}")
    print(f"Gyroscope bias for forearm: {gyro_bias_lower}")

    # Zeroing calculations using same data
    zero = imuProcessor.calculate_zeroing(imulist)
    print(f"Zeroing baseline for elbow angle: {zero}")

    imu_queue.queue.clear()
    emg_activation_queue.queue.clear()

    # Test 1: Pure IMU processing
    print("Starting IMU recording")
    emg_data_array = []
    emg_timestamps = []
    imu_data_array = []
    imu_timestamps = []
    start_time = time.time()
    while time.time() - start_time < 20:  # Run the test for 20 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data, timestamp_imu = imu_queue.get_nowait()
            imu_data_array.append(imu_data)
            imu_timestamps.append(timestamp_imu)
        except queue.Empty:
            pass

        try:
            emg_data, timestamp_emg = emg_activation_queue.get_nowait()
            emg_data_array.append(emg_data)
            emg_timestamps.append(timestamp_emg)
        except queue.Empty:
            pass
    
    stop_event.set()  # Signal threads to stop
    imu_thread.join()
    emg_thread.join()

    print(f"len of emg {len(emg_data_array)}, len of imu {len(imu_data_array)}")

    df = pd.DataFrame({
        'timestamp_imu': imu_timestamps,
        'imu_data': imu_data_array
    })
    df2 = pd.DataFrame({
        'timestamp_emg': emg_timestamps,
        'emg_data': emg_data_array
    })
    df3 = pd.DataFrame({
        'IMU_zero': zero,
        'IMU_gyro_bias_upper': gyro_bias_upper,
        'IMU_gyro_bias_lower': gyro_bias_lower,
    })
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    df.to_csv(os.path.join(SAVE_PATH, 'imu_data.csv'), index=False)
    df2.to_csv(os.path.join(SAVE_PATH, 'emg_data.csv'), index=False)
    df3.to_csv(os.path.join(SAVE_PATH, 'imu_calibration_data.csv'), index=False)