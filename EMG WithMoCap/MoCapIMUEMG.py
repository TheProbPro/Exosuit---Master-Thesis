import os

from Sensors.EMGSensor import DelsysEMGIMU, DelsysEMG
from SignalProcessing.IMUProcessing import IMUProcessing
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, EMG_IMU_optimizer, EMG_IMU_optimizer_2
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
n_Sensors = 3

IMU_FS = 148
EMG_FS = 1259

THETA_MIN = math.radians(0)
THETA_MAX = math.radians(140)

USER_NAME = 'VictorBNielsen'
SAVE_PATH = "Outputs/IMU_EMG_MoCap_Test/"

# This script is to test the data acquisition and processing together with mocap for validation. It will require 3 emg sensors one on bicep one on tricep and one on forearm.

def Read_IMU(Sensor, output_queue):
    while not stop_event.is_set():
        data = Sensor.read_imu()
        timestamp = time.time()
        try:
            output_queue.put_nowait((data, timestamp))
        except queue.Full:
            output_queue.get_nowait()
            output_queue.put_nowait((data, timestamp))


def Read_EMG(Sensor, raw_queue):
    # Initialize filters
    filter_bicep = rt_filtering(EMG_FS, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_FS, 450, 20, 2)
    a_net_lowpass = rt_desired_Angle_lowpass(sample_rate=EMG_FS, lp_cutoff=2, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)
    
    while not stop_event.is_set():
        reading = Sensor.read_emg()
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
        filtered_net_a = float(a_net_lowpass.lowpass(np.atleast_1d(net_a))[0])  # Lowpass filter the net activation

        try:
            raw_queue.put_nowait((filtered_net_a, timestamp))
        except queue.Full:
            raw_queue.get_nowait()
            raw_queue.put_nowait((filtered_net_a, timestamp))

    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

#TODO: Maybe make some functions that can process the IMU and the EMG data

def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    # Create output queues
    imu_queue = queue.Queue(maxsize=3)
    emg_queue = queue.Queue(maxsize=3)

    # Define desired angle lowpass filter
    lowpass = rt_desired_Angle_lowpass(IMU_FS)
    imu_lowpass = rt_desired_Angle_lowpass(IMU_FS, lp_cutoff=5)
    imuProcessor = IMUProcessing()

    # Create EMG-IMU sensor instance and start it
    emg_imu = DelsysEMGIMU(emg_channel_range=(0,n_Sensors-1), imu_channel_range=(0,(9*n_Sensors)-1), emg_samples_per_read=1, imu_samples_per_read=1, host='localhost', emg_units='mV')
    emg_imu.start()
    time.sleep(1.0)

    t_emg = threading.Thread(target=Read_EMG, args=(emg_imu, emg_queue))
    t_imu = threading.Thread(target=Read_IMU, args=(emg_imu, imu_queue))
    t_emg.start()
    t_imu.start()
    time.sleep(1.0)  # Allow some time for the threads to start and gather data

    # perform 1 sec of data acquisition to calculate the gyroscope bias
    print("Press enter to start 1 second of data acquisition for gyroscope bias and angle zeroing calculation...")
    input()
    imu_data_for_bias = []
    start = time.time()
    while time.time() - start < 1.0:
        try:
            imu_data, _ = imu_queue.get_nowait()
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


    # TODO: Implement more tests if needed

    # 10 second loop to read and process data
    print("Press enter to start 10 seconds of data acquisition and processing...")
    input()
    imu_angle_list = []
    kn = 6
    kd = 2
    q_next = 0
    start_time = time.time()
    a_prev = 0
    elbow_angle_prev = 0
    while time.time() - start_time < 10.0:
        try:
            imu_data, imu_timestamp = imu_queue.get_nowait()
            emg_data, emg_timestamp = emg_queue.get_nowait()
        except queue.Empty:
            continue

        a = float(lowpass.lowpass(np.atleast_1d(emg_data))[0])
        a_diff = (a - a_prev)
        a_prev = a

        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        dt = 1/IMU_FS
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]
        omega = (elbow_angle - elbow_angle_prev) / dt
        elbow_angle_prev = elbow_angle

        q_next, v = EMG_IMU_optimizer_2(a, a_diff, omega, kn, kd, q_next, THETA_MIN, THETA_MAX, np.pi, dt)

        # Save imu elbow angle and timestamp for later analysis
        imu_angle_list.append((elbow_angle, imu_timestamp))

        # Process EMG data to get desired angle
        imu_angle_list.append((q_next, imu_timestamp, emg_timestamp))

    stop_event.set()
    t_emg.join()
    t_imu.join()
    emg_imu.stop()
    
    # Save the angles and timestamps to .csv for later analysis
    imu_df = pd.DataFrame(imu_angle_list, columns=['Elbow_Angle', 'IMU_Timestamp', 'EMG_Timestamp'])

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    imu_df.to_csv(os.path.join(SAVE_PATH, 'imu_angles.csv'), index=False)

    # Plot the angles over time for visual inspection
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(imu_df['Timestamp'], imu_df['Elbow_Angle'], label='IMU Elbow Angle (degrees)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (degrees)')
    # plt.title('Elbow Angle from IMU Data')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(emg_df['Timestamp'], emg_df['Desired_Angle'], label='EMG Desired Angle (degrees)', color='orange')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Desired Angle (degrees)')
    # plt.title('Desired Elbow Angle from EMG Data')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()