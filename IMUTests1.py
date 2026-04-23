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

SAVE_PATH = "Outputs/IMUTests1"

stop_event = threading.Event()

def Read_IMU(Sensor, output_queue):
    while not stop_event.is_set():
        data = Sensor.read_imu()
        try:
            output_queue.put_nowait(data)
        except queue.Full:
            output_queue.get_nowait()  # discard oldest
            output_queue.put_nowait(data)

def Read_EMG(Sensor, emg_activation_queue):
    # Initialize filters
    filter_bicep = rt_filtering(EMG_FS, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_FS, 450, 20, 2)
    a_net_lowpass = rt_desired_Angle_lowpass(sample_rate=EMG_FS, lp_cutoff=2, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USERNAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    while not stop_event.is_set():
        reading = Sensor.read_emg()

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
            emg_activation_queue.put_nowait(filtered_net_a)
        except queue.Full:
            emg_activation_queue.get_nowait()
            emg_activation_queue.put_nowait(filtered_net_a)

    Tricep_RMS_queue.queue.clear()
    Bicep_RMS_queue.queue.clear()

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)


# Define main
if __name__ == "__main__":
    # Define EMG queues
    emg_activation_queue = queue.Queue(maxsize=3)
    imu_queue = queue.Queue(maxsize=3)

    # Define desired angle lowpass filter
    lowpass = rt_desired_Angle_lowpass(IMU_FS)
    imu_lowpass = rt_desired_Angle_lowpass(IMU_FS, lp_cutoff=5)
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
    input()
    imu_data_for_bias = []
    start = time.time()
    while time.time() - start < 1.0:
        try:
            imu_data = imu_queue.get_nowait()
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

    # Test 1: Pure IMU processing
    test1_desired_angles = []
    test1_activations = []
    print("Starting Test 1: IMU to position")
    ptime = []
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue
        
        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        test1_activations.append(a)
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))

        # test1_desired_angles.append(elbow_angle)
        test1_desired_angles.append(imu_lowpass.lowpass(np.atleast_1d(elbow_angle))[0])
        t = time.time()
        ptime.append(t - last_time)
        last_time = t

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    # emg_activation_queue.queue.clear()  # Clear the queue after the test
    # imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()
    imu_lowpass.reset()  # Reset the lowpass filter state after the test


    # Test 2: IMU processing + optimization 1
    test2_desired_IMU_angles = []
    test2_desired_angles = []
    test2_activations = []
    k = 3 * np.pi
    q = 0  # Initial angle (degrees)
    test2_desired_angles.append(q)
    print("Starting Test 2: IMU processing + optimization 1")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue
        
        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]

        dt = 1/IMU_FS
        ptime.append(dt)
        last_time = t
        optimized_angle_imu = optimize_1(k, a, dt, elbow_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_1(k, a, dt, test2_desired_angles[-1], THETA_MIN, THETA_MAX)

        test2_desired_IMU_angles.append(optimized_angle_imu)
        test2_desired_angles.append(optimized_angle)
        test2_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    # emg_activation_queue.queue.clear()  # Clear the queue after the test
    # imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()
    imu_lowpass.reset()  # Reset the lowpass filter state after the test
    lowpass.reset()  # Reset the lowpass filter state after the test

    # remove the first element of the desired angles list to align with the rest of the tests
    test2_desired_angles.pop(0)

    # Test 3: IMU processing + optimization 2
    test3_desired_IMU_angles = []
    test3_desired_angles = []
    test3_activations = []
    k = 4 * np.pi
    q = 0  # Initial angle (degrees)
    test3_desired_angles.append(q)
    print("Starting Test 3: IMU processing + optimization 2")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue
        
        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]

        dt = 1/IMU_FS
        ptime.append(dt)
        last_time = t
        optimized_angle_IMU = optimize_2(k, a, dt, elbow_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_2(k, a, dt, test3_desired_angles[-1], THETA_MIN, THETA_MAX)

        test3_desired_IMU_angles.append(optimized_angle_IMU)
        test3_desired_angles.append(optimized_angle)
        test3_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    # emg_activation_queue.queue.clear()  # Clear the queue after the test
    # imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()
    imu_lowpass.reset()  # Reset the lowpass filter state after the test
    lowpass.reset()  # Reset the lowpass filter state after the test

    # remove the first element of the desired angles list to align with the rest of the tests
    test3_desired_angles.pop(0)

    # Test 4: IMU processing + optimization 4
    test4_desired_IMU_angles = []
    test4_desired_angles = []
    test4_activations = []
    k = 2 * np.pi
    q = 0  # Initial angle (degrees)
    test4_desired_angles.append(q)
    delta_q_prev_IMU = 0
    delta_q_prev = 0
    print("Starting Test 4: IMU processing + optimization 4")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue
        
        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]

        dt = 1/IMU_FS
        last_time = t
        ptime.append(dt)
        optimized_angle_IMU, delta_q_prev_IMU = optimize_4(k, a, dt, elbow_angle, delta_q_prev_IMU, THETA_MIN, THETA_MAX)
        optimized_angle, delta_q_prev = optimize_4(k, a, dt, test4_desired_angles[-1], delta_q_prev, THETA_MIN, THETA_MAX)

        test4_desired_IMU_angles.append(optimized_angle_IMU)
        test4_desired_angles.append(optimized_angle)
        test4_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    # emg_activation_queue.queue.clear()  # Clear the queue after the test
    # imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()
    imu_lowpass.reset()  # Reset the lowpass filter state after the test
    lowpass.reset()  # Reset the lowpass filter state after the test

    test4_desired_angles.pop(0)

    # Test 5: IMU processing + optimization 5
    test5_desired_IMU_angles = []
    test5_desired_angles = []
    test5_activations = []
    k = 4
    # b = 0.01
    b = 0.01
    v = 0
    v_imu = 0
    q = 0  # Initial angle (degrees)
    test5_desired_angles.append(q)
    print("Starting Test 5: IMU processing + optimization 5")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue
        
        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]

        dt = 1/IMU_FS
        ptime.append(dt)
        last_time = t
        optimized_angle_IMU, v_imu = optimize_5_pd(a, v_imu, dt, elbow_angle, THETA_MIN, THETA_MAX, np.pi, k, b)
        optimized_angle, v = optimize_5_pd(a, v, dt, test5_desired_angles[-1], THETA_MIN, THETA_MAX, np.pi, k, b)

        test5_desired_IMU_angles.append(optimized_angle_IMU)
        test5_desired_angles.append(optimized_angle)
        test5_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    # emg_activation_queue.queue.clear()  # Clear the queue after the test
    # imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()
    imu_lowpass.reset()  # Reset the lowpass filter state after the test
    lowpass.reset()  # Reset the lowpass filter state after the test

    # remove the first element of the desired angles list to align with the rest of the tests
    test5_desired_angles.pop(0)

    # Test 6: IMU processing + optimization 5 with IMU optimization
    test6_desired_angles = []
    test6_desired_IMU_angles = []
    test6_activations = []
    test6_diff_activations = []
    test6_omega = []
    a_prev = 0
    q_next = 0
    v = 0
    elbow_angle_prev = 0
    kn = 8
    kd = 2
    kp = 5
    b = 2
    print("Starting Test 6: IMU processing + optimization 5 with IMU optimization")
    start_time = time.time()
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        a_diff = (a - a_prev) # / dt
        a_prev = a
        test6_diff_activations.append(a_diff)
        test6_activations.append(a)
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]
        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]
        omega = (elbow_angle - elbow_angle_prev) / dt
        test6_omega.append(omega)
        test6_desired_IMU_angles.append(elbow_angle)
        elbow_angle_prev = elbow_angle
        
        dt = 1/IMU_FS
        ptime.append(dt)

        q_next, v, acc = EMG_IMU_optimizer(a, a_diff, v, omega, kn, kd, kp, b, q_next, elbow_angle, THETA_MIN, THETA_MAX, np.pi, dt)
        test6_desired_angles.append(q_next)

    lowpass.reset()  # Reset the lowpass filter state after the test
    imu_lowpass.reset()  # Reset the lowpass filter state after the test
    ptime.clear()

    # Test 7 IMU processing + optimization 5 with IMU optimization 2
    test7_desired_angles = []
    test7_desired_IMU_angles = []
    test7_activations = []
    test7_diff_activations = []
    test7_omega = []
    kn = 6
    kd = 2
    q_next = 0
    print("Starting Test 7: IMU processing + optimization 5 with IMU optimization 2")
    start_time = time.time()
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        try:
            imu_data = imu_queue.get_nowait()
            a = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = float(lowpass.lowpass(np.atleast_1d(a))[0])  # Lowpass filter activation and keep scalar type
        a_diff = (a - a_prev) # / dt
        a_prev = a
        test7_diff_activations.append(a_diff)
        test7_activations.append(a)
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]
        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imu_lowpass.lowpass(np.atleast_1d(np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))))[0]
        omega = (elbow_angle - elbow_angle_prev) / dt
        test7_omega.append(omega)
        test7_desired_IMU_angles.append(elbow_angle)
        elbow_angle_prev = elbow_angle
        
        dt = 1/IMU_FS
        ptime.append(dt)

        # q_next, v = EMG_IMU_optimizer_2(a, a_diff, omega, kn, kd, elbow_angle, THETA_MIN, THETA_MAX, np.pi, dt)
        q_next, v = EMG_IMU_optimizer_2(a, a_diff, omega, kn, kd, q_next, THETA_MIN, THETA_MAX, np.pi, dt)
        test7_desired_angles.append(q_next)

    lowpass.reset()  # Reset the lowpass filter state after the test
    imu_lowpass.reset()  # Reset the lowpass filter state after the test
    ptime.clear()

    stop_event.set()  # Signal threads to stop
    imu_thread.join()
    emg_thread.join()

    # Calculate the velocity, acceleration and jerk for each test
    time_vector1 = np.arange(len(test1_desired_angles)) * dt
    test1_velocities = np.gradient(test1_desired_angles, dt)
    test1_accelerations = np.gradient(test1_velocities, dt)
    test1_jerks = np.gradient(test1_accelerations, dt)

    time_vector2 = np.arange(len(test2_desired_angles)) * dt
    test2_velocities = np.gradient(test2_desired_angles, dt)
    test2_accelerations = np.gradient(test2_velocities, dt)
    test2_jerks = np.gradient(test2_accelerations, dt)

    time_vector2_IMU = np.arange(len(test2_desired_IMU_angles)) * dt
    test2_IMU_velocities = np.gradient(test2_desired_IMU_angles, dt)
    test2_IMU_accelerations = np.gradient(test2_IMU_velocities, dt)
    test2_IMU_jerks = np.gradient(test2_IMU_accelerations, dt)

    time_vector3 = np.arange(len(test3_desired_angles)) * dt
    test3_velocities = np.gradient(test3_desired_angles, dt)
    test3_accelerations = np.gradient(test3_velocities, dt)
    test3_jerks = np.gradient(test3_accelerations, dt)

    time_vector3_IMU = np.arange(len(test3_desired_IMU_angles)) * dt
    test3_IMU_velocities = np.gradient(test3_desired_IMU_angles, dt)
    test3_IMU_accelerations = np.gradient(test3_IMU_velocities, dt)
    test3_IMU_jerks = np.gradient(test3_IMU_accelerations, dt)

    time_vector4 = np.arange(len(test4_desired_angles)) * dt
    test4_velocities = np.gradient(test4_desired_angles, dt)
    test4_accelerations = np.gradient(test4_velocities, dt)
    test4_jerks = np.gradient(test4_accelerations, dt)

    time_vector4_IMU = np.arange(len(test4_desired_IMU_angles)) * dt
    test4_IMU_velocities = np.gradient(test4_desired_IMU_angles, dt)
    test4_IMU_accelerations = np.gradient(test4_IMU_velocities, dt)
    test4_IMU_jerks = np.gradient(test4_IMU_accelerations, dt)

    time_vector5 = np.arange(len(test5_desired_angles)) * dt
    test5_velocities = np.gradient(test5_desired_angles, dt)
    test5_accelerations = np.gradient(test5_velocities, dt)
    test5_jerks = np.gradient(test5_accelerations, dt)

    time_vector5_IMU = np.arange(len(test5_desired_IMU_angles)) * dt
    test5_IMU_velocities = np.gradient(test5_desired_IMU_angles, dt)
    test5_IMU_accelerations = np.gradient(test5_IMU_velocities, dt)
    test5_IMU_jerks = np.gradient(test5_IMU_accelerations, dt)

    time_vector6 = np.arange(len(test6_desired_angles)) * dt
    test6_velocities = np.gradient(test6_desired_angles, dt)
    test6_accelerations = np.gradient(test6_velocities, dt)
    test6_jerks = np.gradient(test6_accelerations, dt)

    time_vector7 = np.arange(len(test7_desired_angles)) * dt
    test7_velocities = np.gradient(test7_desired_angles, dt)
    test7_accelerations = np.gradient(test7_velocities, dt)
    test7_jerks = np.gradient(test7_accelerations, dt)

    # Print stats for each test
    print(f"Test 1: EMG to position - jerk mean: {np.mean(np.abs(test1_jerks)):.2f} degrees/s^3, jerk max: {np.max(test1_jerks):.2f} degrees/s^3, jerk min: {np.min(test1_jerks):.2f} degrees/s^3")
    print(f"Test 2: EMG to position - jerk mean: {np.mean(np.abs(test2_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_jerks):.2f} degrees/s^3")
    print(f"Test 2: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test2_IMU_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_IMU_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_IMU_jerks):.2f} degrees/s^3")
    print(f"Test 3: EMG to position - jerk mean: {np.mean(np.abs(test3_jerks)):.2f} degrees/s^3, jerk max: {np.max(test3_jerks):.2f} degrees/s^3, jerk min: {np.min(test3_jerks):.2f} degrees/s^3")
    print(f"Test 3: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test3_IMU_jerks)):.2f} degrees/s^3, jerk max: {np.max(test3_IMU_jerks):.2f} degrees/s^3, jerk min: {np.min(test3_IMU_jerks):.2f} degrees/s^3")
    print(f"Test 4: EMG to position - jerk mean: {np.mean(np.abs(test4_jerks)):.2f} degrees/s^3, jerk max: {np.max(test4_jerks):.2f} degrees/s^3, jerk min: {np.min(test4_jerks):.2f} degrees/s^3")
    print(f"Test 4: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test4_IMU_jerks)):.2f} degrees/s^3, jerk max: {np.max(test4_IMU_jerks):.2f} degrees/s^3, jerk min: {np.min(test4_IMU_jerks):.2f} degrees/s^3")
    print(f"Test 5: EMG to position - jerk mean: {np.mean(np.abs(test5_jerks)):.2f} degrees/s^3, jerk max: {np.max(test5_jerks):.2f} degrees/s^3, jerk min: {np.min(test5_jerks):.2f} degrees/s^3")
    print(f"Test 5: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test5_IMU_jerks)):.2f} degrees/s^3, jerk max: {np.max(test5_IMU_jerks):.2f} degrees/s^3, jerk min: {np.min(test5_IMU_jerks):.2f} degrees/s^3")
    print(f"Test 6: EMG to position - jerk mean: {np.mean(np.abs(test6_jerks)):.2f} degrees/s^3, jerk max: {np.max(test6_jerks):.2f} degrees/s^3, jerk min: {np.min(test6_jerks):.2f} degrees/s^3")
    print(f"Test 7: EMG to position - jerk mean: {np.mean(np.abs(test7_jerks)):.2f} degrees/s^3, jerk max: {np.max(test7_jerks):.2f} degrees/s^3, jerk min: {np.min(test7_jerks):.2f} degrees/s^3")
    
    # Generate plots for all tests
    plt.figure(figsize=(15, 10))
    plt.title("Test 1: IMU to position")
    plt.subplot(5, 1, 1)
    plt.plot(test1_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5,1,2)
    plt.plot(test1_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 3)
    plt.plot(test1_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test1_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test1_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    test1_results_df = pd.DataFrame({
        "Time": time_vector1,
        "Net Activation": test1_activations,
        "Desired Angle": test1_desired_angles,
    })
    test1_results_df.to_csv(SAVE_PATH + "/test1_results.csv", index=False)

    plt.figure(figsize=(15, 10))
    plt.title("Test 2: EMG to position + optimization 1")
    plt.subplot(5,1,1)
    plt.plot(test2_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test2_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test2_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test2_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test2_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("Test 2: IMU to position (optimized)")
    plt.subplot(5,1,1)
    plt.plot(test2_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test2_desired_IMU_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test2_IMU_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test2_IMU_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test2_IMU_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    test2_results_df = pd.DataFrame({
        "Time": time_vector2,
        "Time IMU": time_vector2_IMU,
        "Net Activation": test2_activations,
        "Desired Angle": test2_desired_angles,
        "Desired Angle (IMU optimized)": test2_desired_IMU_angles,
    })
    test2_results_df.to_csv(SAVE_PATH + "/test2_results.csv", index=False)


    plt.figure(figsize=(15, 10))
    plt.title("Test 3: EMG to position + optimization 2")
    plt.subplot(5,1,1)
    plt.plot(test3_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test3_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test3_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test3_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test3_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("Test 3: IMU to position (EMG optimized)")
    plt.subplot(5,1,1)
    plt.plot(test3_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test3_desired_IMU_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test3_IMU_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test3_IMU_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test3_IMU_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    test3_results_df = pd.DataFrame({
        "Time": time_vector3,
        "Time IMU": time_vector3_IMU,
        "Net Activation": test3_activations,
        "Desired Angle": test3_desired_angles,
        "Desired Angle (IMU optimized)": test3_desired_IMU_angles,
    })
    test3_results_df.to_csv(SAVE_PATH + "/test3_results.csv", index=False)

    plt.figure(figsize=(15, 10))
    plt.title("Test 4: EMG to position + optimization 4")
    plt.subplot(5,1,1)
    plt.plot(test4_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test4_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test4_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test4_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test4_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("Test 4: IMU to position (optimized)")
    plt.subplot(5,1,1)
    plt.plot(test4_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test4_desired_IMU_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test4_IMU_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test4_IMU_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test4_IMU_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    test4_results_df = pd.DataFrame({
        "Time": time_vector4,
        "Time IMU": time_vector4_IMU,
        "Net Activation": test4_activations,
        "Desired Angle": test4_desired_angles,
        "Desired Angle (IMU optimized)": test4_desired_IMU_angles,
    })
    test4_results_df.to_csv(SAVE_PATH + "/test4_results.csv", index=False)

    plt.figure(figsize=(15, 10))
    plt.title("Test 5: EMG to position + optimization 5")
    plt.subplot(5,1,1)
    plt.plot(test5_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test5_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test5_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test5_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test5_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("Test 5: IMU to position (optimized)")
    plt.subplot(5,1,1)
    plt.plot(test5_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test5_desired_IMU_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test5_IMU_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test5_IMU_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test5_IMU_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    test5_results_df = pd.DataFrame({
        "Time": time_vector5,
        "Time IMU": time_vector5_IMU,
        "Net Activation": test5_activations,
        "Desired Angle": test5_desired_angles,
        "Desired Angle (IMU optimized)": test5_desired_IMU_angles,
    })
    test5_results_df.to_csv(SAVE_PATH + "/test5_results.csv", index=False)

    plt.figure(figsize=(15, 10))
    plt.title("Test 6: EMG to position + optimization 5 with IMU optimization")
    plt.subplot(3,1,1)
    plt.plot(test6_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(3, 1, 2)
    plt.plot(test6_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(3, 1, 3)
    plt.plot(test6_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.tight_layout()
    plt.show()

    test6_results_df = pd.DataFrame({
        "Time": time_vector6,
        "Net Activation": test6_activations,
        "IMU Angle": test6_desired_IMU_angles,
        "Desired Angle": test6_desired_angles,
        "IMU Velocity": test6_omega,
        "Net Activation (Differens)": test6_diff_activations
    })
    test6_results_df.to_csv(SAVE_PATH + "/test6_results.csv", index=False)
    
    plt.figure(figsize=(15, 10))
    plt.title("Test 7: EMG to position + optimization 5 with IMU optimization 2")
    plt.subplot(3,1,1)
    plt.plot(test7_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(3, 1, 2)
    plt.plot(test7_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(3, 1, 3)
    plt.plot(test7_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.tight_layout()
    plt.show()

    test7_results_df = pd.DataFrame({
        "Time": time_vector7,
        "Desired Angle": test7_desired_angles,
        "Net Activation": test7_activations,
        "IMU Angle": test7_desired_IMU_angles,
        "IMU Velocity": test7_omega,
        "Net Activation (Differens)": test7_diff_activations
    })
    test7_results_df.to_csv(SAVE_PATH + "/test7_results.csv", index=False)