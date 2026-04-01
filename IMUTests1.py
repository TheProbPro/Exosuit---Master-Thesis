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
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd
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

'''
Tests: firstly just EMG no IMU, then test the best performing ones with both EMG and IMU.
'''

# Define global parameters
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

USERNAME = "VictorBNielsen"

n_Sensors = 3

EMG_FS = 1259  # EMG sampling frequency (Hz)
MOTOR_FS = 200  # Motor control frequency (Hz)
IMU_FS = 148  # IMU sampling frequency (Hz) TODO: SET THIS LATER

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

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

        try:
            emg_activation_queue.put_nowait(activation)
        except queue.Full:
            emg_activation_queue.get_nowait()
            emg_activation_queue.put_nowait(activation)

    Tricep_RMS_queue.queue.clear()
    Bicep_RMS_queue.queue.clear()

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)


# Define main
if __name__ == "__main__":
    # Define EMG queues
    emg_activation_queue = queue.Queue(maxsize=5)
    imu_queue = queue.Queue(maxsize=5)

    # Define desired angle lowpass filter
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=EMG_FS, lp_cutoff=3, order=2)
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
    print("Starting Test 1: IMU to position")
    ptime = []
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            imu_data = imu_queue.get_nowait()
            imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

            # Extract accelerometer and gyroscope data for upper and lower arm
            acc_upper = imu_data[0:3]
            gyr_upper = imu_data[3:6]
            acc_lower = imu_data[18:21]
            gyr_lower = imu_data[21:24]

            # Process imu data to get quaternions and elbow angle
            quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
            elbow_angle = imuProcessor.calculate_elbow_angle(quat_upper, quat_lower)

            test1_desired_angles.append(elbow_angle)
            ptime.append(time.time() - last_time)
            last_time = time.time()
        except queue.Empty:
            continue

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()


    # Test 2: IMU processing + optimization 1
    test2_desired_IMU_angles = []
    test2_desired_angles = []
    test2_activations = []
    k = 4.8 * np.pi
    q = 0  # Initial angle (degrees)
    print("Starting Test 2: IMU processing + optimization 1")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            imu_data = imu_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)

        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imuProcessor.calculate_elbow_angle(quat_upper, quat_lower)

        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()
        optimized_angle_emg = optimize_1(k, a, dt, elbow_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_1(k, a, dt, q, THETA_MIN, THETA_MAX)

        test2_desired_IMU_angles.append(optimized_angle_emg)
        test2_desired_angles.append(optimized_angle)
        test2_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 3: IMU processing + optimization 2
    test3_desired_IMU_angles = []
    test3_desired_angles = []
    test3_activations = []
    k = 14 * np.pi
    q = 0  # Initial angle (degrees)
    print("Starting Test 3: EMG processing + optimization 2")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            imu_data = imu_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imuProcessor.calculate_elbow_angle(quat_upper, quat_lower)

        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()
        optimized_angle_IMU = optimize_2(k, a, dt, elbow_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_2(k, a, dt, q, THETA_MIN, THETA_MAX)

        test3_desired_IMU_angles.append(optimized_angle_IMU)
        test3_desired_angles.append(optimized_angle)
        test3_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 4: IMU processing + optimization 4
    test4_desired_IMU_angles = []
    test4_desired_angles = []
    test4_activations = []
    k = 11.5 * np.pi
    q = 0  # Initial angle (degrees)
    delta_q_prev_emg = 0
    delta_q_prev = 0
    print("Starting Test 4: IMU processing + optimization 4")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            imu_data = imu_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imuProcessor.calculate_elbow_angle(quat_upper, quat_lower)

        dt = time.time() - last_time
        last_time = time.time()
        ptime.append(dt)
        optimized_angle_IMU, delta_q_prev_IMU = optimize_4(k, a, dt, elbow_angle, delta_q_prev_IMU, THETA_MIN, THETA_MAX)
        optimized_angle, delta_q_prev = optimize_4(k, a, dt, q, delta_q_prev, THETA_MIN, THETA_MAX)

        test4_desired_IMU_angles.append(optimized_angle_IMU)
        test4_desired_angles.append(optimized_angle)
        test4_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 5: IMU processing + optimization 5
    test5_desired_IMU_angles = []
    test5_desired_angles = []
    test5_activations = []
    k = 2
    b = 0.01
    v = 4 * np.pi
    q = 0  # Initial angle (degrees)

    print("Starting Test 5: IMU processing + optimization 5")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            imu_data = imu_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        
        imu_data = np.asarray(imu_data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = imu_data[0:3]
        gyr_upper = imu_data[3:6]
        acc_lower = imu_data[18:21]
        gyr_lower = imu_data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = imuProcessor.calculate_elbow_angle(quat_upper, quat_lower)

        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()
        optimized_angle_IMU = optimize_5_pd(a, v, dt, elbow_angle, THETA_MIN, THETA_MAX, k, b)
        optimized_angle = optimize_5_pd(a, v, dt, q, THETA_MIN, THETA_MAX, k, b)

        test5_desired_IMU_angles.append(optimized_angle_IMU)
        test5_desired_angles.append(optimized_angle)
        test5_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    imu_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Calculate the velocity, acceleration and jerk for each test
    test1_velocities = np.diff(test1_desired_angles) / dt
    test1_accelerations = np.diff(test1_velocities) / dt
    test1_jerks = np.diff(test1_accelerations) / dt

    test2_velocities = np.diff(test2_desired_angles) / dt
    test2_accelerations = np.diff(test2_velocities) / dt
    test2_jerks = np.diff(test2_accelerations) / dt

    test2_IMU_velocities = np.diff(test2_desired_IMU_angles) / dt
    test2_IMU_accelerations = np.diff(test2_IMU_velocities) / dt
    test2_IMU_jerks = np.diff(test2_IMU_accelerations) / dt

    test3_velocities = np.diff(test3_desired_angles) / dt
    test3_accelerations = np.diff(test3_velocities) / dt
    test3_jerks = np.diff(test3_accelerations) / dt

    test3_IMU_velocities = np.diff(test3_desired_IMU_angles) / dt
    test3_IMU_accelerations = np.diff(test3_IMU_velocities) / dt
    test3_IMU_jerks = np.diff(test3_IMU_accelerations) / dt

    test4_velocities = np.diff(test4_desired_angles) / dt
    test4_accelerations = np.diff(test4_velocities) / dt
    test4_jerks = np.diff(test4_accelerations) / dt

    test4_IMU_velocities = np.diff(test4_desired_IMU_angles) / dt
    test4_IMU_accelerations = np.diff(test4_IMU_velocities) / dt
    test4_IMU_jerks = np.diff(test4_IMU_accelerations) / dt

    test5_velocities = np.diff(test5_desired_angles) / dt
    test5_accelerations = np.diff(test5_velocities) / dt
    test5_jerks = np.diff(test5_accelerations) / dt

    test5_IMU_velocities = np.diff(test5_desired_IMU_angles) / dt
    test5_IMU_accelerations = np.diff(test5_IMU_velocities) / dt
    test5_IMU_jerks = np.diff(test5_IMU_accelerations) / dt

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

    # Generate plots for all tests
    plt.figure(figsize=(15, 10))
    plt.title("Test 1: EMG to position")
    plt.subplot(4, 1, 1)
    plt.plot(test1_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(4, 1, 3)
    plt.plot(test1_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(4, 1, 4)
    plt.plot(test1_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(4, 1, 5)
    plt.plot(test1_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

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
    plt.title("Test 2: EMG to position (EMG optimized)")
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
    plt.title("Test 3: EMG to position (EMG optimized)")
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
    plt.title("Test 4: EMG to position (EMG optimized)")
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
    plt.title("Test 5: EMG to position (EMG optimized)")
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