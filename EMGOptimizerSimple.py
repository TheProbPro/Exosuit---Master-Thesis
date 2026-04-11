'''
The EMG processing tests will consist of both prediction and optimization algorithms.
The prediction algorithms should be tested both before and after the optimization algorithms, and both at the same time.
So this means:
 EMG -> optimization -> prediction
 EMG -> prediction -> optimization
 EMG -> prediction -> optimization -> prediction
'''
# Custom includes
from EMGTests1 import MOTOR_FS
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6
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

FS = 2000 #Hz

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

SAVE_PATH = "Outputs/ALLEMG"

if __name__ == "__main__":
    print("Initializing EMG's...")
    # Define filters and interpretors for EMG processing
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=2, order=2)
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=3, order=2)
    # desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=FS, lp_cutoff=3, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USERNAME, BicepEMG=True, TricepEMG=True)

    # Initialize queues for EMG data
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    # Initialize EMG sensors
    emg = DelsysEMG(channel_range=(0,1))
    emg.start()
    time.sleep(1.0)  # Allow some time for the EMG to start and gather data

    print("EMG initialized")

    #----------------------------------------------------------------------------------------------------------------------------------
    
    test1_desired_angles = []
    test1_activations = []
    dt = 1/FS
    print("Press Enter to start test 1: EMG to position no optimization")
    input()
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
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
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        test1_activations.append(filtered_net_a)  # Store filtered net activation (bicep - tricep)

        test1_desired_angles.append(interpreter.compute_angle(filtered_net_a))  # Compute desired angle using filtered net activation (bicep - tricep)
        
        last_time = time.time()
    
    print(f"length of test1_desired_angles: {len(test1_desired_angles)}, frequency {(len(test1_desired_angles)/10):.2f} Hz, average processing time {10/len(test1_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test2_desired_angles = []
    test2_activations = []
    test2_t = []
    # k = (1.4 * np.pi) / 3
    k_1 = np.pi
    k_2 = 2 * np.pi
    k_3 = np.pi / 2
    delta_q_prev = 0
    k_4 = 1.4
    b_4 = 0.01
    q = 0  # Initial angle (rad)
    test2_desired_angles.append(q)
    v = 0
    alpha = 0.8
    a_cmd_prev = 0
    print("Press Enter to start test 2: EMG to position with optimization 1")
    input()
    start_time = time.time()
    last_time = start_time
    last_t = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
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

        delta_t = 1/FS
        test2_t.append(delta_t)

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        a_cmd = alpha * a_cmd_prev + (1 - alpha) * filtered_net_a
        diff_a = (a_cmd - a_cmd_prev) # / delta_t
        a_cmd_prev = a_cmd

        # optimized_angle = optimize_1(k_1, a_cmd, delta_t, test2_desired_angles[-1], THETA_MIN, THETA_MAX)
        optimized_angle = optimize_2(k_2, a_cmd, delta_t, test2_desired_angles[-1], THETA_MIN, THETA_MAX)
        # optimized_angle, delta_q_prev = optimize_4(k_3, a_cmd, delta_t, test2_desired_angles[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        # optimized_angle, v = optimize_5_pd(a_cmd, v, delta_t, test2_desired_angles[-1], THETA_MIN, THETA_MAX, k_4, b_4)
        # optimized_angle, v, acc = optimizer_6(filtered_net_a, v, delta_t, test2_desired_angles[-1], THETA_MIN, THETA_MAX, b=2.0)
        # optimized_angle = desired_angle_lowpass.lowpass(np.atleast_1d(optimized_angle))[0]
        test2_desired_angles.append(optimized_angle)
        test2_activations.append(filtered_net_a)

        last_time = time.time()

    # remove the initial angle from the optimized angles lists
    test2_desired_angles.remove(test2_desired_angles[0])

    print(f"length of test2_desired_angles: {len(test2_desired_angles)}, frequency {(len(test2_desired_angles)/10):.2f} Hz, average processing time {10/len(test2_desired_angles)} ms")

    
    # ----------------------------------------------------------------------------------------------------------------------------------

    # Calculate the velocity, acceleration and jerk for the test
    test1_velocities = np.gradient(test1_desired_angles, dt) # np.diff(test1_desired_angles) / dt  # Use np.gradient for better numerical stability
    test1_accelerations = np.gradient(test1_velocities, dt) # np.diff(test1_velocities) / dt
    test1_jerks = np.gradient(test1_accelerations, dt) # np.diff(test1_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test1_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]  # Time vector for velocity (one less than desired angles)
    time_vector_acceleration = time_vector[:-2]  # Time vector for acceleration (one less than velocity)
    time_vector_jerk = time_vector[:-3]  # Time vector for jerk (one less than acceleration)


    # Print stats for the test
    print(f"Test 1: EMG to position - jerk mean: {np.mean(np.abs(test1_jerks)):.2f} degrees/s^3, jerk max: {np.max(test1_jerks):.2f} degrees/s^3, jerk min: {np.min(test1_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.title("Test 1: EMG to position")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test1_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test1_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test1_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test1_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test1_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    # # save test 1 data to csv
    # if not os.path.exists(SAVE_PATH):
    #     os.makedirs(SAVE_PATH)

    # test1_results_df = pd.DataFrame({
    #     "Time": time_vector,
    #     "Net Activation": test1_activations,
    #     "Desired Angle": test1_desired_angles,
    # })
    # test1_results_df.to_csv(SAVE_PATH + "/test1_results.csv", index=False)    

    # Calculate the velocity, acceleration and jerk for the test
    test2_velocities = np.gradient(test2_desired_angles, dt) # np.diff(test2_desired_angles) / dt
    test2_accelerations = np.gradient(test2_velocities, dt) # np.diff(test2_velocities) / dt
    test2_jerks = np.gradient(test2_accelerations, dt) # np.diff(test2_accelerations) / dt
    
    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test2_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 2: EMG to position - jerk mean: {np.mean(np.abs(test2_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_jerks):.2f} degrees/s^3")
    
    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 2: EMG to position with optimization 1")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test2_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test2_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test2_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test2_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test2_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save test 2 data to CSV
    # test2_results_df = pd.DataFrame({
    #     "Time": time_vector,
    #     "Net Activation": test2_activations,
    #     "Optimized Desired Angle": test2_desired_angles,
    #     "Optimized Desired Angle EMG": test2_desired_emg_angles,
    # })
    # test2_results_df.to_csv(SAVE_PATH + "/test2_results.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------------------------
    emg.stop()