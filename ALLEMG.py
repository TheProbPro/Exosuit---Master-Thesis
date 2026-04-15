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
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer
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

SAVE_CSV = False

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

    test2_desired_emg_angles = []
    test2_desired_angles = []
    test2_activations = []
    test2_t = []
    # k = 4.8 * np.pi
    k = 3 * np.pi
    q = 0  # Initial angle (rad)
    test2_desired_angles.append(q)
    print("Press Enter to start test 2: EMG to position with optimization 1")
    input()
    start_time = time.time()
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        desired_angle = float(interpreter.compute_angle(filtered_net_a))  # Compute desired angle using filtered net activation (bicep - tricep)

        delta_t = 1/FS
        test2_t.append(delta_t)

        optimized_angle_emg = optimize_1(k, filtered_net_a, delta_t, desired_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_1(k, filtered_net_a, delta_t, test2_desired_angles[-1], THETA_MIN, THETA_MAX)
        test2_desired_emg_angles.append(optimized_angle_emg)
        test2_desired_angles.append(optimized_angle)
        test2_activations.append(filtered_net_a)

    # remove the initial angle from the optimized angles lists
    test2_desired_angles.remove(test2_desired_angles[0])

    print(f"length of test2_desired_angles: {len(test2_desired_angles)}, frequency {(len(test2_desired_angles)/10):.2f} Hz, average processing time {10/len(test2_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------
        
    test3_desired_emg_angles = []
    test3_desired_angles = []
    test3_activations = []
    test3_t = []
    # k = 14*np.pi #18 * np.pi
    k = 4 * np.pi
    q = 0  # Initial angle (degrees)
    test3_desired_angles.append(q)
    print("Press Enter to start test 3: EMG to position with optimization 2")
    input()
    start_time = time.time()
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        desired_angle = float(interpreter.compute_angle(filtered_net_a))

        delta_t = 1/FS
        test3_t.append(delta_t)

        optimized_angle_emg = optimize_2(k, filtered_net_a, delta_t, desired_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_2(k, filtered_net_a, delta_t, test3_desired_angles[-1], THETA_MIN, THETA_MAX)
        test3_desired_emg_angles.append(optimized_angle_emg)
        test3_desired_angles.append(optimized_angle)
        test3_activations.append(filtered_net_a)

    # remove the initial angle from the optimized angles lists
    test3_desired_angles.remove(test3_desired_angles[0])

    print(f"length of test3_desired_angles: {len(test3_desired_angles)}, frequency {(len(test3_desired_angles)/10):.2f} Hz, average processing time {10/len(test3_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test4_desired_emg_angles = []
    test4_desired_angles = []
    test4_activations = []
    test4_t = []
    # k = 11.5 * np.pi
    k = 2 * np.pi
    q = 0  # Initial angle (degrees)
    test4_desired_angles.append(q)
    delta_q_prev_emg = 0
    delta_q_prev = 0
    print("Press Enter to start test 4: EMG to position with optimization 4")
    input()
    start_time = time.time()
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        desired_angle = float(interpreter.compute_angle(filtered_net_a))

        delta_t = 1/FS
        test4_t.append(delta_t)

        optimized_angle_emg, delta_q_prev_emg = optimize_4(k, filtered_net_a, delta_t, desired_angle, delta_q_prev_emg, THETA_MIN, THETA_MAX)
        optimized_angle, delta_q_prev = optimize_4(k, filtered_net_a, delta_t, test4_desired_angles[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        test4_desired_emg_angles.append(optimized_angle_emg)
        test4_desired_angles.append(optimized_angle)
        test4_activations.append(filtered_net_a)

    # remove the initial angle from the optimized angles lists
    test4_desired_angles.remove(test4_desired_angles[0])

    print(f"length of test4_desired_angles: {len(test4_desired_angles)}, frequency {(len(test4_desired_angles)/10):.2f} Hz, average processing time {10/len(test4_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test5_desired_emg_angles = []
    test5_desired_angles = []
    test5_activations = []
    test5_t = []
    k = 4
    b = 0.1
    # v = 4 * np.pi
    v = 0
    v_emg = 0
    q = 0  # Initial angle (degrees)
    test5_desired_angles.append(q)
    print("Press Enter to start test 5: EMG to position with optimization 5")
    input()
    start_time = time.time()
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        desired_angle = float(interpreter.compute_angle(filtered_net_a))

        delta_t = 1/FS
        test5_t.append(delta_t)

        optimized_angle_emg, v_emg = optimize_5_pd(filtered_net_a, v_emg, delta_t, desired_angle, THETA_MIN, THETA_MAX, np.pi, k, b)
        optimized_angle, v = optimize_5_pd(filtered_net_a, v, delta_t, test5_desired_angles[-1], THETA_MIN, THETA_MAX, np.pi, k, b)
        test5_desired_emg_angles.append(optimized_angle_emg)
        test5_desired_angles.append(optimized_angle)
        test5_activations.append(filtered_net_a)

    # remove the initial angle from the optimized angles lists
    test5_desired_angles.remove(test5_desired_angles[0])

    print(f"length of test5_desired_angles: {len(test5_desired_angles)}, frequency {(len(test5_desired_angles)/10):.2f} Hz, average processing time {10/len(test5_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test_9_desired_emg_angles = []
    test_9_desired_angles = []
    test_9_activations = []
    test_9_t = []
    v = 0
    v_emg = 0
    test_9_desired_angles.append(q)
    print("Press Enter to start test 9: EMG to position with optimization 5")
    input()
    start_time = time.time()
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        desired_angle = float(interpreter.compute_angle(filtered_net_a))

        delta_t = 1/FS
        test_9_t.append(delta_t)

        optimized_angle_emg, v_emg, acc_emg = optimizer_6(filtered_net_a, v_emg, delta_t, desired_angle, THETA_MIN, THETA_MAX, b=10.0, k =np.pi*10.0*2)
        optimized_angle, v, acc = optimizer_6(filtered_net_a, v, delta_t, test_9_desired_angles[-1], THETA_MIN, THETA_MAX, b=10.0, k =np.pi*10.0*2)
        test_9_desired_emg_angles.append(optimized_angle_emg)
        test_9_desired_angles.append(optimized_angle)
        test_9_activations.append(filtered_net_a)

    # remove the initial angle from the optimized angles lists
    test_9_desired_angles.remove(test_9_desired_angles[0])

    print(f"length of test_9_desired_angles: {len(test_9_desired_angles)}, frequency {(len(test_9_desired_angles)/10):.2f} Hz, average processing time {10/len(test_9_desired_angles)} ms")

    #-----------------------------------------------------------------------------------------------------------------------------------

    test_10_desired_angles_emg = []
    test_10_desired_angles = []
    test_10_activations = []
    test_10_t = []
    kn = 10.0
    kd = 2.0
    b = 2.0
    v = 0
    v_emg = 0
    test_10_desired_angles.append(q)
    a_prev = 0

    print("Press Enter to start test 10: EMG to position with optimization 10")
    input()
    start_time = time.time()
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        
        delta_t = 1/FS
        test_10_t.append(delta_t)
        a_d = (filtered_net_a - a_prev) / delta_t
        a_prev = filtered_net_a
        desired_angle = float(interpreter.compute_angle(filtered_net_a))

        optimized_angle_emg, v_emg, acc_emg = EMG_Optimizer(filtered_net_a, a_d, v_emg, kn, kd, b, desired_angle, THETA_MIN, THETA_MAX, np.pi, delta_t)
        optimized_angle, v, acc = EMG_Optimizer(filtered_net_a, a_d, v, kn, kd, b, test_10_desired_angles[-1], THETA_MIN, THETA_MAX, np.pi, delta_t)
        test_10_desired_angles_emg.append(optimized_angle_emg)
        test_10_desired_angles.append(optimized_angle)
        test_10_activations.append(filtered_net_a)

    # remove the initial angle from the optimized angles lists
    test_10_desired_angles.remove(test_10_desired_angles[0])

    print(f"length of test_10_desired_angles: {len(test_10_desired_angles)}, frequency {(len(test_10_desired_angles)/10):.2f} Hz, average processing time {10/len(test_10_desired_angles)} ms")

    #-----------------------------------------------------------------------------------------------------------------------------------

    dt = 1/FS
    phi = 0
    tau = 0.5
    DMP = pDMP(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    # Teach DMP 0 trajectory for 2s
    y_old = 0
    dy_old = 0
    print("Teaching DMP 0 trajectory for 3s")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 3:  # Teach for 3 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        phi += 16*np.pi * dt/tau
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

        # sleep_time = dt - (time.time() - last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()
    print("DMP teaching completed")

    test6_desired_angles = []
    test6_activations = []
    v = np.pi/10 #np.pi/22
    print("Press Enter to start test 6: EMG processing + pDMP")
    input()
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run for 10 seconds
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

        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        U = np.asarray([filtered_net_a*v])  # EMG activation as input
        DMP.update(U)
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        test6_desired_angles.append(x[0])
        test6_activations.append(filtered_net_a)

        # sleep_time = dt - (time.time() - last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    print(f"length of test6_desired_angles: {len(test6_desired_angles)}, frequency {(len(test6_desired_angles)/10):.2f} Hz, average processing time {10/len(test6_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    dt = 1/FS
    phi = 0
    tau = 0.5
    DMP = pDMPCoupling1(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    # Teach DMP 0 trajectory for 3s
    y_old = 0
    dy_old = 0
    print("Teaching pDMP coupling 1 with 0 trajectory for 2s")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 3:  # Teach for 3 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        phi += 2*np.pi * dt/tau
        y = np.array([0])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))
        DMP.learn(y, dy, ddy)

        # old values	
        y_old = y
        dy_old = dy
        
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

        # sleep_time = dt - (time.time() - last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    print("pDMP coupling 1 teaching completed")
    test7_desired_angles = []
    test7_activations = []
    print("Press Enter to start test 7: EMG processing + pDMP coupling 1")
    input()
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run for 10 seconds
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

        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        DMP.repeat()

        DMP.integration(np.array([filtered_net_a]))

        x, dx, ph, ta = DMP.get_state()
        test7_desired_angles.append(x[0])
        test7_activations.append(filtered_net_a)

        # sleep_time = dt - (time.time() - last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    print(f"length of test7_desired_angles: {len(test7_desired_angles)}, frequency {(len(test7_desired_angles)/10):.2f} Hz, average processing time {10/len(test7_desired_angles)} ms")

    # ---------------------------------------------------------------------------------------------------------------------------------

    dt = 1/FS
    phi = 0
    tau = 5
    omega0 = 2*np.pi/tau
    DMP = pDMPOmega(DOF=1, N=25, alpha=8, beta=2, lambd=0.999, dt=dt)
    DMP.set_frequency([omega0])
    # Teach DMP 0 trajectory for 3s
    y_old = 0
    dy_old = 0
    print("Teaching pDMP omega with 0 trajectory for 5s")
    start_time = time.time()
    last_time = start_time
    samples = (1/dt) * 5
    for i in range(int(samples)):
        t = i * dt
        y = np.array([np.sin(omega0*t)])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt

        DMP.set_frequency(np.array([omega0]))

        DMP.learn(y, dy, ddy)
        DMP.integration()

        # old values	
        y_old = y
        dy_old = dy
        
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()
    

    print("pDMP omega teaching completed")
    test8_desired_angles = []
    test8_activations = []
    k = 1.0
    print("Press Enter to start test 8: EMG processing + pDMP omega")
    input()
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run for 10 seconds
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

        omega = omega0 * (1 + k * filtered_net_a)
        DMP.set_frequency([omega])
        DMP.repeat()
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        test8_desired_angles.append(x[0])
        test8_activations.append(filtered_net_a)

        # sleep_time = dt - (time.time() - last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    print(f"length of test8_desired_angles: {len(test8_desired_angles)}, frequency {(len(test8_desired_angles)/10):.2f} Hz, average processing time {10/len(test8_desired_angles)} ms")    

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

    # save test 1 data to csv
    if SAVE_CSV:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        test1_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test1_activations,
            "Desired Angle": test1_desired_angles,
        })
        test1_results_df.to_csv(SAVE_PATH + "/test1_results.csv", index=False)    

    # Calculate the velocity, acceleration and jerk for the test
    test2_velocities = np.gradient(test2_desired_angles, dt) # np.diff(test2_desired_angles) / dt
    test2_accelerations = np.gradient(test2_velocities, dt) # np.diff(test2_velocities) / dt
    test2_jerks = np.gradient(test2_accelerations, dt) # np.diff(test2_accelerations) / dt
    test2_emg_velocities = np.gradient(test2_desired_emg_angles, dt) # np.diff(test2_desired_emg_angles) / dt
    test2_emg_accelerations = np.gradient(test2_emg_velocities, dt) # np.diff(test2_emg_velocities) / dt
    test2_emg_jerks = np.gradient(test2_emg_accelerations, dt) # np.diff(test2_emg_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test2_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 2: EMG to position - jerk mean: {np.mean(np.abs(test2_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_jerks):.2f} degrees/s^3")
    print(f"Test 2: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test2_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_emg_jerks):.2f} degrees/s^3")

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

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 2: EMG to position with optimization 1")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test2_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test2_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test2_emg_velocities, label="Velocity (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test2_emg_accelerations, label="Acceleration (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test2_emg_jerks, label="Jerk (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save test 2 data to CSV
    if SAVE_CSV:
        test2_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test2_activations,
            "Optimized Desired Angle": test2_desired_angles,
            "Optimized Desired Angle EMG": test2_desired_emg_angles,
        })
        test2_results_df.to_csv(SAVE_PATH + "/test2_results.csv", index=False)

    # Calculate the velocity, acceleration and jerk for the test
    test3_velocities = np.gradient(test3_desired_angles, dt) # np.diff(test3_desired_angles) / dt
    test3_accelerations = np.gradient(test3_velocities, dt) # np.diff(test3_velocities) / dt
    test3_jerks = np.gradient(test3_accelerations, dt) # np.diff(test3_accelerations) / dt
    test3_emg_velocities = np.gradient(test3_desired_emg_angles, dt) # np.diff(test3_desired_emg_angles) / dt
    test3_emg_accelerations = np.gradient(test3_emg_velocities, dt) # np.diff(test3_emg_velocities) / dt
    test3_emg_jerks = np.gradient(test3_emg_accelerations, dt) # np.diff(test3_emg_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test3_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 3: EMG to position - jerk mean: {np.mean(np.abs(test3_jerks)):.2f} degrees/s^3, jerk max: {np.max(test3_jerks):.2f} degrees/s^3, jerk min: {np.min(test3_jerks):.2f} degrees/s^3")
    print(f"Test 3: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test3_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test3_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test3_emg_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 3: EMG to position with optimization 2")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test3_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test3_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test3_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test3_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test3_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 3: EMG to position with optimization 2")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test3_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test3_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test3_emg_velocities, label="Velocity (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test3_emg_accelerations, label="Acceleration (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test3_emg_jerks, label="Jerk (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save test 3 data to CSV
    if SAVE_CSV:
        test3_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test3_activations,
            "Optimized Desired Angle": test3_desired_angles,
            "Optimized Desired Angle EMG": test3_desired_emg_angles,
        })
        test3_results_df.to_csv(SAVE_PATH + "/test3_results.csv", index=False)


    # Calculate the velocity, acceleration and jerk for the test
    test4_velocities = np.gradient(test4_desired_angles, dt) # np.diff(test4_desired_angles) / dt
    test4_accelerations = np.gradient(test4_velocities, dt) # np.diff(test4_velocities) / dt
    test4_jerks = np.gradient(test4_accelerations, dt) # np.diff(test4_accelerations) / dt
    test4_emg_velocities = np.gradient(test4_desired_emg_angles, dt) # np.diff(test4_desired_emg_angles) / dt
    test4_emg_accelerations = np.gradient(test4_emg_velocities, dt) # np.diff(test4_emg_velocities) / dt
    test4_emg_jerks = np.gradient(test4_emg_accelerations, dt) # np.diff(test4_emg_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test4_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 4: EMG to position - jerk mean: {np.mean(np.abs(test4_jerks)):.2f} degrees/s^3, jerk max: {np.max(test4_jerks):.2f} degrees/s^3, jerk min: {np.min(test4_jerks):.2f} degrees/s^3")
    print(f"Test 4: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test4_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test4_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test4_emg_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 4: EMG to position with optimization 4")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test4_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test4_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test4_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test4_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test4_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 4: EMG to position with optimization 4")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test4_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test4_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test4_emg_velocities, label="Velocity (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test4_emg_accelerations, label="Acceleration (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test4_emg_jerks, label="Jerk (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test4_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test4_activations,
            "Optimized Desired Angle": test4_desired_angles,
            "Optimized Desired Angle EMG": test4_desired_emg_angles,
        })
        test4_results_df.to_csv(SAVE_PATH + "/test4_results.csv", index=False)

    # Calculate the velocity, acceleration and jerk for the test
    test5_velocities = np.gradient(test5_desired_angles, dt) # np.diff(test5_desired_angles) / dt
    test5_accelerations = np.gradient(test5_velocities, dt) # np.diff(test5_velocities) / dt
    test5_jerks = np.gradient(test5_accelerations, dt) # np.diff(test5_accelerations) / dt
    test5_emg_velocities = np.gradient(test5_desired_emg_angles, dt) # np.diff(test5_desired_emg_angles) / dt
    test5_emg_accelerations = np.gradient(test5_emg_velocities, dt) # np.diff(test5_emg_velocities) / dt
    test5_emg_jerks = np.gradient(test5_emg_accelerations, dt) # np.diff(test5_emg_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test5_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 5: EMG to position - jerk mean: {np.mean(np.abs(test5_jerks)):.2f} degrees/s^3, jerk max: {np.max(test5_jerks):.2f} degrees/s^3, jerk min: {np.min(test5_jerks):.2f} degrees/s^3")
    print(f"Test 5: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test5_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test5_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test5_emg_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 5: EMG to position with optimization 5")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test5_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test5_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test5_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test5_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test5_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 5: EMG to position with optimization 5")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test5_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test5_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test5_emg_velocities, label="Velocity (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test5_emg_accelerations, label="Acceleration (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test5_emg_jerks, label="Jerk (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test5_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test5_activations,
            "Optimized Desired Angle": test5_desired_angles,
            "Optimized Desired Angle EMG": test5_desired_emg_angles,
        })
        test5_results_df.to_csv(SAVE_PATH + "/test5_results.csv", index=False)

    # Calculate velocity acceleration and jerk
    time_vector = np.arange(len(test_9_desired_angles)) * dt
    test_9_emg_velocities = np.gradient(test_9_desired_emg_angles, dt) # np.diff(test_9_desired_emg_angles) / dt
    test_9_emg_accelerations = np.gradient(test_9_emg_velocities, dt) # np.diff(test_9_emg_velocities) / dt
    test_9_emg_jerks = np.gradient(test_9_emg_accelerations, dt) # np.diff(test_9_emg_accelerations) / dt

    test_9_velocities = np.gradient(test_9_desired_angles, dt) # np.diff(test_9_desired_angles) / dt
    test_9_accelerations = np.gradient(test_9_velocities, dt) # np.diff(test_9_velocities) / dt
    test_9_jerks = np.gradient(test_9_accelerations, dt) # np.diff(test_9_accelerations) / dt

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 9: EMG to position with optimization 6")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test_9_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test_9_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test_9_emg_velocities, label="Velocity (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test_9_emg_accelerations, label="Acceleration (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test_9_emg_jerks, label="Jerk (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 9: EMG to position with optimization 6")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test_9_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test_9_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test_9_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test_9_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test_9_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test9_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test_9_activations,
            "Optimized Desired Angle": test_9_desired_angles,
            "Optimized Desired Angle EMG": test_9_desired_emg_angles,
        })
        test9_results_df.to_csv(SAVE_PATH + "/test9_results.csv", index=False)

    # calculate velocity acceleration and jerk
    time_vector = np.arange(len(test_10_desired_angles)) * dt
    test_10_emg_velocities = np.gradient(test_10_desired_angles_emg, dt) # np.diff(test_10_desired_emg_angles) / dt
    test_10_emg_accelerations = np.gradient(test_10_emg_velocities, dt) # np.diff(test_10_emg_velocities) / dt
    test_10_emg_jerks = np.gradient(test_10_emg_accelerations, dt) # np.diff(test_10_emg_accelerations) / dt

    test_10_velocities = np.gradient(test_10_desired_angles, dt) # np.diff(test_10_desired_angles) / dt
    test_10_accelerations = np.gradient(test_10_velocities, dt) # np.diff(test_10_velocities) / dt
    test_10_jerks = np.gradient(test_10_accelerations, dt) # np.diff(test_10_accelerations) / dt

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 10: EMG to position with EMG Optimizer")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test_10_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test_10_desired_angles_emg, label="Optimized Desired Angle (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test_10_emg_velocities, label="Velocity (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test_10_emg_accelerations, label="Acceleration (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test_10_emg_jerks, label="Jerk (EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 10: EMG to position with EMG Optimizer")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test_10_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test_10_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test_10_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test_10_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test_10_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test10_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test_10_activations,
            "Optimized Desired Angle": test_10_desired_angles,
            "Optimized Desired Angle EMG": test_10_desired_angles_emg,
        })
        test10_results_df.to_csv(SAVE_PATH + "/test10_results.csv", index=False)

    # Calculate the velocity, acceleration and jerk for the test
    test6_velocities = np.gradient(test6_desired_angles, dt) # np.diff(test6_desired_angles) / dt
    test6_accelerations = np.gradient(test6_velocities, dt) # np.diff(test6_velocities) / dt
    test6_jerks = np.gradient(test6_accelerations, dt) # np.diff(test6_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test6_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 6: EMG processing + pDMP - jerk mean: {np.mean(np.abs(test6_jerks)):.2f} degrees/s^3, jerk max: {np.max(test6_jerks):.2f} degrees/s^3, jerk min: {np.min(test6_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 6: EMG processing + pDMP")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test6_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test6_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test6_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test6_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test6_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test6_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test6_activations,
            "Desired Angle": test6_desired_angles,
        })
        test6_results_df.to_csv(SAVE_PATH + "/test6_results.csv", index=False)

    # Calculate the velocity, acceleration and jerk for the test
    test7_velocities = np.gradient(test7_desired_angles, dt) # np.diff(test7_desired_angles) / dt
    test7_accelerations = np.gradient(test7_velocities, dt) # np.diff(test7_velocities) / dt
    test7_jerks = np.gradient(test7_accelerations, dt) # np.diff(test7_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test7_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 7: EMG processing + pDMP coupling 1 - jerk mean: {np.mean(np.abs(test7_jerks)):.2f} degrees/s^3, jerk max: {np.max(test7_jerks):.2f} degrees/s^3, jerk min: {np.min(test7_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 7: EMG processing + pDMP coupling 1")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test7_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test7_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test7_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test7_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test7_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test7_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test7_activations,
            "Desired Angle": test7_desired_angles,
        })
        test7_results_df.to_csv(SAVE_PATH + "/test7_results.csv", index=False)

    # Calculate the velocity, acceleration and jerk for the test
    test8_velocities = np.gradient(test8_desired_angles, dt) # np.diff(test8_desired_angles) / dt
    test8_accelerations = np.gradient(test8_velocities, dt) # np.diff(test8_velocities) / dt
    test8_jerks = np.gradient(test8_accelerations, dt) # np.diff(test8_accelerations) / dt

    # Create time vector for plot to stretch from 0 to 10s instead of samples for plotting
    time_vector = np.arange(len(test8_desired_angles)) * dt
    time_vector_velocity = time_vector[:-1]
    time_vector_acceleration = time_vector[:-2]
    time_vector_jerk = time_vector[:-3]

    # Print stats for the test
    print(f"Test 8: EMG processing + pDMP omega - jerk mean: {np.mean(np.abs(test8_jerks)):.2f} degrees/s^3, jerk max: {np.max(test8_jerks):.2f} degrees/s^3, jerk min: {np.min(test8_jerks):.2f} degrees/s^3")

    # plot the results
    plt.figure(figsize=(15, 10))
    plt.suptitle("Test 8: EMG processing + pDMP omega")
    plt.subplot(5, 1, 1)
    plt.plot(time_vector, test8_activations, label="Net Activation (Bicep - Tricep)")
    plt.xlabel("Time (s)")
    plt.ylabel("Net Activation")
    plt.ylim(-1, 1)
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test8_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector, test8_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector, test8_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector, test8_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    if SAVE_CSV:
        test8_results_df = pd.DataFrame({
            "Time": time_vector,
            "Net Activation": test8_activations,
            "Desired Angle": test8_desired_angles,
        })
        test8_results_df.to_csv(SAVE_PATH + "/test8_results.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------------------------
    emg.stop()