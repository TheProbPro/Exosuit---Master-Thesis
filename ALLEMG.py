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
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'

USERNAME = "VictorBNielsen"

FS = 2000 #Hz

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

if __name__ == "__main__":
    print("Initializing EMG's...")
    # Define filters and interpretors for EMG processing
    filter_bicep = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
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
        test1_activations.append(activation[0] - activation[1])  # Store net activation (bicep - tricep)

        test1_desired_angles.append(interpreter.compute_angle(activation[0], activation[1]))
        
        # sleep_time = dt - (time.time()-last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()
    
    print(f"length of test1_desired_angles: {len(test1_desired_angles)}, frequency {(len(test1_desired_angles)/10):.2f} Hz, average processing time {10/len(test1_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test2_desired_emg_angles = []
    test2_desired_angles = []
    test2_activations = []
    k = 4.8 * np.pi
    q = 0  # Initial angle (rad)
    test2_desired_angles.append(q)
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(interpreter.compute_angle(activation[0], activation[1]))
 
        t = time.time() 
        delta_t = t - last_t
        last_t = t

        optimized_angle_emg = optimize_1(k, a, delta_t, filtered_desired_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_1(k, a, delta_t, test2_desired_angles[-1], THETA_MIN, THETA_MAX)
        test2_desired_emg_angles.append(optimized_angle_emg)
        test2_desired_angles.append(optimized_angle)
        test2_activations.append(a)

        # sleep_time = dt - (time.time()-last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    # remove the initial angle from the optimized angles lists
    test2_desired_angles.remove(test2_desired_angles[0])

    print(f"length of test2_desired_angles: {len(test2_desired_angles)}, frequency {(len(test2_desired_angles)/10):.2f} Hz, average processing time {10/len(test2_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------
        
    test3_desired_emg_angles = []
    test3_desired_angles = []
    test3_activations = []
    k = 14*np.pi #18 * np.pi
    q = 0  # Initial angle (degrees)
    test3_desired_angles.append(q)
    print("Press Enter to start test 3: EMG to position with optimization 2")
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(interpreter.compute_angle(activation[0], activation[1]))

        t = time.time() 
        delta_t = t - last_t
        last_t = t

        optimized_angle_emg = optimize_2(k, a, delta_t, filtered_desired_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_2(k, a, delta_t, test3_desired_angles[-1], THETA_MIN, THETA_MAX)
        test3_desired_emg_angles.append(optimized_angle_emg)
        test3_desired_angles.append(optimized_angle)
        test3_activations.append(a)

        # sleep_time = dt - (time.time()-last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    # remove the initial angle from the optimized angles lists
    test3_desired_angles.remove(test3_desired_angles[0])

    print(f"length of test3_desired_angles: {len(test3_desired_angles)}, frequency {(len(test3_desired_angles)/10):.2f} Hz, average processing time {10/len(test3_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test4_desired_emg_angles = []
    test4_desired_angles = []
    test4_activations = []
    k = 11.5 * np.pi
    q = 0  # Initial angle (degrees)
    test4_desired_angles.append(q)
    delta_q_prev_emg = 0
    delta_q_prev = 0
    print("Press Enter to start test 4: EMG to position with optimization 4")
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(interpreter.compute_angle(activation[0], activation[1]))
        
        t = time.time() 
        delta_t = t - last_t
        last_t = t

        optimized_angle_emg, delta_q_prev_emg = optimize_4(k, a, delta_t, filtered_desired_angle, delta_q_prev_emg, THETA_MIN, THETA_MAX)
        optimized_angle, delta_q_prev = optimize_4(k, a, delta_t, test4_desired_angles[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        test4_desired_emg_angles.append(optimized_angle_emg)
        test4_desired_angles.append(optimized_angle)
        test4_activations.append(a)
        
        # sleep_time = dt - (time.time()-last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    # remove the initial angle from the optimized angles lists
    test4_desired_angles.remove(test4_desired_angles[0])

    print(f"length of test4_desired_angles: {len(test4_desired_angles)}, frequency {(len(test4_desired_angles)/10):.2f} Hz, average processing time {10/len(test4_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

    test5_desired_emg_angles = []
    test5_desired_angles = []
    test5_activations = []
    k = 2
    b = 0.01
    v = 4 * np.pi
    q = 0  # Initial angle (degrees)
    test5_desired_angles.append(q)
    print("Press Enter to start test 5: EMG to position with optimization 5")
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

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(interpreter.compute_angle(activation[0], activation[1]))

        t = time.time()
        delta_t = t - last_t
        last_t = t

        optimized_angle_emg = optimize_5_pd(a, v, delta_t, filtered_desired_angle, THETA_MIN, THETA_MAX, k, b)
        optimized_angle = optimize_5_pd(a, v, delta_t, test5_desired_angles[-1], THETA_MIN, THETA_MAX, k, b)
        test5_desired_emg_angles.append(optimized_angle_emg)
        test5_desired_angles.append(optimized_angle)
        test5_activations.append(a)

        # sleep_time = dt - (time.time()-last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    # remove the initial angle from the optimized angles lists
    test5_desired_angles.remove(test5_desired_angles[0])

    print(f"length of test5_desired_angles: {len(test5_desired_angles)}, frequency {(len(test5_desired_angles)/10):.2f} Hz, average processing time {10/len(test5_desired_angles)} ms")

    #----------------------------------------------------------------------------------------------------------------------------------

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
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)

        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        U = np.asarray([a*v])  # EMG activation as input
        DMP.update(U)
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        test6_desired_angles.append(x[0])
        test6_activations.append(a)

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
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)

        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        DMP.repeat()

        DMP.integration(np.array([a]))

        x, dx, ph, ta = DMP.get_state()
        test7_desired_angles.append(x[0])
        test7_activations.append(a)

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
        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        
        omega = omega0 * (1 + k * a)
        DMP.set_frequency([omega])
        DMP.repeat()
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        test8_desired_angles.append(x[0])
        test8_activations.append(a)

        # sleep_time = dt - (time.time() - last_time)
        # if sleep_time > 0:
        #     time.sleep(sleep_time)
        last_time = time.time()

    print(f"length of test8_desired_angles: {len(test8_desired_angles)}, frequency {(len(test8_desired_angles)/10):.2f} Hz, average processing time {10/len(test8_desired_angles)} ms")    

    # ----------------------------------------------------------------------------------------------------------------------------------

    # Calculate the velocity, acceleration and jerk for the test
    test1_velocities = np.diff(test1_desired_angles) / dt
    test1_accelerations = np.diff(test1_velocities) / dt
    test1_jerks = np.diff(test1_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test1_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test1_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test1_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test1_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test2_velocities = np.diff(test2_desired_angles) / dt
    test2_accelerations = np.diff(test2_velocities) / dt
    test2_jerks = np.diff(test2_accelerations) / dt
    test2_emg_velocities = np.diff(test2_desired_emg_angles) / dt
    test2_emg_accelerations = np.diff(test2_emg_velocities) / dt
    test2_emg_jerks = np.diff(test2_emg_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test2_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.plot(time_vector, test2_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test2_emg_velocities, label="Velocity (EMG)")
    plt.plot(time_vector_velocity, test2_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test2_emg_accelerations, label="Acceleration (EMG)")
    plt.plot(time_vector_acceleration, test2_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test2_emg_jerks, label="Jerk (EMG)")
    plt.plot(time_vector_jerk, test2_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test3_velocities = np.diff(test3_desired_angles) / dt
    test3_accelerations = np.diff(test3_velocities) / dt
    test3_jerks = np.diff(test3_accelerations) / dt
    test3_emg_velocities = np.diff(test3_desired_emg_angles) / dt
    test3_emg_accelerations = np.diff(test3_emg_velocities) / dt
    test3_emg_jerks = np.diff(test3_emg_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test3_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.plot(time_vector, test3_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test3_emg_velocities, label="Velocity (EMG)")
    plt.plot(time_vector_velocity, test3_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test3_emg_accelerations, label="Acceleration (EMG)")
    plt.plot(time_vector_acceleration, test3_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test3_emg_jerks, label="Jerk (EMG)")
    plt.plot(time_vector_jerk, test3_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test4_velocities = np.diff(test4_desired_angles) / dt
    test4_accelerations = np.diff(test4_velocities) / dt
    test4_jerks = np.diff(test4_accelerations) / dt
    test4_emg_velocities = np.diff(test4_desired_emg_angles) / dt
    test4_emg_accelerations = np.diff(test4_emg_velocities) / dt
    test4_emg_jerks = np.diff(test4_emg_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test4_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.plot(time_vector, test4_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test4_emg_velocities, label="Velocity (EMG)")
    plt.plot(time_vector_velocity, test4_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test4_emg_accelerations, label="Acceleration (EMG)")
    plt.plot(time_vector_acceleration, test4_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test4_emg_jerks, label="Jerk (EMG)")
    plt.plot(time_vector_jerk, test4_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test5_velocities = np.diff(test5_desired_angles) / dt
    test5_accelerations = np.diff(test5_velocities) / dt
    test5_jerks = np.diff(test5_accelerations) / dt
    test5_emg_velocities = np.diff(test5_desired_emg_angles) / dt
    test5_emg_accelerations = np.diff(test5_emg_velocities) / dt
    test5_emg_jerks = np.diff(test5_emg_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test5_desired_emg_angles, label="Optimized Desired Angle (EMG)")
    plt.plot(time_vector, test5_desired_angles, label="Optimized Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test5_emg_velocities, label="Velocity (EMG)")
    plt.plot(time_vector_velocity, test5_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test5_emg_accelerations, label="Acceleration (EMG)")
    plt.plot(time_vector_acceleration, test5_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test5_emg_jerks, label="Jerk (EMG)")
    plt.plot(time_vector_jerk, test5_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test6_velocities = np.diff(test6_desired_angles) / dt
    test6_accelerations = np.diff(test6_velocities) / dt
    test6_jerks = np.diff(test6_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test6_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test6_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test6_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test6_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test7_velocities = np.diff(test7_desired_angles) / dt
    test7_accelerations = np.diff(test7_velocities) / dt
    test7_jerks = np.diff(test7_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test7_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test7_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test7_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test7_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    # Calculate the velocity, acceleration and jerk for the test
    test8_velocities = np.diff(test8_desired_angles) / dt
    test8_accelerations = np.diff(test8_velocities) / dt
    test8_jerks = np.diff(test8_accelerations) / dt

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
    plt.subplot(5, 1, 2)
    plt.plot(time_vector, test8_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(time_vector_velocity, test8_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time_vector_acceleration, test8_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(time_vector_jerk, test8_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------------------------------------------------------------------
    emg.stop()