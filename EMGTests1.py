'''
The EMG processing tests will consist of both prediction and optimization algorithms.
The prediction algorithms should be tested both before and after the optimization algorithms, and both at the same time.
So this means:
 EMG -> optimization -> prediction
 EMG -> prediction -> optimization
 EMG -> prediction -> optimization -> prediction
'''
# Custom includes
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
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

USERNAME = "VictorBNielsen"

EMG_FS = 2000  # EMG sampling frequency (Hz)
MOTOR_FS = 166.7  # Motor control frequency (Hz)
IMU_FS = 0  # IMU sampling frequency (Hz) TODO: SET THIS LATER

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

TAU_MAX = 4.1
TAU_MIN = -TAU_MAX

stop_event = threading.Event()

# Define threading method for gathering emg data
def read_EMG(emg_pos_queue, emg_activation_queue):
    # Initialize filters
    filter_bicep = rt_filtering(EMG_FS, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_FS, 450, 20, 2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USERNAME, BicepEMG=True, TricepEMG=True)
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

        try:
            emg_activation_queue.put_nowait(activation)
        except queue.Full:
            emg_activation_queue.get_nowait()
            emg_activation_queue.put_nowait(activation)

        desired_angle_deg = math.degrees(interpreter.compute_angle(activation[0], activation[1]))

        try:
            emg_pos_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            emg_pos_queue.get_nowait()
            emg_pos_queue.put_nowait(desired_angle_deg)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

# Graceful Ctrl-C
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)


# Define main
if __name__ == "__main__":
    # Define EMG queues
    emg_pos_queue = queue.Queue(maxsize=5)
    emg_activation_queue = queue.Queue(maxsize=5)

    # Define desired angle lowpass filter
    desired_angle_lowpass = rt_desired_Angle_lowpass(sample_rate=EMG_FS, lp_cutoff=3, order=2)

    # Create and start the EMG thread
    emg_thread = threading.Thread(target=read_EMG, args=(emg_pos_queue, emg_activation_queue))
    emg_thread.start()
    time.sleep(1.0)  # Allow some time for the EMG thread to start and gather data

    # Test 1: Pure EMG processing
    test1_desired_angles = []
    print("Starting Test 1: EMG to position")
    ptime = []
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            desired_angle_deg = emg_pos_queue.get_nowait()
            filtered_desired_angle = float(desired_angle_lowpass.lowpass(np.atleast_1d(desired_angle_deg))[0])
            test1_desired_angles.append(filtered_desired_angle)
            ptime.append(time.time() - last_time)
            last_time = time.time()
        except queue.Empty:
            continue

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()


    # Test 2: EMG processing + optimization 1
    test2_desired_emg_angles = []
    test2_desired_angles = []
    test2_activations = []
    k = 1.3 * np.pi / 3
    q = 0  # Initial angle (degrees)
    print("Starting Test 2: EMG processing + optimization 1")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            desired_angle_deg = emg_pos_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(desired_angle_lowpass.lowpass(np.atleast_1d(desired_angle_deg))[0])

        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()
        optimized_angle_emg = optimize_1(k, a, dt, filtered_desired_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_1(k, a, dt, q, THETA_MIN, THETA_MAX)

        test2_desired_emg_angles.append(optimized_angle_emg)
        test2_desired_angles.append(optimized_angle)
        test2_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 3: EMG processing + optimization 2
    test3_desired_emg_angles = []
    test3_desired_angles = []
    test3_activations = []
    k = np.pi
    q = 0  # Initial angle (degrees)
    print("Starting Test 3: EMG processing + optimization 2")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            desired_angle_deg = emg_pos_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(desired_angle_lowpass.lowpass(np.atleast_1d(desired_angle_deg))[0])

        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()
        optimized_angle_emg = optimize_2(k, a, dt, filtered_desired_angle, THETA_MIN, THETA_MAX)
        optimized_angle = optimize_2(k, a, dt, q, THETA_MIN, THETA_MAX)

        test3_desired_emg_angles.append(optimized_angle_emg)
        test3_desired_angles.append(optimized_angle)
        test3_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 4: EMG processing + optimization 4
    test4_desired_emg_angles = []
    test4_desired_angles = []
    test4_activations = []
    k = 0.9 * np.pi
    q = 0  # Initial angle (degrees)
    delta_q_prev_emg = 0
    delta_q_prev = 0
    print("Starting Test 4: EMG processing + optimization 4")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            desired_angle_deg = emg_pos_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(desired_angle_lowpass.lowpass(np.atleast_1d(desired_angle_deg))[0])

        dt = time.time() - last_time
        last_time = time.time()
        ptime.append(dt)
        optimized_angle_emg, delta_q_prev_emg = optimize_4(k, a, dt, filtered_desired_angle, delta_q_prev_emg, THETA_MIN, THETA_MAX)
        optimized_angle, delta_q_prev = optimize_4(k, a, dt, q, delta_q_prev, THETA_MIN, THETA_MAX)

        test4_desired_emg_angles.append(optimized_angle_emg)
        test4_desired_angles.append(optimized_angle)
        test4_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 5: EMG processing + optimization 5
    test5_desired_emg_angles = []
    test5_desired_angles = []
    test5_activations = []
    k = 1
    b = 0.01
    v = 0.9 * np.pi
    q = 0  # Initial angle (degrees)

    print("Starting Test 5: EMG processing + optimization 5")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run the test for 10 seconds
        try:
            desired_angle_deg = emg_pos_queue.get_nowait()
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        filtered_desired_angle = float(desired_angle_lowpass.lowpass(np.atleast_1d(desired_angle_deg))[0])

        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()
        optimized_angle_emg = optimize_5_pd(a, v, dt, filtered_desired_angle, THETA_MIN, THETA_MAX, k, b)
        optimized_angle = optimize_5_pd(a, v, dt, q, THETA_MIN, THETA_MAX, k, b)

        test5_desired_emg_angles.append(optimized_angle_emg)
        test5_desired_angles.append(optimized_angle)
        test5_activations.append(a)

    print(f"processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 6: emg processing + pDMP
    dt = 1/166.7
    phi = 0
    tau = 0.5
    DMP = pDMP(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    # Teach DMP 0 trajectory for 2s
    y_old = 0
    dy_old = 0
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 2:  # Teach for 2 seconds
        phi += 16*np.pi * dt/tau
        y = np.array([0])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        DMP.set_phase(phi)
        DMP.set_period(tau)
        DMP.learn(y, dy, ddy)
        DMP.integration()
        ptime.append(time.time() - last_time)
        last_time = time.time()

        # old values	
        y_old = y
        dy_old = dy
        
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

    print(f"Teaching processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    ptime.clear()

    # Run DMP with EMG input for 10s
    test6_desired_angles = []
    test6_activations = []
    v = (1.3*np.pi)/2
    print("Starting Test 6: EMG processing + pDMP")
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run for 10 seconds
        try:
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        
        DMP.set_phase(phi)
        DMP.set_period(tau)

        U = np.asarray([a*v])  # EMG activation as input
        DMP.update(U)
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        test6_desired_angles.append(x[0])
        test6_activations.append(a)
        dt = time.time() - last_time
        ptime.append(dt)
        last_time = time.time()

    print(f"Running processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Test 7: emg processing + pDMP coupling
    dt = 1/166.7
    phi = 0
    tau = 0.5
    DMP = pDMPCoupling1(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    # Teach DMP 0 trajectory for 2s
    y_old = 0
    dy_old = 0
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 2:  # Teach for 2 seconds
        phi += 2*np.pi * dt/tau
        y = np.array([0])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        DMP.set_phase(phi)
        DMP.set_period(tau)
        DMP.learn(y, dy, ddy)
        DMP.integration()
        ptime.append(time.time() - last_time)
        last_time = time.time()

        # old values	
        y_old = y
        dy_old = dy
        
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

    print(f"Teaching processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    ptime.clear()

    # Run DMP with EMG input for 10s
    test7_desired_angles = []
    test7_activations = []
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 10:  # Run for 10 seconds
        try:
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        
        DMP.set_phase(phi)
        DMP.set_period(tau)

        DMP.repeat()

        DMP.integration(a)

        x, dx, ph, ta = DMP.get_state()
        test7_desired_angles.append(x[0])
        test7_activations.append(a)
        ptime.append(time.time() - last_time)
        last_time = time.time()

    print(f"Running processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()


    # Test 8: emg processing + pDMP omega
    dt = 1/166.7
    phi = 0
    tau = 0.5
    omega0 = 2*np.pi/tau
    DMP = pDMPOmega(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    DMP.set_frequency([omega0])
    # Teach DMP 0 trajectory for 2s
    y_old = 0
    dy_old = 0
    start_time = time.time()
    last_time = start_time
    while time.time() - start_time < 2:  # Teach for 2 seconds
        y = np.array([np.sin(omega0*(time.time() - start_time))])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt

        DMP.learn(y, dy, ddy)
        DMP.integration()
        ptime.append(time.time() - last_time)
        last_time = time.time()

        # old values	
        y_old = y
        dy_old = dy
        
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

    print(f"Teaching processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    ptime.clear()

    # Run DMP with EMG input for 10s
    test8_desired_angles = []
    test8_activations = []
    start_time = time.time()
    last_time = start_time
    k = 1.0
    while time.time() - start_time < 10:  # Run for 10 seconds
        try:
            activation = emg_activation_queue.get_nowait()
        except queue.Empty:
            continue

        a = activation[0] - activation[1]  # Compute net activation (bicep - tricep)
        omega = omega0 * (1 + k * a)
        
        DMP.set_frequency([omega])

        DMP.repeat()

        DMP.integration()

        x, dx, ph, ta = DMP.get_state()
        test8_desired_angles.append(x[0])
        test8_activations.append(a)
        ptime.append(time.time() - last_time)
        last_time = time.time()

    print(f"Running processing time {np.mean(ptime):.2f} ms, operating frequency {1/np.mean(ptime):.2f} Hz")
    emg_pos_queue.queue.clear()  # Clear the queue after the test
    emg_activation_queue.queue.clear()  # Clear the queue after the test
    ptime.clear()

    # Calculate the velocity, acceleration and jerk for each test
    test1_velocities = np.diff(test1_desired_angles) / dt
    test1_accelerations = np.diff(test1_velocities) / dt
    test1_jerks = np.diff(test1_accelerations) / dt

    test2_velocities = np.diff(test2_desired_angles) / dt
    test2_accelerations = np.diff(test2_velocities) / dt
    test2_jerks = np.diff(test2_accelerations) / dt

    test2_emg_velocities = np.diff(test2_desired_emg_angles) / dt
    test2_emg_accelerations = np.diff(test2_emg_velocities) / dt
    test2_emg_jerks = np.diff(test2_emg_accelerations) / dt

    test3_velocities = np.diff(test3_desired_angles) / dt
    test3_accelerations = np.diff(test3_velocities) / dt
    test3_jerks = np.diff(test3_accelerations) / dt

    test3_emg_velocities = np.diff(test3_desired_emg_angles) / dt
    test3_emg_accelerations = np.diff(test3_emg_velocities) / dt
    test3_emg_jerks = np.diff(test3_emg_accelerations) / dt

    test4_velocities = np.diff(test4_desired_angles) / dt
    test4_accelerations = np.diff(test4_velocities) / dt
    test4_jerks = np.diff(test4_accelerations) / dt

    test4_emg_velocities = np.diff(test4_desired_emg_angles) / dt
    test4_emg_accelerations = np.diff(test4_emg_velocities) / dt
    test4_emg_jerks = np.diff(test4_emg_accelerations) / dt
    
    test5_velocities = np.diff(test5_desired_angles) / dt
    test5_accelerations = np.diff(test5_velocities) / dt
    test5_jerks = np.diff(test5_accelerations) / dt

    test5_emg_velocities = np.diff(test5_desired_emg_angles) / dt
    test5_emg_accelerations = np.diff(test5_emg_velocities) / dt
    test5_emg_jerks = np.diff(test5_emg_accelerations) / dt

    test6_velocities = np.diff(test6_desired_angles) / dt
    test6_accelerations = np.diff(test6_velocities) / dt
    test6_jerks = np.diff(test6_accelerations) / dt

    test7_velocities = np.diff(test7_desired_angles) / dt
    test7_accelerations = np.diff(test7_velocities) / dt
    test7_jerks = np.diff(test7_accelerations) / dt

    test8_velocities = np.diff(test8_desired_angles) / dt
    test8_accelerations = np.diff(test8_velocities) / dt
    test8_jerks = np.diff(test8_accelerations) / dt

    # Print stats for each test
    print(f"Test 1: EMG to position - jerk mean: {np.mean(np.abs(test1_jerks)):.2f} degrees/s^3, jerk max: {np.max(test1_jerks):.2f} degrees/s^3, jerk min: {np.min(test1_jerks):.2f} degrees/s^3")
    print(f"Test 2: EMG to position - jerk mean: {np.mean(np.abs(test2_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_jerks):.2f} degrees/s^3")
    print(f"Test 2: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test2_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test2_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test2_emg_jerks):.2f} degrees/s^3")
    print(f"Test 3: EMG to position - jerk mean: {np.mean(np.abs(test3_jerks)):.2f} degrees/s^3, jerk max: {np.max(test3_jerks):.2f} degrees/s^3, jerk min: {np.min(test3_jerks):.2f} degrees/s^3")
    print(f"Test 3: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test3_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test3_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test3_emg_jerks):.2f} degrees/s^3")
    print(f"Test 4: EMG to position - jerk mean: {np.mean(np.abs(test4_jerks)):.2f} degrees/s^3, jerk max: {np.max(test4_jerks):.2f} degrees/s^3, jerk min: {np.min(test4_jerks):.2f} degrees/s^3")
    print(f"Test 4: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test4_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test4_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test4_emg_jerks):.2f} degrees/s^3")
    print(f"Test 5: EMG to position - jerk mean: {np.mean(np.abs(test5_jerks)):.2f} degrees/s^3, jerk max: {np.max(test5_jerks):.2f} degrees/s^3, jerk min: {np.min(test5_jerks):.2f} degrees/s^3")
    print(f"Test 5: EMG to position (EMG optimized) - jerk mean: {np.mean(np.abs(test5_emg_jerks)):.2f} degrees/s^3, jerk max: {np.max(test5_emg_jerks):.2f} degrees/s^3, jerk min: {np.min(test5_emg_jerks):.2f} degrees/s^3")
    print(f"Test 6: EMG to position - jerk mean: {np.mean(np.abs(test6_jerks)):.2f} degrees/s^3, jerk max: {np.max(test6_jerks):.2f} degrees/s^3, jerk min: {np.min(test6_jerks):.2f} degrees/s^3")
    print(f"Test 7: EMG to position - jerk mean: {np.mean(np.abs(test7_jerks)):.2f} degrees/s^3, jerk max: {np.max(test7_jerks):.2f} degrees/s^3, jerk min: {np.min(test7_jerks):.2f} degrees/s^3")
    print(f"Test 8: EMG to position - jerk mean: {np.mean(np.abs(test8_jerks)):.2f} degrees/s^3, jerk max: {np.max(test8_jerks):.2f} degrees/s^3, jerk min: {np.min(test8_jerks):.2f} degrees/s^3")

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
    plt.plot(test2_desired_emg_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test2_emg_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test2_emg_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test2_emg_jerks, label="Jerk")
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
    plt.plot(test3_desired_emg_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test3_emg_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test3_emg_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test3_emg_jerks, label="Jerk")
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
    plt.plot(test4_desired_emg_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test4_emg_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test4_emg_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test4_emg_jerks, label="Jerk")
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
    plt.plot(test5_desired_emg_angles, label="Desired Angle (EMG optimized)")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test5_emg_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test5_emg_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test5_emg_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("Test 6: EMG to position + pDMP")
    plt.subplot(5,1,1)
    plt.plot(test6_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test6_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test6_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test6_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test6_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.title("Test 7: EMG to position + pDMP coupling")
    plt.subplot(5,1,1)
    plt.plot(test7_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test7_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test7_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test7_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test7_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 10))
    plt.title("Test 8: EMG to position + pDMP omega")
    plt.subplot(5,1,1)
    plt.plot(test8_activations, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(test8_desired_angles, label="Desired Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Desired Angle (degrees)")
    plt.subplot(5, 1, 3)
    plt.plot(test8_velocities, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (degrees/s)")
    plt.subplot(5, 1, 4)
    plt.plot(test8_accelerations, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (degrees/s^2)")
    plt.subplot(5, 1, 5)
    plt.plot(test8_jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (degrees/s^3)")
    plt.tight_layout()
    plt.show()
