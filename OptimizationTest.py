from Optimizations import *
from ProjectInRobotics.pDMP.pDMP_functions import pDMP, pDMPCoupling1, pDMPOmega
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    
    'font.size': 10,          # default text size
    'axes.titlesize': 14,     # title
    'axes.labelsize': 12,     # x and y labels
    'xtick.labelsize': 10,    # x tick labels
    'ytick.labelsize': 10,    # y tick labels
    'legend.fontsize': 10,    
    'figure.titlesize': 16
})

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

def compute_jerk_metrics(j):

    abs_j = np.abs(j)

    metrics = {
        "mean": np.mean(abs_j),
        "median": np.median(abs_j),
        "sigma": np.std(abs_j),
        "max": np.max(abs_j),
        "q25": np.percentile(abs_j, 25),
        "q75": np.percentile(abs_j, 75),
    }

    return j, abs_j, metrics

Integrator_labels = [
        "Integrator 1",
        "Integrator 2",
        "Integrator 3",
        "Integrator 4",
        "Integrator 5",
        "Integrator 6",
        "Integrator 7",
        "Integrator 8",
        "pDMP weight update",
        "pDMP coupling term"
    ]

integrator1 = []
integrator2 = []
integrator3 = []
integrator4 = []
integrator5 = []
integrator6 = []
integrator7 = []
integrator8 = []
pDMPIntegrator = []
couplingDMPIntegrator = []

FS = 2000 # EMG
if __name__ == "__main__":
    print("Starting EMG optimization test at 2000 Hz...")
    print(f"Theta max: {THETA_MAX}, Theta min: {THETA_MIN}")
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time_v= np.linspace(0, 20, FS*20)  # Time vector from 0 to 20 seconds
    activation = np.sin(2 * np.pi * 0.15 * time_v)  # Sine wave with frequency of 0.2 Hz

    # Small random noise
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 1, size=time_v.shape)

    # Smooth it with a moving average
    window_size = 100  # increase for smoother wobble
    kernel = np.ones(window_size) / window_size
    smooth_noise = np.convolve(noise, kernel, mode="same")

    # Scale the noise so it only creates a small wobble
    # noise_amplitude = 0.03
    noise_amplitude = 0.06

    activation += noise_amplitude * smooth_noise

    activation = np.clip(activation, -1, 1)

    # calculate the difference in activations
    activation_diff = np.diff(activation, prepend=activation[0]) / (1/FS)

    # Plot activation and activation difference
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_v, activation, label='Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.title('Muscle Activation (EMG Signal)')
    plt.subplot(2, 1, 2)
    plt.plot(time_v, activation_diff, label='Activation Difference', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation Difference')
    plt.title('Difference of Muscle Activation')
    plt.tight_layout()
    plt.show()

    # Create empty lists to store optimized angles for each optimizer
    optimized_angles_1 = []
    optimized_angles_2 = []
    # optimized_angles_3 = []
    optimized_angles_4 = []
    optimized_angles_5 = []
    optimized_angles_6 = []
    optimized_angles_7 = []
    DMP_angles = []
    DMP_Coupled_angles = []
    DMP_Omega_angles = []
    
    # Initialize parameters for the optimizers along with the optimizers themselves
    k = (1.2*np.pi) / 3 #* 2# EMG
    # k = (1.4*np.pi)/3
    t = 1/FS  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    optimized_angles_1.append(q)
    for a in activation:
        optimized_angles_1.append(optimize_1(k, a, t, optimized_angles_1[-1], THETA_MIN, THETA_MAX))

    print(f"maximum angle for optimizer 1: {np.rad2deg(max(optimized_angles_1)):.2f} degrees, minimum angle for optimizer 1: {np.rad2deg(min(optimized_angles_1)):.2f} degrees")

    # k= 2 * np.pi # EMG
    k = np.pi * 0.9
    optimized_angles_2.append(q)
    for a in activation:
        optimized_angles_2.append(optimize_2(k, a, t, optimized_angles_2[-1], THETA_MIN, THETA_MAX))
    print(f"maximum angle for optimizer 2: {np.rad2deg(max(optimized_angles_2)):.2f} degrees, minimum angle for optimizer 2: {np.rad2deg(min(optimized_angles_2)):.2f} degrees")
    
    # k= 5.8 * np.pi
    # optimized_angles_3.append(q)
    # for a in activation:
    #     optimized_angles_3.append(optimize_3(k, a, t, optimized_angles_3[-1], THETA_MIN, THETA_MAX, 0.1))
    # print(f"maximum angle for optimizer 3: {np.rad2deg(max(optimized_angles_3)):.2f} degrees, minimum angle for optimizer 3: {np.rad2deg(min(optimized_angles_3)):.2f} degrees")

    k = (1.6*np.pi) / 4 # EMG
    # k = (1.4*np.pi)/3
    optimized_angles_4.append(q)
    delta_q_prev = 0
    for a in activation:
        optimized_angle, delta_q_prev = optimize_4(k, a, t, optimized_angles_4[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        optimized_angles_4.append(optimized_angle)
    print(f"maximum angle for optimizer 4: {np.rad2deg(max(optimized_angles_4)):.2f} degrees, minimum angle for optimizer 4: {np.rad2deg(min(optimized_angles_4)):.2f} degrees")
    
    k = 0 # EMG
    # k = np.pi / 4
    n = (1.3*np.pi) / 3
    b = 0.01 # 0.001
    optimized_angles_5.append(q)
    for a in activation:
        q_next, k = optimize_5_pd(a, k, t, optimized_angles_5[-1], THETA_MIN, THETA_MAX, np.pi, n, b)
        optimized_angles_5.append(q_next)
    print(f"maximum angle for optimizer 5: {np.rad2deg(max(optimized_angles_5)):.2f} degrees, minimum angle for optimizer 5: {np.rad2deg(min(optimized_angles_5)):.2f} degrees")
    
    v = 0  # Initial velocity
    k = np.pi * 1.6
    b = 4
    optimized_angles_6.append(q)
    for a in activation:
        q_next, v, acc = optimizer_6(a, v, t, optimized_angles_6[-1], THETA_MIN, THETA_MAX, np.pi, b, k)
        optimized_angles_6.append(q_next)
    print(f"maximum angle for optimizer 6: {np.rad2deg(max(optimized_angles_6)):.2f} degrees, minimum angle for optimizer 6: {np.rad2deg(min(optimized_angles_6)):.2f} degrees")

    optimized_angles_7.append(q)
    kn = 4
    kd = 4
    b = 4
    for a, da in zip(activation, activation_diff):
        q_next, v, acc = EMG_Optimizer(a, da, v, kn, kd, b, optimized_angles_7[-1], THETA_MIN, THETA_MAX, np.pi, t)
        optimized_angles_7.append(q_next)
    print(f"maximum angle for optimizer 7: {np.rad2deg(max(optimized_angles_7)):.2f} degrees, minimum angle for optimizer 7: {np.rad2deg(min(optimized_angles_7)):.2f} degrees")

    #========================================= DMP's ====================================
    # Teach DMP
    dt = 1/FS
    phi = 0
    tau = 0.5
    DMP = pDMP(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    DMP.set_output_limits(THETA_MIN, THETA_MAX, squash_gain=1.0)
    DMP.set_output_state(np.array([0.0]))
    y_old = 0
    dy_old = 0
    start_time = time.time()
    while time.time() - start_time < 3:  # Teach for 3 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        phi += 2*np.pi * dt/tau #16*np.pi * dt/tau
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

    # Run DMP
    v = np.pi/35 #np.pi/22
    # v = np.pi/2
    for a in activation:
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        U = np.asarray([a*v])  # EMG activation as input
        DMP.update(U)
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        DMP_angles.append(x[0])

    print(f"maximum angle for DMP: {np.rad2deg(max(DMP_angles)):.2f} degrees, minimum angle for DMP: {np.rad2deg(min(DMP_angles)):.2f} degrees")

    # Teach Coupled DMP
    DMP = pDMPCoupling1(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    # Teach DMP 0 trajectory for 3s
    y_old = 0
    dy_old = 0
    start_time = time.time()
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

    # Run Coupled DMP
    for a in activation:
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        DMP.repeat()

        DMP.integration(np.array([a]))

        x, dx, ph, ta = DMP.get_state()
        DMP_Coupled_angles.append(x[0])

    print(f"maximum angle for Coupled DMP: {np.rad2deg(max(DMP_Coupled_angles)):.2f} degrees, minimum angle for Coupled DMP: {np.rad2deg(min(DMP_Coupled_angles)):.2f} degrees")

    # Teach Omega DMP
    tau = 5
    omega0 = 2*np.pi/tau
    mid = np.deg2rad(70)
    DMP = pDMPOmega(DOF=1, N=25, alpha=8, beta=2, lambd=0.999, dt=dt)
    DMP.set_frequency([omega0])
    # Teach DMP 0 trajectory for 3s
    y_old = 0
    dy_old = 0
    start_time = time.time()
    samples = (1/dt) * 5
    for i in range(int(samples)):
        t = i * dt
        y = np.array([mid * np.sin(omega0*t) + mid])
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

    # Run Omega DMP
    k = 1.0
    for a in activation:
        omega = omega0 * (1 + k * a)
        DMP.set_frequency([omega])
        DMP.repeat()
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        DMP_Omega_angles.append(x[0])

    print(f"maximum angle for Omega DMP: {np.rad2deg(max(DMP_Omega_angles)):.2f} degrees, minimum angle for Omega DMP: {np.rad2deg(min(DMP_Omega_angles)):.2f} degrees")

    t = 1/FS

    # Remove the initial angle from the optimized angles lists
    optimized_angles_1.remove(optimized_angles_1[0])
    optimized_angles_2.remove(optimized_angles_2[0])
    # optimized_angles_3.remove(optimized_angles_3[0])
    optimized_angles_4.remove(optimized_angles_4[0])
    optimized_angles_5.remove(optimized_angles_5[0])
    optimized_angles_6.remove(optimized_angles_6[0])
    optimized_angles_7.remove(optimized_angles_7[0])

    integrator1.extend(optimized_angles_1)
    integrator2.extend(optimized_angles_2)
    integrator3.extend(optimized_angles_4)
    integrator4.extend(optimized_angles_5)
    integrator5.extend(optimized_angles_6)
    integrator6.extend(optimized_angles_7)
    pDMPIntegrator.extend(DMP_angles)
    couplingDMPIntegrator.extend(DMP_Coupled_angles)
    

    # Calculate the velocity, acceleration and jerk for each optimizer
    velocities_1 = np.gradient(optimized_angles_1, t)
    accelerations_1 = np.gradient(velocities_1, t)
    jerks_1 = np.gradient(accelerations_1, t)

    velocities_2 = np.gradient(optimized_angles_2, t)
    accelerations_2 = np.gradient(velocities_2, t)
    jerks_2 = np.gradient(accelerations_2, t)

    # velocities_3 = np.diff(optimized_angles_3) / t
    # accelerations_3 = np.diff(velocities_3) / t
    # jerks_3 = np.diff(accelerations_3) / t

    velocities_4 = np.gradient(optimized_angles_4, t)
    accelerations_4 = np.gradient(velocities_4, t)
    jerks_4 = np.gradient(accelerations_4, t)

    velocities_5 = np.gradient(optimized_angles_5, t)
    accelerations_5 = np.gradient(velocities_5, t)
    jerks_5 = np.gradient(accelerations_5, t)

    velocities_6 = np.gradient(optimized_angles_6, t)
    accelerations_6 = np.gradient(velocities_6, t)
    jerks_6 = np.gradient(accelerations_6, t)

    velocities_7 = np.gradient(optimized_angles_7, t)
    accelerations_7 = np.gradient(velocities_7, t)
    jerks_7 = np.gradient(accelerations_7, t)

    DMP_velocities = np.gradient(DMP_angles, t)
    DMP_accelerations = np.gradient(DMP_velocities, t)
    DMP_jerks = np.gradient(DMP_accelerations, t)

    DMP_Coupled_velocities = np.gradient(DMP_Coupled_angles, t)
    DMP_Coupled_accelerations = np.gradient(DMP_Coupled_velocities, t)
    DMP_Coupled_jerks = np.gradient(DMP_Coupled_accelerations, t)

    DMP_Omega_velocities = np.gradient(DMP_Omega_angles, t)
    DMP_Omega_accelerations = np.gradient(DMP_Omega_velocities, t)
    DMP_Omega_jerks = np.gradient(DMP_Omega_accelerations, t)

    # Plot each optimized angle in different graphs comparing them to the input signal and with the position, velocity, acceleration and jerk.
    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 1: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_1, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_1, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_1, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_1, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 2: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_2, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_2, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_2, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_2, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    # plt.figure(figsize=(12, 10))
    # plt.title("Optimizer 3: EMG")
    # plt.subplot(5, 1, 1)
    # plt.plot(time, activation, label="Activation")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Activation")

    # plt.subplot(5, 1, 2)
    # plt.plot(time, optimized_angles_3, label="Optimized Angle")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Optimized Angle (rad)")
    
    # plt.subplot(5, 1, 3)
    # plt.plot(time[:-1], velocities_3, label="Velocity")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity (rad/s)")

    # plt.subplot(5, 1, 4)
    # plt.plot(time[:-2], accelerations_3, label="Acceleration")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (rad/s^2)")

    # plt.subplot(5, 1, 5)
    # plt.plot(time[:-3], jerks_3, label="Jerk")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Jerk (rad/s^3)")
    # plt.tight_layout()
    # plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 4: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_4, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_4, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_4, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_4, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 5: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_5, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_5, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_5, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_5, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 6: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_6, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_6, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_6, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_6, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 7: EMG with PD control and acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_7, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_7, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_7, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_7, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("pDMP")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, DMP_angles, label="DMP Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, DMP_velocities, label="DMP Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, DMP_accelerations, label="DMP Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, DMP_jerks, label="DMP Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("pDMP Coupled")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, DMP_Coupled_angles, label="DMP Coupled Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, DMP_Coupled_velocities, label="DMP Coupled Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, DMP_Coupled_accelerations, label="DMP Coupled Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, DMP_Coupled_jerks, label="DMP Coupled Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("pDMP Omega")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, DMP_Omega_angles, label="DMP Omega Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Omega Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, DMP_Omega_velocities, label="DMP Omega Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Omega Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, DMP_Omega_accelerations, label="DMP Omega Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Omega Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, DMP_Omega_jerks, label="DMP Omega Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Omega Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    # create labels
    labels = [
        "Optimizer 1",
        "Optimizer 2",
        "Optimizer 3",
        "Optimizer 4",
        "Optimizer 5",
        "Optimizer 6",
        "pDMP",
        "pDMP Coupled",
        "pDMP Omega"
    ]
    # Calculate the jerk metrics
    j1, abs_j1, j_metrics1 = compute_jerk_metrics(jerks_1)
    j2, abs_j2, j_metrics2 = compute_jerk_metrics(jerks_2)
    # jerk_metrics_3 = compute_jerk_metrics(jerks_3)
    j4, abs_j4, j_metrics4 = compute_jerk_metrics(jerks_4)
    j5, abs_j5, j_metrics5 = compute_jerk_metrics(jerks_5)
    j6, abs_j6, j_metrics6 = compute_jerk_metrics(jerks_6)
    j7, abs_j7, j_metrics7 = compute_jerk_metrics(jerks_7)
    jDMP, abs_jDMP, j_metricsDMP = compute_jerk_metrics(DMP_jerks)
    jDMP_Coupled, abs_jDMP_Coupled, j_metricsDMP_Coupled = compute_jerk_metrics(DMP_Coupled_jerks)
    jDMP_Omega, abs_jDMP_Omega, j_metricsDMP_Omega = compute_jerk_metrics(DMP_Omega_jerks)

    # create vectors for the metrics
    means = [j_metrics1["mean"], j_metrics2["mean"], j_metrics4["mean"], j_metrics5["mean"], j_metrics6["mean"], j_metrics7["mean"], j_metricsDMP["mean"], j_metricsDMP_Coupled["mean"], j_metricsDMP_Omega["mean"]]
    medians = [j_metrics1["median"], j_metrics2["median"], j_metrics4["median"], j_metrics5["median"], j_metrics6["median"], j_metrics7["median"], j_metricsDMP["median"], j_metricsDMP_Coupled["median"], j_metricsDMP_Omega["median"]]
    sigmas = [j_metrics1["sigma"], j_metrics2["sigma"], j_metrics4["sigma"], j_metrics5["sigma"], j_metrics6["sigma"], j_metrics7["sigma"], j_metricsDMP["sigma"], j_metricsDMP_Coupled["sigma"], j_metricsDMP_Omega["sigma"]]
    maxs = [j_metrics1["max"], j_metrics2["max"], j_metrics4["max"], j_metrics5["max"], j_metrics6["max"], j_metrics7["max"], j_metricsDMP["max"], j_metricsDMP_Coupled["max"], j_metricsDMP_Omega["max"]]
    q25s = [j_metrics1["q25"], j_metrics2["q25"], j_metrics4["q25"], j_metrics5["q25"], j_metrics6["q25"], j_metrics7["q25"], j_metricsDMP["q25"], j_metricsDMP_Coupled["q25"], j_metricsDMP_Omega["q25"]]
    q75s = [j_metrics1["q75"], j_metrics2["q75"], j_metrics4["q75"], j_metrics5["q75"], j_metrics6["q75"], j_metrics7["q75"], j_metricsDMP["q75"], j_metricsDMP_Coupled["q75"], j_metricsDMP_Omega["q75"]]
    lower_errors = [mean - q25 for mean, q25 in zip(means, q25s)]
    upper_errors = [q75 - mean for mean, q75 in zip(means, q75s)]
    lower_median_errors = [mean - median for mean, median in zip(means, medians)]
    upper_median_errors = [median - mean for mean, median in zip(means, medians)]
    lower_errors = np.maximum(lower_errors, 0)
    upper_errors = np.maximum(upper_errors, 0)
    lower_median_errors = np.maximum(lower_median_errors, 0)
    upper_median_errors = np.maximum(upper_median_errors, 0)

    # Print mean and median jerk for each optimizer
    print("Jerk Metrics for Each Optimizer:")
    for label, mean, median, sigma, max_val in zip(labels, means, medians, sigmas, maxs):
        print(f"{label}: Mean Jerk = {mean:.2e}, Median Jerk = {median:.2e}, Sigma = {sigma:.2e}, Max Jerk = {max_val:.2e}")

    # Create bar plots
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=[lower_median_errors, upper_median_errors], color='skyblue')
    plt.scatter(labels, maxs, color='red', label='Max Jerk')
    plt.ylabel('Mean Absolute Jerk (rad/s^3)')
    plt.yscale("symlog", linthresh=0.01)
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Mean Jerk (log scale)")
    # plt.title("Mean Jerk")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, medians, yerr=[lower_median_errors, upper_median_errors], color='lightgreen')
    plt.scatter(labels, maxs, color='red', label='Max Jerk')
    plt.yscale("symlog", linthresh=0.01)
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Median Jerk (log scale)")
    # plt.title("Median Jerk for Different Optimizers")
    plt.legend()
    plt.tight_layout()
    plt.show()

    abs_jerk_data = [abs_j1, abs_j2, abs_j4, abs_j5, abs_j6, abs_j7, abs_jDMP, abs_jDMP_Coupled, abs_jDMP_Omega]

    #create box plots
    plt.figure(figsize=(7, 4))
    plt.boxplot(abs_jerk_data, labels=labels, showfliers=False)
    # plt.yscale('log')
    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Distribution of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    # create violin plot
    plt.figure(figsize=(7, 4))
    violin = plt.violinplot(
        abs_jerk_data,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(
        ticks=np.arange(1, len(labels) + 1),
        labels=labels,
        rotation=45
    )
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Violin Plot of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    print(f"best median jerk: {min(medians):.2e}, optimizer: {labels[medians.index(min(medians))]}")





    print("Starting IMU optimization test at 148 Hz...")
    FS = 148 # IMU
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time_v = np.linspace(0, 20, FS*20)  # Time vector from 0 to 10 seconds
    activation = np.sin(2 * np.pi * 0.15 * time_v)  # Sine wave with frequency of 0.2 Hz

    # Small random noise
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 1, size=time_v.shape)

    # Smooth it with a moving average
    window_size = 30  # increase for smoother wobble
    kernel = np.ones(window_size) / window_size
    smooth_noise = np.convolve(noise, kernel, mode="same")

    # Scale the noise so it only creates a small wobble
    # noise_amplitude = 0.03
    noise_amplitude = 0.06

    activation += noise_amplitude * smooth_noise

    activation = np.clip(activation, -1, 1)

    delay = 0.08  # 80 ms delay (typical electromechanical delay)
    q_true = np.sin(2 * np.pi * 0.15 * (time_v-delay))
    omega = np.gradient(q_true, t)
    imu_q = q_true + 0.01 * np.random.randn(len(q_true))  # noisy angle

    plt.plot(time_v, activation, label="Activation")
    plt.plot(time_v, imu_q, label="True Angle")
    plt.legend()
    plt.show()

    # Create empty lists to store optimized angles for each optimizer
    optimized_angles_1 = []
    optimized_angles_2 = []
    # optimized_angles_3 = []
    optimized_angles_4 = []
    optimized_angles_5 = []
    optimized_angles_6 = []
    optimized_angles_8 = []
    optimized_angles_9 = []
    
    # Initialize parameters for the optimizers along with the optimizers themselves
    # k = 4.8 * np.pi # IMU
    k = (1.2 * np.pi) / 3
    t = 1/FS  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    optimized_angles_1.append(q)
    for a in activation:
        optimized_angles_1.append(optimize_1(k, a, t, optimized_angles_1[-1], THETA_MIN, THETA_MAX))

    print(f"maximum angle for optimizer 1: {np.rad2deg(max(optimized_angles_1)):.2f} degrees, minimum angle for optimizer 1: {np.rad2deg(min(optimized_angles_1)):.2f} degrees")

    # k = 14 * np.pi # IMU
    k = 1.5 * np.pi
    optimized_angles_2.append(q)
    for a in activation:
        optimized_angles_2.append(optimize_2(k, a, t, optimized_angles_2[-1], THETA_MIN, THETA_MAX))
    print(f"maximum angle for optimizer 2: {np.rad2deg(max(optimized_angles_2)):.2f} degrees, minimum angle for optimizer 2: {np.rad2deg(min(optimized_angles_2)):.2f} degrees")
    
    # k= 5.8 * np.pi
    # optimized_angles_3.append(q)
    # for a in activation:
    #     optimized_angles_3.append(optimize_3(k, a, t, optimized_angles_3[-1], THETA_MIN, THETA_MAX, 0.05))
    # print(f"maximum angle for optimizer 3: {np.rad2deg(max(optimized_angles_3)):.2f} degrees, minimum angle for optimizer 3: {np.rad2deg(min(optimized_angles_3)):.2f} degrees")

    # k = 4.8 * np.pi # IMU
    k = (1.2 * np.pi) / 3
    optimized_angles_4.append(q)
    delta_q_prev = 0
    for a in activation:
        optimized_angle, delta_q_prev = optimize_4(k, a, t, optimized_angles_4[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        optimized_angles_4.append(optimized_angle)
    print(f"maximum angle for optimizer 4: {np.rad2deg(max(optimized_angles_4)):.2f} degrees, minimum angle for optimizer 4: {np.rad2deg(min(optimized_angles_4)):.2f} degrees")
    
    k = 0 # IMU
    n = (1.3*np.pi) / 3
    b = 0.01 # 0.001
    optimized_angles_5.append(q)
    for a in activation:
        q_next, k = optimize_5_pd(a, k, t, optimized_angles_5[-1], THETA_MIN, THETA_MAX, np.pi, n, b)
        optimized_angles_5.append(q_next)
    print(f"maximum angle for optimizer 5: {np.rad2deg(max(optimized_angles_5)):.2f} degrees, minimum angle for optimizer 5: {np.rad2deg(min(optimized_angles_5)):.2f} degrees")
    
    k = np.pi * 1.6
    b = 4
    v = 0  # Initial velocity
    optimized_angles_6.append(q)
    for a in activation:
        q_next, v, acc = optimizer_6(a, v, t, optimized_angles_6[-1], THETA_MIN, THETA_MAX, np.pi, b, k)
        optimized_angles_6.append(q_next)
    print(f"maximum angle for optimizer 6: {np.rad2deg(max(optimized_angles_6)):.2f} degrees, minimum angle for optimizer 6: {np.rad2deg(min(optimized_angles_6)):.2f} degrees")

    v = 0
    optimized_angles_8.append(q)
    for a, da, w, imu in zip(activation, activation_diff, omega, imu_q):
        q_next, v, acc = EMG_IMU_optimizer(
            a, da, v, w,
            kn=2, kd=2, kp=2, b=3,
            q=optimized_angles_8[-1],
            imu_q=imu,
            theta_min=THETA_MIN,
            theta_max=THETA_MAX,
            v_max=np.pi,
            t=t
        )
        optimized_angles_8.append(q_next)

    optimized_angles_9.append(q)
    for a, da, w, imu in zip(activation, activation_diff, omega, imu_q):
        q_next, v = EMG_IMU_optimizer_2(
            a, da, w,
            kn=1.2, kd=1.2,
            imu_q=optimized_angles_9[-1],
            theta_min=THETA_MIN,
            theta_max=THETA_MAX,
            v_max=np.pi,
            t=t
        )
        optimized_angles_9.append(q_next)

    # Remove the initial angle from the optimized angles lists
    optimized_angles_1.remove(optimized_angles_1[0])
    optimized_angles_2.remove(optimized_angles_2[0])
    # optimized_angles_3.remove(optimized_angles_3[0])
    optimized_angles_4.remove(optimized_angles_4[0])
    optimized_angles_5.remove(optimized_angles_5[0])
    optimized_angles_6.remove(optimized_angles_6[0])
    optimized_angles_8.remove(optimized_angles_8[0])
    optimized_angles_9.remove(optimized_angles_9[0])

    integrator7.extend(optimized_angles_8)
    integrator8.extend(optimized_angles_9)
    

    # Calculate the velocity, acceleration and jerk for each optimizer
    velocities_1 = np.gradient(optimized_angles_1, t)
    accelerations_1 = np.gradient(velocities_1, t)
    jerks_1 = np.gradient(accelerations_1, t)

    velocities_2 = np.gradient(optimized_angles_2, t)
    accelerations_2 = np.gradient(velocities_2, t)
    jerks_2 = np.gradient(accelerations_2, t)

    # velocities_3 = np.diff(optimized_angles_3) / t
    # accelerations_3 = np.diff(velocities_3) / t
    # jerks_3 = np.diff(accelerations_3) / t

    velocities_4 = np.gradient(optimized_angles_4, t)
    accelerations_4 = np.gradient(velocities_4, t)
    jerks_4 = np.gradient(accelerations_4, t)

    velocities_5 = np.gradient(optimized_angles_5, t)
    accelerations_5 = np.gradient(velocities_5, t)
    jerks_5 = np.gradient(accelerations_5, t)

    velocities_6 = np.gradient(optimized_angles_6, t)
    accelerations_6 = np.gradient(velocities_6, t)
    jerks_6 = np.gradient(accelerations_6, t)

    velocities_8 = np.gradient(optimized_angles_8, t)
    accelerations_8 = np.gradient(velocities_8, t)
    jerks_8 = np.gradient(accelerations_8, t)

    velocities_9 = np.gradient(optimized_angles_9, t)
    accelerations_9 = np.gradient(velocities_9, t)
    jerks_9 = np.gradient(accelerations_9, t)

    # Plot each optimized angle in different graphs comparing them to the input signal and with the position, velocity, acceleration and jerk.
    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 1: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_1, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_1, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_1, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_1, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 2: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_2, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_2, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_2, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_2, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    # plt.figure(figsize=(12, 10))
    # plt.title("Optimizer 3: IMU")
    # plt.subplot(5, 1, 1)
    # plt.plot(time, activation, label="Activation")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Activation")

    # plt.subplot(5, 1, 2)
    # plt.plot(time, optimized_angles_3, label="Optimized Angle")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Optimized Angle (rad)")
    
    # plt.subplot(5, 1, 3)
    # plt.plot(time[:-1], velocities_3, label="Velocity")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity (rad/s)")

    # plt.subplot(5, 1, 4)
    # plt.plot(time[:-2], accelerations_3, label="Acceleration")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Acceleration (rad/s^2)")

    # plt.subplot(5, 1, 5)
    # plt.plot(time[:-3], jerks_3, label="Jerk")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Jerk (rad/s^3)")
    # plt.tight_layout()
    # plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 4: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_4, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_4, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_4, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_4, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 5: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_5, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_5, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_5, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_5, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 6: IMU with PD control")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_6, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_6, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_6, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_6, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 8: IMU with PD control and acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    # plt.plot(time, omega, label="Angular Velocity", color='orange')
    plt.plot(time_v, imu_q, label="IMU Angle", color='green')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_8, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_8, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_8, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_8, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 9: IMU with PD control and no acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    # plt.plot(time, omega, label="Angular Velocity", color='orange')
    plt.plot(time_v, imu_q, label="IMU Angle", color='green')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, optimized_angles_9, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_9, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_9, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_9, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    # create labels
    labels = [
        "Optimizer 1",
        "Optimizer 2",
        "Optimizer 3",
        "Optimizer 4",
        "Optimizer 5",
        "Optimizer 7",
        "Optimizer 8"
    ]
    # Calculate the jerk metrics
    j1, abs_j1, j_metrics1 = compute_jerk_metrics(jerks_1)
    j2, abs_j2, j_metrics2 = compute_jerk_metrics(jerks_2)
    # jerk_metrics_3 = compute_jerk_metrics(jerks_3)
    j4, abs_j4, j_metrics4 = compute_jerk_metrics(jerks_4)
    j5, abs_j5, j_metrics5 = compute_jerk_metrics(jerks_5)
    j6, abs_j6, j_metrics6 = compute_jerk_metrics(jerks_6)
    j8, abs_j8, j_metrics8 = compute_jerk_metrics(jerks_8)
    j9, abs_j9, j_metrics9 = compute_jerk_metrics(jerks_9)

    # create vectors for the metrics
    means = [j_metrics1["mean"], j_metrics2["mean"], j_metrics4["mean"], j_metrics5["mean"], j_metrics6["mean"], j_metrics8["mean"], j_metrics9["mean"]]
    medians = [j_metrics1["median"], j_metrics2["median"], j_metrics4["median"], j_metrics5["median"], j_metrics6["median"], j_metrics8["median"], j_metrics9["median"]]
    sigmas = [j_metrics1["sigma"], j_metrics2["sigma"], j_metrics4["sigma"], j_metrics5["sigma"], j_metrics6["sigma"], j_metrics8["sigma"], j_metrics9["sigma"]]
    maxs = [j_metrics1["max"], j_metrics2["max"], j_metrics4["max"], j_metrics5["max"], j_metrics6["max"], j_metrics8["max"], j_metrics9["max"]]
    q25s = [j_metrics1["q25"], j_metrics2["q25"], j_metrics4["q25"], j_metrics5["q25"], j_metrics6["q25"], j_metrics8["q25"], j_metrics9["q25"]]
    q75s = [j_metrics1["q75"], j_metrics2["q75"], j_metrics4["q75"], j_metrics5["q75"], j_metrics6["q75"], j_metrics8["q75"], j_metrics9["q75"]]
    lower_errors = [mean - q25 for mean, q25 in zip(means, q25s)]
    upper_errors = [q75 - mean for mean, q75 in zip(means, q75s)]
    lower_median_errors = [mean - median for mean, median in zip(means, medians)]
    upper_median_errors = [median - mean for mean, median in zip(means, medians)]
    lower_errors = np.maximum(lower_errors, 0)
    upper_errors = np.maximum(upper_errors, 0)
    lower_median_errors = np.maximum(lower_median_errors, 0)
    upper_median_errors = np.maximum(upper_median_errors, 0)

    # Create bar plots
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=[lower_median_errors, upper_median_errors], color='skyblue')
    plt.scatter(labels, maxs, color='red', label='Max Jerk')
    plt.ylabel('Mean Absolute Jerk (rad/s^3)')
    plt.yscale("symlog", linthresh=0.01)
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Mean Jerk (log scale)")
    # plt.title("Mean Jerk")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, medians, yerr=[lower_median_errors, upper_median_errors], color='lightgreen')
    plt.scatter(labels, maxs, color='red', label='Max Jerk')
    plt.yscale("symlog", linthresh=0.01)
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Median Jerk (log scale)")
    # plt.title("Median Jerk for Different Optimizers")
    plt.legend()
    plt.tight_layout()
    plt.show()

    abs_jerk_data = [abs_j1, abs_j2, abs_j4, abs_j5, abs_j6, abs_j8, abs_j9]

    #create box plots
    plt.figure(figsize=(7, 4))
    plt.boxplot(abs_jerk_data, labels=labels, showfliers=False)
    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Distribution of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    # create violin plot
    plt.figure(figsize=(7, 4))
    violin = plt.violinplot(
        abs_jerk_data,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(
        ticks=np.arange(1, len(labels) + 1),
        labels=labels,
        rotation=45
    )
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Violin Plot of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    print(f"best median jerk: {min(medians):.2e}, optimizer: {labels[medians.index(min(medians))]}")

    # Calculate jerk
    t1 = 1/2000
    t2 = 1/148
    integrator1_velocities = np.gradient(integrator1, t1)
    integrator1_accelerations = np.gradient(integrator1_velocities, t1)
    integrator1_jerks = np.gradient(integrator1_accelerations, t1)

    integrator2_velocities = np.gradient(integrator2, t1)
    integrator2_accelerations = np.gradient(integrator2_velocities, t1)
    integrator2_jerks = np.gradient(integrator2_accelerations, t1)

    integrator3_velocities = np.gradient(integrator3, t1)
    integrator3_accelerations = np.gradient(integrator3_velocities, t1)
    integrator3_jerks = np.gradient(integrator3_accelerations, t1)

    integrator4_velocities = np.gradient(integrator4, t1)
    integrator4_accelerations = np.gradient(integrator4_velocities, t1)
    integrator4_jerks = np.gradient(integrator4_accelerations, t1)

    integrator5_velocities = np.gradient(integrator5, t1)
    integrator5_accelerations = np.gradient(integrator5_velocities, t1)
    integrator5_jerks = np.gradient(integrator5_accelerations, t1)

    integrator6_velocities = np.gradient(integrator6, t1)
    integrator6_accelerations = np.gradient(integrator6_velocities, t1)
    integrator6_jerks = np.gradient(integrator6_accelerations, t1)

    integrator7_velocities = np.gradient(integrator7, t2)
    integrator7_accelerations = np.gradient(integrator7_velocities, t2)
    integrator7_jerks = np.gradient(integrator7_accelerations, t2)

    integrator8_velocities = np.gradient(integrator8, t2)
    integrator8_accelerations = np.gradient(integrator8_velocities, t2)
    integrator8_jerks = np.gradient(integrator8_accelerations, t2)

    pDMPIntegrator_velocities = np.gradient(pDMPIntegrator, t1)
    pDMPIntegrator_accelerations = np.gradient(pDMPIntegrator_velocities, t1)
    pDMPIntegrator_jerks = np.gradient(pDMPIntegrator_accelerations, t1)

    couplingDMPIntegrator_velocities = np.gradient(couplingDMPIntegrator, t1)
    couplingDMPIntegrator_accelerations = np.gradient(couplingDMPIntegrator_velocities, t1)
    couplingDMPIntegrator_jerks = np.gradient(couplingDMPIntegrator_accelerations, t1)

    # Compute jerk metrics
    j_integrator1, abs_j_integrator1, j_metrics_integrator1 = compute_jerk_metrics(integrator1_jerks)
    j_integrator2, abs_j_integrator2, j_metrics_integrator2 = compute_jerk_metrics(integrator2_jerks)
    j_integrator3, abs_j_integrator3, j_metrics_integrator3 = compute_jerk_metrics(integrator3_jerks)
    j_integrator4, abs_j_integrator4, j_metrics_integrator4 = compute_jerk_metrics(integrator4_jerks)
    j_integrator5, abs_j_integrator5, j_metrics_integrator5 = compute_jerk_metrics(integrator5_jerks)
    j_integrator6, abs_j_integrator6, j_metrics_integrator6 = compute_jerk_metrics(integrator6_jerks)
    j_integrator7, abs_j_integrator7, j_metrics_integrator7 = compute_jerk_metrics(integrator7_jerks)
    j_integrator8, abs_j_integrator8, j_metrics_integrator8 = compute_jerk_metrics(integrator8_jerks)
    j_pDMPIntegrator, abs_j_pDMPIntegrator, j_metrics_pDMPIntegrator = compute_jerk_metrics(pDMPIntegrator_jerks)
    j_couplingDMPIntegrator, abs_j_couplingDMPIntegrator, j_metrics_couplingDMPIntegrator = compute_jerk_metrics(couplingDMPIntegrator_jerks)

    # Create vectors for the metrics
    means = [
        j_metrics_integrator1["mean"],
        j_metrics_integrator2["mean"],
        j_metrics_integrator3["mean"],
        j_metrics_integrator4["mean"],
        j_metrics_integrator5["mean"],
        j_metrics_integrator6["mean"],
        j_metrics_integrator7["mean"],
        j_metrics_integrator8["mean"],
        j_metrics_pDMPIntegrator["mean"],
        j_metrics_couplingDMPIntegrator["mean"]
    ]
    medians = [
        j_metrics_integrator1["median"],
        j_metrics_integrator2["median"],
        j_metrics_integrator3["median"],
        j_metrics_integrator4["median"],
        j_metrics_integrator5["median"],
        j_metrics_integrator6["median"],
        j_metrics_integrator7["median"],
        j_metrics_integrator8["median"],
        j_metrics_pDMPIntegrator["median"],
        j_metrics_couplingDMPIntegrator["median"]
    ]
    sigmas = [
        j_metrics_integrator1["sigma"],
        j_metrics_integrator2["sigma"],
        j_metrics_integrator3["sigma"],
        j_metrics_integrator4["sigma"],
        j_metrics_integrator5["sigma"],
        j_metrics_integrator6["sigma"],
        j_metrics_integrator7["sigma"],
        j_metrics_integrator8["sigma"],
        j_metrics_pDMPIntegrator["sigma"],
        j_metrics_couplingDMPIntegrator["sigma"]
    ]
    maxs = [
        j_metrics_integrator1["max"],
        j_metrics_integrator2["max"],
        j_metrics_integrator3["max"],
        j_metrics_integrator4["max"],
        j_metrics_integrator5["max"],
        j_metrics_integrator6["max"],
        j_metrics_integrator7["max"],
        j_metrics_integrator8["max"],
        j_metrics_pDMPIntegrator["max"],
        j_metrics_couplingDMPIntegrator["max"]
    ]
    q25s = [
        j_metrics_integrator1["q25"],
        j_metrics_integrator2["q25"],
        j_metrics_integrator3["q25"],
        j_metrics_integrator4["q25"],
        j_metrics_integrator5["q25"],
        j_metrics_integrator6["q25"],
        j_metrics_integrator7["q25"],
        j_metrics_integrator8["q25"],
        j_metrics_pDMPIntegrator["q25"],
        j_metrics_couplingDMPIntegrator["q25"]
    ]
    q75s = [
        j_metrics_integrator1["q75"],
        j_metrics_integrator2["q75"],
        j_metrics_integrator3["q75"],
        j_metrics_integrator4["q75"],
        j_metrics_integrator5["q75"],
        j_metrics_integrator6["q75"],
        j_metrics_integrator7["q75"],
        j_metrics_integrator8["q75"],
        j_metrics_pDMPIntegrator["q75"],
        j_metrics_couplingDMPIntegrator["q75"]
    ]
    lower_errors = [mean - q25 for mean, q25 in zip(means, q25s)]
    upper_errors = [q75 - mean for mean, q75 in zip(means, q75s)]
    lower_median_errors = [mean - median for mean, median in zip(means, medians)]
    upper_median_errors = [median - mean for mean, median in zip(means, medians)]
    lower_errors = np.maximum(lower_errors, 0)
    upper_errors = np.maximum(upper_errors, 0)
    lower_median_errors = np.maximum(lower_median_errors, 0)
    upper_median_errors = np.maximum(upper_median_errors, 0)

    abs_jerk_data = [
        abs_j_integrator1,
        abs_j_integrator2,
        abs_j_integrator3,
        abs_j_integrator4,
        abs_j_integrator5,
        abs_j_integrator6,
        abs_j_integrator7,
        abs_j_integrator8,
        abs_j_pDMPIntegrator,
        abs_j_couplingDMPIntegrator
    ]
    
    # Create box plot
    plt.figure(figsize=(7, 4))
    plt.boxplot(abs_jerk_data, labels=Integrator_labels, showfliers=False)
    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(rotation=45)
    plt.xlabel("Trajectory generator")
    plt.ylabel("Absolute Jerk (log scale)")
    plt.tight_layout()
    plt.show()
