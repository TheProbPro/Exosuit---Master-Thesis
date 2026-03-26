from Optimizations import *
import numpy as np
import matplotlib.pyplot as plt

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

if __name__ == "__main__":
    print(f"Theta max: {THETA_MAX}, Theta min: {THETA_MIN}")
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time = np.linspace(0, 10, 1660)  # Time vector from 0 to 10 seconds
    activation = np.sin(2 * np.pi * 0.19 * time)  # Sine wave with frequency of 0.2 Hz

    # Create empty lists to store optimized angles for each optimizer
    optimized_angles_1 = []
    optimized_angles_2 = []
    optimized_angles_3 = []
    optimized_angles_4 = []
    optimized_angles_5 = []
    
    # Initialize parameters for the optimizers along with the optimizers themselves
    # k = 4.5*np.pi
    k= 1.3 * np.pi / 3
    t = 1/166  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    optimized_angles_1.append(q)
    for a in activation:
        optimized_angles_1.append(optimize_1(k, a, t, optimized_angles_1[-1], THETA_MIN, THETA_MAX))

    k= np.pi
    optimized_angles_2.append(q)
    for a in activation:
        optimized_angles_2.append(optimize_2(k, a, t, optimized_angles_2[-1], THETA_MIN, THETA_MAX))
    
    k= np.pi
    optimized_angles_3.append(q)
    for a in activation:
        optimized_angles_3.append(optimize_3(k, a, t, optimized_angles_3[-1], THETA_MIN, THETA_MAX, 0.1))
    
    k = 0.9 * np.pi
    optimized_angles_4.append(q)
    delta_q_prev = 0
    for a in activation:
        optimized_angle, delta_q_prev = optimize_4(k, a, t, optimized_angles_4[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        optimized_angles_4.append(optimized_angle)
    
    # k = np.pi
    optimized_angles_5.append(q)
    for a in activation:
        optimized_angles_5.append(optimize_5_pd(a, k, t, optimized_angles_5[-1], THETA_MIN, THETA_MAX, 1, 0.01))
    
    # Remove the initial angle from the optimized angles lists
    optimized_angles_1.remove(optimized_angles_1[0])
    optimized_angles_2.remove(optimized_angles_2[0])
    optimized_angles_3.remove(optimized_angles_3[0])
    optimized_angles_4.remove(optimized_angles_4[0])
    optimized_angles_5.remove(optimized_angles_5[0])
    

    # Calculate the velocity, acceleration and jerk for each optimizer
    velocities_1 = np.diff(optimized_angles_1) / t
    accelerations_1 = np.diff(velocities_1) / t
    jerks_1 = np.diff(accelerations_1) / t

    velocities_2 = np.diff(optimized_angles_2) / t
    accelerations_2 = np.diff(velocities_2) / t
    jerks_2 = np.diff(accelerations_2) / t

    velocities_3 = np.diff(optimized_angles_3) / t
    accelerations_3 = np.diff(velocities_3) / t
    jerks_3 = np.diff(accelerations_3) / t

    velocities_4 = np.diff(optimized_angles_4) / t
    accelerations_4 = np.diff(velocities_4) / t
    jerks_4 = np.diff(accelerations_4) / t

    velocities_5 = np.diff(optimized_angles_5) / t
    accelerations_5 = np.diff(velocities_5) / t
    jerks_5 = np.diff(accelerations_5) / t

    # Plot each optimized angle in different graphs comparing them to the input signal and with the position, velocity, acceleration and jerk.
    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 1")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_1, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")

    plt.subplot(5, 1, 3)
    plt.plot(time[:-1], velocities_1, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time[:-2], accelerations_1, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/s^2)")

    plt.subplot(5, 1, 5)
    plt.plot(time[:-3], jerks_1, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/s^3)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 2")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_2, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time[:-1], velocities_2, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time[:-2], accelerations_2, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/s^2)")

    plt.subplot(5, 1, 5)
    plt.plot(time[:-3], jerks_2, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/s^3)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 3")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_3, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time[:-1], velocities_3, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time[:-2], accelerations_3, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/s^2)")

    plt.subplot(5, 1, 5)
    plt.plot(time[:-3], jerks_3, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/s^3)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 4")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_4, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time[:-1], velocities_4, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time[:-2], accelerations_4, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/s^2)")

    plt.subplot(5, 1, 5)
    plt.plot(time[:-3], jerks_4, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/s^3)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 5")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_5, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")

    plt.subplot(5, 1, 3)
    plt.plot(time[:-1], velocities_5, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time[:-2], accelerations_5, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/s^2)")

    plt.subplot(5, 1, 5)
    plt.plot(time[:-3], jerks_5, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/s^3)")
    plt.tight_layout()
    plt.show()