from Optimizations import *
import numpy as np
import matplotlib.pyplot as plt

THETA_MIN = 0
THETA_MAX = 140

if __name__ == "__main__":
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time = np.linspace(0, 10, 166)  # Time vector from 0 to 10 seconds
    activation = np.sin(2 * np.pi * 0.2 * time)  # Sine wave with frequency of 0.2 Hz

    # Create empty lists to store optimized angles for each optimizer
    optimized_angles_1 = []
    optimized_angles_2 = []
    optimized_angles_3 = []
    optimized_angles_4 = []
    optimized_angles_5 = []
    
    # Initialize parameters for the optimizers along with the optimizers themselves
    k = np.pi
    t = 1/166  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    optimized_angles_1.append(q)
    for a in activation:
        optimized_angles_1.append(optimize_1(k, a, t, optimized_angles_1[-1], THETA_MIN, THETA_MAX))

    optimized_angles_2.append(q)
    for a in activation:
        optimized_angles_2.append(optimize_2(k, a, t, optimized_angles_2[-1], THETA_MIN, THETA_MAX))
    
    optimized_angles_3.append(q)
    for a in activation:
        optimized_angles_3.append(optimize_3(k, a, t, optimized_angles_3[-1], THETA_MIN, THETA_MAX))
    
    optimized_angles_4.append(q)
    for a in activation:
        optimized_angles_4.append(optimize_4(k, a, t, optimized_angles_4[-1], THETA_MIN, THETA_MAX))
    
    optimized_angles_5.append(q)
    for a in activation:
        optimized_angles_5.append(optimize_5_pd(a, k, t, optimized_angles_5[-1], THETA_MIN, THETA_MAX, 1))
    
    
    
    plt.plot(time, activation)
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.title("Test EMG Signal")
    plt.show()

    optimized_angles_1.remove(optimized_angles_1[0])  # Remove the initial angle from the list
    optimized_angles_2.remove(optimized_angles_2[0])  # Remove the initial angle from the list
    optimized_angles_3.remove(optimized_angles_3[0])  # Remove the initial angle from the list
    optimized_angles_4.remove(optimized_angles_4[0])  # Remove the initial angle from the list
    optimized_angles_5.remove(optimized_angles_5[0])  # Remove the initial angle from the list
    print("Optimized angles for Optimizer 1:", len(optimized_angles_1), np.max(optimized_angles_1), np.min(optimized_angles_1))
    print("Optimized angles for Optimizer 2:", len(optimized_angles_2), np.max(optimized_angles_2), np.min(optimized_angles_2))
    print("Optimized angles for Optimizer 3:", len(optimized_angles_3), np.max(optimized_angles_3), np.min(optimized_angles_3))
    print("Optimized angles for Optimizer 4:", len(optimized_angles_4), np.max(optimized_angles_4), np.min(optimized_angles_4))
    print("Optimized angles for Optimizer 5:", len(optimized_angles_5), np.max(optimized_angles_5), np.min(optimized_angles_5))
    # Plot the optimized angles for each optimizer
    plt.figure(figsize=(12, 8))
    plt.subplot(5, 1, 1)
    plt.plot(time, optimized_angles_1)
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle")
    plt.title("Optimizer 1")
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_2)
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle")
    plt.title("Optimizer 2")
    plt.subplot(5, 1, 3)
    plt.plot(time, optimized_angles_3)
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle")
    plt.title("Optimizer 3")
    plt.subplot(5, 1, 4)
    plt.plot(time, optimized_angles_4)
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle")
    plt.title("Optimizer 4")
    plt.subplot(5, 1, 5)
    plt.plot(time, optimized_angles_5)
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle")
    plt.title("Optimizer 5")
    plt.show()