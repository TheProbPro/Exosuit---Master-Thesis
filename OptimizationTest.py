from Optimizations import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

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

FS = 2000 # EMG
if __name__ == "__main__":
    print("Starting EMG optimization test at 2000 Hz...")
    print(f"Theta max: {THETA_MAX}, Theta min: {THETA_MIN}")
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time = np.linspace(0, 10, FS*10)  # Time vector from 0 to 10 seconds
    activation = np.sin(2 * np.pi * 0.2 * time)  # Sine wave with frequency of 0.2 Hz

    # calculate the difference in activations
    activation_diff = np.diff(activation, prepend=activation[0]) / (1/FS)

    # Plot activation and activation difference
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, activation, label='Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.title('Muscle Activation (EMG Signal)')
    plt.subplot(2, 1, 2)
    plt.plot(time, activation_diff, label='Activation Difference', color='orange')
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
    
    # Initialize parameters for the optimizers along with the optimizers themselves
    k = np.pi/2 # EMG
    # k = (1.4*np.pi)/3
    t = 1/FS  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    optimized_angles_1.append(q)
    for a in activation:
        optimized_angles_1.append(optimize_1(k, a, t, optimized_angles_1[-1], THETA_MIN, THETA_MAX))

    print(f"maximum angle for optimizer 1: {np.rad2deg(max(optimized_angles_1)):.2f} degrees, minimum angle for optimizer 1: {np.rad2deg(min(optimized_angles_1)):.2f} degrees")

    k= 2 * np.pi # EMG
    # k = 2 * np.pi
    optimized_angles_2.append(q)
    for a in activation:
        optimized_angles_2.append(optimize_2(k, a, t, optimized_angles_2[-1], THETA_MIN, THETA_MAX))
    print(f"maximum angle for optimizer 2: {np.rad2deg(max(optimized_angles_2)):.2f} degrees, minimum angle for optimizer 2: {np.rad2deg(min(optimized_angles_2)):.2f} degrees")
    
    # k= 5.8 * np.pi
    # optimized_angles_3.append(q)
    # for a in activation:
    #     optimized_angles_3.append(optimize_3(k, a, t, optimized_angles_3[-1], THETA_MIN, THETA_MAX, 0.1))
    # print(f"maximum angle for optimizer 3: {np.rad2deg(max(optimized_angles_3)):.2f} degrees, minimum angle for optimizer 3: {np.rad2deg(min(optimized_angles_3)):.2f} degrees")

    k = np.pi / 2 # EMG
    # k = (1.4*np.pi)/3
    optimized_angles_4.append(q)
    delta_q_prev = 0
    for a in activation:
        optimized_angle, delta_q_prev = optimize_4(k, a, t, optimized_angles_4[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        optimized_angles_4.append(optimized_angle)
    print(f"maximum angle for optimizer 4: {np.rad2deg(max(optimized_angles_4)):.2f} degrees, minimum angle for optimizer 4: {np.rad2deg(min(optimized_angles_4)):.2f} degrees")
    
    k = 0 # EMG
    # k = np.pi / 4
    n = 1.4
    b = 0.01 # 0.001
    optimized_angles_5.append(q)
    for a in activation:
        q_next, k = optimize_5_pd(a, k, t, optimized_angles_5[-1], THETA_MIN, THETA_MAX, np.pi, n, b)
        optimized_angles_5.append(q_next)
    print(f"maximum angle for optimizer 5: {np.rad2deg(max(optimized_angles_5)):.2f} degrees, minimum angle for optimizer 5: {np.rad2deg(min(optimized_angles_5)):.2f} degrees")
    
    v = 0  # Initial velocity
    optimized_angles_6.append(q)
    for a in activation:
        q_next, v, acc = optimizer_6(a, v, t, optimized_angles_6[-1], THETA_MIN, THETA_MAX)
        optimized_angles_6.append(q_next)
    print(f"maximum angle for optimizer 6: {np.rad2deg(max(optimized_angles_6)):.2f} degrees, minimum angle for optimizer 6: {np.rad2deg(min(optimized_angles_6)):.2f} degrees")

    optimized_angles_7.append(q)
    kn = 2
    kd = 2
    b = 2
    for a, da in zip(activation, activation_diff):
        q_next, v, acc = EMG_Optimizer(a, da, v, kn, kd, b, optimized_angles_7[-1], THETA_MIN, THETA_MAX, np.pi, t)
        optimized_angles_7.append(q_next)
    print(f"maximum angle for optimizer 7: {np.rad2deg(max(optimized_angles_7)):.2f} degrees, minimum angle for optimizer 7: {np.rad2deg(min(optimized_angles_7)):.2f} degrees")


    # Remove the initial angle from the optimized angles lists
    optimized_angles_1.remove(optimized_angles_1[0])
    optimized_angles_2.remove(optimized_angles_2[0])
    # optimized_angles_3.remove(optimized_angles_3[0])
    optimized_angles_4.remove(optimized_angles_4[0])
    optimized_angles_5.remove(optimized_angles_5[0])
    optimized_angles_6.remove(optimized_angles_6[0])
    optimized_angles_7.remove(optimized_angles_7[0])
    

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

    # Plot each optimized angle in different graphs comparing them to the input signal and with the position, velocity, acceleration and jerk.
    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 1: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_1, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")

    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_1, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_1, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_1, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 2: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_2, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_2, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_2, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_2, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
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
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_4, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_4, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_4, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_4, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 5: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_5, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")

    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_5, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_5, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_5, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 6: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_6, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_6, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_6, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_6, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 7: EMG with PD control and acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_7, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_7, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_7, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_7, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    # create labels
    labels = [
        "Optimizer 1",
        "Optimizer 2",
        "Optimizer 3",
        "Optimizer 4",
        "Optimizer 5",
        "Optimizer 6"
    ]
    # Calculate the jerk metrics
    j1, abs_j1, j_metrics1 = compute_jerk_metrics(jerks_1)
    j2, abs_j2, j_metrics2 = compute_jerk_metrics(jerks_2)
    # jerk_metrics_3 = compute_jerk_metrics(jerks_3)
    j4, abs_j4, j_metrics4 = compute_jerk_metrics(jerks_4)
    j5, abs_j5, j_metrics5 = compute_jerk_metrics(jerks_5)
    j6, abs_j6, j_metrics6 = compute_jerk_metrics(jerks_6)
    j7, abs_j7, j_metrics7 = compute_jerk_metrics(jerks_7)

    # create vectors for the metrics
    means = [j_metrics1["mean"], j_metrics2["mean"], j_metrics4["mean"], j_metrics5["mean"], j_metrics6["mean"], j_metrics7["mean"]]
    medians = [j_metrics1["median"], j_metrics2["median"], j_metrics4["median"], j_metrics5["median"], j_metrics6["median"], j_metrics7["median"]]
    sigmas = [j_metrics1["sigma"], j_metrics2["sigma"], j_metrics4["sigma"], j_metrics5["sigma"], j_metrics6["sigma"], j_metrics7["sigma"]]
    maxs = [j_metrics1["max"], j_metrics2["max"], j_metrics4["max"], j_metrics5["max"], j_metrics6["max"], j_metrics7["max"]]
    q25s = [j_metrics1["q25"], j_metrics2["q25"], j_metrics4["q25"], j_metrics5["q25"], j_metrics6["q25"], j_metrics7["q25"]]
    q75s = [j_metrics1["q75"], j_metrics2["q75"], j_metrics4["q75"], j_metrics5["q75"], j_metrics6["q75"], j_metrics7["q75"]]
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
    plt.yscale('log')
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
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Median Jerk (log scale)")
    # plt.title("Median Jerk for Different Optimizers")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #create box plots
    plt.figure(figsize=(7, 4))
    plt.boxplot([abs_j1, abs_j2, abs_j4, abs_j5, abs_j6, abs_j7], labels=labels, showfliers=False)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Distribution of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    print(f"best median jerk: {min(medians):.2e}, optimizer: {labels[medians.index(min(medians))]}")





    print("Starting IMU optimization test at 148 Hz...")
    FS = 148 # IMU
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time = np.linspace(0, 10, FS*10)  # Time vector from 0 to 10 seconds
    activation = np.sin(2 * np.pi * 0.2 * time)  # Sine wave with frequency of 0.2 Hz
    delay = 0.08  # 80 ms delay (typical electromechanical delay)
    q_true = np.sin(2 * np.pi * 0.2 * (time-delay))
    omega = np.gradient(q_true, t)
    imu_q = q_true + 0.05 * np.random.randn(len(q_true))  # noisy angle

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
    k = (1.4 * np.pi) / 3
    t = 1/FS  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    optimized_angles_1.append(q)
    for a in activation:
        optimized_angles_1.append(optimize_1(k, a, t, optimized_angles_1[-1], THETA_MIN, THETA_MAX))

    print(f"maximum angle for optimizer 1: {np.rad2deg(max(optimized_angles_1)):.2f} degrees, minimum angle for optimizer 1: {np.rad2deg(min(optimized_angles_1)):.2f} degrees")

    # k = 14 * np.pi # IMU
    k = 2 * np.pi
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
    k = (1.4 * np.pi) / 3
    optimized_angles_4.append(q)
    delta_q_prev = 0
    for a in activation:
        optimized_angle, delta_q_prev = optimize_4(k, a, t, optimized_angles_4[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        optimized_angles_4.append(optimized_angle)
    print(f"maximum angle for optimizer 4: {np.rad2deg(max(optimized_angles_4)):.2f} degrees, minimum angle for optimizer 4: {np.rad2deg(min(optimized_angles_4)):.2f} degrees")
    
    k = 0 # IMU
    n = 1.3
    b = 0.005
    optimized_angles_5.append(q)
    for a in activation:
        q_next, k = optimize_5_pd(a, k, t, optimized_angles_5[-1], THETA_MIN, THETA_MAX, n, b)
        optimized_angles_5.append(q_next)
    print(f"maximum angle for optimizer 5: {np.rad2deg(max(optimized_angles_5)):.2f} degrees, minimum angle for optimizer 5: {np.rad2deg(min(optimized_angles_5)):.2f} degrees")
    
    v = 0  # Initial velocity
    optimized_angles_6.append(q)
    for a in activation:
        q_next, v, acc = optimizer_6(a, v, t, optimized_angles_6[-1], THETA_MIN, THETA_MAX)
        optimized_angles_6.append(q_next)
    print(f"maximum angle for optimizer 6: {np.rad2deg(max(optimized_angles_6)):.2f} degrees, minimum angle for optimizer 6: {np.rad2deg(min(optimized_angles_6)):.2f} degrees")

    v = 0
    optimized_angles_8.append(q)
    for a, da, w, imu in zip(activation, activation_diff, omega, imu_q):
        q_next, v, acc = EMG_IMU_optimizer(
            a, da, v, w,
            kn=2, kd=2, kp=2, b=2,
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
            kn=2, kd=2,
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
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_1, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")

    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_1, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_1, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_1, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 2: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_2, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_2, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_2, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_2, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
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
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_4, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_4, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_4, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_4, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 5: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")

    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_5, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")

    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_5, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")

    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_5, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")

    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_5, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 6: IMU with PD control")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_6, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_6, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_6, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_6, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 8: IMU with PD control and acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    # plt.plot(time, omega, label="Angular Velocity", color='orange')
    plt.plot(time, imu_q, label="IMU Angle", color='green')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_8, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_8, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_8, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_8, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 9: IMU with PD control and no acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time, activation, label="Activation")
    # plt.plot(time, omega, label="Angular Velocity", color='orange')
    plt.plot(time, imu_q, label="IMU Angle", color='green')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.subplot(5, 1, 2)
    plt.plot(time, optimized_angles_9, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.subplot(5, 1, 3)
    plt.plot(time, velocities_9, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.subplot(5, 1, 4)
    plt.plot(time, accelerations_9, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.subplot(5, 1, 5)
    plt.plot(time, jerks_9, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
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
    plt.yscale('log')
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
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Median Jerk (log scale)")
    # plt.title("Median Jerk for Different Optimizers")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #create box plots
    plt.figure(figsize=(7, 4))
    plt.boxplot([abs_j1, abs_j2, abs_j4, abs_j5, abs_j6, abs_j8, abs_j9], labels=labels, showfliers=False)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Distribution of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    print(f"best median jerk: {min(medians):.2e}, optimizer: {labels[medians.index(min(medians))]}")