import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Files to be processed
Files = [
    "Outputs/ALLEMG/test1_results.csv",
    "Outputs/ALLEMG/test2_results.csv",
    "Outputs/ALLEMG/test3_results.csv",
    "Outputs/ALLEMG/test4_results.csv",
    "Outputs/ALLEMG/test5_results.csv",
    "Outputs/ALLEMG/test6_results.csv",
    "Outputs/ALLEMG/test7_results.csv",
    "Outputs/ALLEMG/test8_results.csv",
    "Outputs/ALLEMG/test9_results.csv",
    "Outputs/ALLEMG/test10_results.csv",
    "Outputs/IMUTests1/test1_results.csv",
    "Outputs/IMUTests1/test2_results.csv",
    "Outputs/IMUTests1/test3_results.csv",
    "Outputs/IMUTests1/test4_results.csv",
    "Outputs/IMUTests1/test5_results.csv",
    "Outputs/IMUTests1/test6_results.csv",
    "Outputs/IMUTests1/test7_results.csv"
]

Headers = [
    ["Time", "Net Activation, Desired Angle"],
    ["Time", "Net Activation", "Optimized Desired Angle", "Optimized Desired Angle EMG"],
    ["Time", "Net Activation", "Optimized Desired Angle", "Optimized Desired Angle EMG"],
    ["Time", "Net Activation", "Optimized Desired Angle", "Optimized Desired Angle EMG"],
    ["Time", "Net Activation", "Optimized Desired Angle", "Optimized Desired Angle EMG"],
    ["Time", "Net Activation", "Desired Angle"],
    ["Time", "Net Activation", "Desired Angle"],
    ["Time", "Net Activation", "Desired Angle"],
    ["Time", "Net Activation", "Optimized Desired Angle", "Optimized Desired Angle EMG"],
    ["Time", "Net Activation", "Optimized Desired Angle", "Optimized Desired Angle EMG"],
    ["Time", "Net Activation", "Desired Angle"],
    ["Time", "Time IMU", "Net Activation", "Desired Angle", "Desired Angle (IMU optimized)"],
    ["Time", "Time IMU", "Net Activation", "Desired Angle", "Desired Angle (IMU optimized)"],
    ["Time", "Time IMU", "Net Activation", "Desired Angle", "Desired Angle (IMU optimized)"],
    ["Time", "Time IMU", "Net Activation", "Desired Angle", "Desired Angle (IMU optimized)"],
    ["Time", "Net Activation", "IMU Angle", "Desired Angle", "IMU Velocity", "Net Activation (Differens)"],
    ["Time", "Desired Angle", "Net Activation", "IMU Angle", "IMU Velocity", "Net Activation (Differens)"]
]

def compute_derivatives(traj, dt):
    # Velocity (1st derivative)
    vel = np.gradient(traj, dt, axis=0)
    
    # Acceleration (2nd derivative)
    acc = np.gradient(vel, dt, axis=0)
    
    # Jerk (3rd derivative)
    jerk = np.gradient(acc, dt, axis=0)
    
    return vel, acc, jerk

show_trajectories = True

if __name__ == "__main__":
    dt_emg = 1/2000
    dt_imu = 1/148

    i = 1
    emg_1_jerks = []
    emg_2_jerks = []
    emg_2_jerks_emg = []
    emg_3_jerks = []
    emg_3_jerks_emg = []
    emg_4_jerks = []
    emg_4_jerks_emg = []
    emg_5_jerks = []
    emg_5_jerks_emg = []
    emg_6_jerks = []
    emg_7_jerks = []
    emg_8_jerks = []
    emg_9_jerks = []
    emg_9_jerks_emg = []
    emg_10_jerks = []
    emg_10_jerks_emg = []
    imu_1_jerks = []
    imu_2_jerks = []
    imu_2_jerks_emg = []
    imu_3_jerks = []
    imu_3_jerks_emg = []
    imu_4_jerks = []
    imu_4_jerks_emg = []
    imu_5_jerks = []
    imu_5_jerks_emg = []
    imu_6_jerks = []
    imu_6_jerks_emg = []
    imu_7_jerks = []
    imu_7_jerks_emg = []


    for file, header in zip(Files, Headers):
        if i == 1:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            emg_1_jerks = jerk

            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Desired Angle")
                plt.title("Desired Angle")
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                plt.title("Velocity")
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                plt.title("Acceleration")
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                plt.title("Jerk")
                plt.tight_layout()
                plt.show()

        elif i == 2:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Optimized Desired Angle"].to_numpy()
            desired_angle_emg = data["Optimized Desired Angle EMG"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            vel_emg, acc_emg, jerk_emg = compute_derivatives(desired_angle_emg, dt_emg)
            emg_2_jerks = jerk
            emg_2_jerks_emg = jerk_emg

            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_emg, label="Optimized Desired Angle EMG")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_emg, label="Velocity EMG")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_emg, label="Acceleration EMG")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_emg, label="Jerk EMG")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 3:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Optimized Desired Angle"].to_numpy()
            desired_angle_emg = data["Optimized Desired Angle EMG"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            vel_emg, acc_emg, jerk_emg = compute_derivatives(desired_angle_emg, dt_emg)
            emg_3_jerks = jerk
            emg_3_jerks_emg = jerk_emg

            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_emg, label="Optimized Desired Angle EMG")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_emg, label="Velocity EMG")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_emg, label="Acceleration EMG")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_emg, label="Jerk EMG")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 4:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Optimized Desired Angle"].to_numpy()
            desired_angle_emg = data["Optimized Desired Angle EMG"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            vel_emg, acc_emg, jerk_emg = compute_derivatives(desired_angle_emg, dt_emg)
            emg_4_jerks = jerk
            emg_4_jerks_emg = jerk_emg


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_emg, label="Optimized Desired Angle EMG")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_emg, label="Velocity EMG")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_emg, label="Acceleration EMG")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_emg, label="Jerk EMG")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()
        
        elif i == 5:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Optimized Desired Angle"].to_numpy()
            desired_angle_emg = data["Optimized Desired Angle EMG"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            vel_emg, acc_emg, jerk_emg = compute_derivatives(desired_angle_emg, dt_emg)
            emg_5_jerks = jerk
            emg_5_jerks_emg = jerk_emg



            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_emg, label="Optimized Desired Angle EMG")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_emg, label="Velocity EMG")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_emg, label="Acceleration EMG")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_emg, label="Jerk EMG")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 6:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            emg_6_jerks = jerk


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Desired Angle")
                plt.title("Desired Angle")
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                plt.title("Velocity")
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                plt.title("Acceleration")
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                plt.title("Jerk")
                plt.tight_layout()
                plt.show()

        elif i == 7:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            emg_7_jerks = jerk


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Desired Angle")
                plt.title("Desired Angle")
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                plt.title("Velocity")
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                plt.title("Acceleration")
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                plt.title("Jerk")
                plt.tight_layout()
                plt.show()

        elif i == 8:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            emg_8_jerks = jerk


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Desired Angle")
                plt.title("Desired Angle")
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                plt.title("Velocity")
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                plt.title("Acceleration")
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                plt.title("Jerk")
                plt.tight_layout()
                plt.show()

        elif i == 9:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Optimized Desired Angle"].to_numpy()
            desired_angle_emg = data["Optimized Desired Angle EMG"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            vel_emg, acc_emg, jerk_emg = compute_derivatives(desired_angle_emg, dt_emg)
            emg_9_jerks = jerk
            emg_9_jerks_emg = jerk_emg

            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_emg, label="Optimized Desired Angle EMG")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_emg, label="Velocity EMG")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_emg, label="Acceleration EMG")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_emg, label="Jerk EMG")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 10:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Optimized Desired Angle"].to_numpy()
            desired_angle_emg = data["Optimized Desired Angle EMG"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_emg)
            vel_emg, acc_emg, jerk_emg = compute_derivatives(desired_angle_emg, dt_emg)
            emg_10_jerks = jerk
            emg_10_jerks_emg = jerk_emg


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_emg, label="Optimized Desired Angle EMG")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_emg, label="Velocity EMG")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_emg, label="Acceleration EMG")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_emg, label="Jerk EMG")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 11:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            imu_1_jerks = jerk


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Desired Angle")
                plt.title("Desired Angle")
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                plt.title("Velocity")
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                plt.title("Acceleration")
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                plt.title("Jerk")
                plt.tight_layout()
                plt.show()

        elif i == 12:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()
            desired_angle_imu = data["Desired Angle (IMU optimized)"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            vel_imu, acc_imu, jerk_imu = compute_derivatives(desired_angle_imu, dt_imu)
            imu_2_jerks = jerk
            imu_2_jerks_emg = jerk_emg


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_imu, label="Optimized Desired Angle IMU")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_imu, label="Velocity IMU")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_imu, label="Acceleration IMU")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_imu, label="Jerk IMU")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 13:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()
            desired_angle_imu = data["Desired Angle (IMU optimized)"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            vel_imu, acc_imu, jerk_imu = compute_derivatives(desired_angle_imu, dt_imu)
            imu_3_jerks = jerk
            imu_3_jerks_emg = jerk_emg

            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_imu, label="Optimized Desired Angle IMU")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_imu, label="Velocity IMU")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_imu, label="Acceleration IMU")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_imu, label="Jerk IMU")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 14:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()
            desired_angle_imu = data["Desired Angle (IMU optimized)"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            vel_imu, acc_imu, jerk_imu = compute_derivatives(desired_angle_imu, dt_imu)
            imu_4_jerks = jerk
            imu_4_jerks_emg = jerk_emg


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_imu, label="Optimized Desired Angle IMU")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_imu, label="Velocity IMU")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_imu, label="Acceleration IMU")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_imu, label="Jerk IMU")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 15:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()
            desired_angle_imu = data["Desired Angle (IMU optimized)"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            vel_imu, acc_imu, jerk_imu = compute_derivatives(desired_angle_imu, dt_imu)
            imu_5_jerks = jerk
            imu_5_jerks_emg = jerk_emg


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_imu, label="Optimized Desired Angle IMU")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_imu, label="Velocity IMU")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_imu, label="Acceleration IMU")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_imu, label="Jerk IMU")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 16:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()
            desired_angle_imu = data["IMU Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            vel_imu, acc_imu, jerk_imu = compute_derivatives(desired_angle_imu, dt_imu)
            imu_6_jerks = jerk
            imu_6_jerks_emg = jerk_emg



            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_imu, label="Optimized Desired Angle IMU")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_imu, label="Velocity IMU")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_imu, label="Acceleration IMU")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_imu, label="Jerk IMU")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif i == 17:
            data = pd.read_csv(file)

            # time = data["Time"].to_numpy()
            net_a = data["Net Activation"].to_numpy()
            desired_angle = data["Desired Angle"].to_numpy()
            desired_angle_imu = data["IMU Angle"].to_numpy()

            vel, acc, jerk = compute_derivatives(desired_angle, dt_imu)
            vel_imu, acc_imu, jerk_imu = compute_derivatives(desired_angle_imu, dt_imu)
            imu_7_jerks = jerk
            imu_7_jerks_emg = jerk_emg


            if show_trajectories:
                plt.figure(figsize=(12, 8))
                plt.subplot(4, 1, 1)
                plt.plot(desired_angle, label="Optimized Desired Angle")
                # plt.plot(desired_angle_imu, label="Optimized Desired Angle IMU")
                plt.title("Desired Angle")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(vel, label="Velocity")
                # plt.plot(vel_imu, label="Velocity IMU")
                plt.title("Velocity")
                plt.legend()
                plt.subplot(4, 1, 3)
                plt.plot(acc, label="Acceleration")
                # plt.plot(acc_imu, label="Acceleration IMU")
                plt.title("Acceleration")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(jerk, label="Jerk")
                # plt.plot(jerk_imu, label="Jerk IMU")
                plt.title("Jerk")
                plt.legend()
                plt.tight_layout()
                plt.show()
        
        i += 1

    emg_1_mean_jerk = np.mean(np.abs(emg_1_jerks))
    emg_1_median_jerk = np.median(np.abs(emg_1_jerks))
    emg_1_sigma = np.std(np.abs(emg_1_jerks))
    emg_1_max_jerk = np.max(np.abs(emg_1_jerks))

    emg_2_mean_jerk = np.mean(np.abs(emg_2_jerks))
    emg_2_median_jerk = np.median(np.abs(emg_2_jerks))
    emg_2_sigma = np.std(np.abs(emg_2_jerks))
    emg_2_max_jerk = np.max(np.abs(emg_2_jerks))

    emg_2_mean_jerk_emg = np.mean(np.abs(emg_2_jerks_emg))
    emg_2_median_jerk_emg = np.median(np.abs(emg_2_jerks_emg))
    emg_2_sigma_emg = np.std(np.abs(emg_2_jerks_emg))
    emg_2_max_jerk_emg = np.max(np.abs(emg_2_jerks_emg))

    emg_3_mean_jerk = np.mean(np.abs(emg_3_jerks))
    emg_3_median_jerk = np.median(np.abs(emg_3_jerks))
    emg_3_sigma = np.std(np.abs(emg_3_jerks))
    emg_3_max_jerk = np.max(np.abs(emg_3_jerks))

    emg_3_mean_jerk_emg = np.mean(np.abs(emg_3_jerks_emg))
    emg_3_median_jerk_emg = np.median(np.abs(emg_3_jerks_emg))
    emg_3_sigma_emg = np.std(np.abs(emg_3_jerks_emg))
    emg_3_max_jerk_emg = np.max(np.abs(emg_3_jerks_emg))

    emg_4_mean_jerk = np.mean(np.abs(emg_4_jerks))
    emg_4_median_jerk = np.median(np.abs(emg_4_jerks))
    emg_4_sigma = np.std(np.abs(emg_4_jerks))
    emg_4_max_jerk = np.max(np.abs(emg_4_jerks))

    emg_4_mean_jerk_emg = np.mean(np.abs(emg_4_jerks_emg))
    emg_4_median_jerk_emg = np.median(np.abs(emg_4_jerks_emg))
    emg_4_sigma_emg = np.std(np.abs(emg_4_jerks_emg))
    emg_4_max_jerk_emg = np.max(np.abs(emg_4_jerks_emg))
    
    emg_5_mean_jerk = np.mean(np.abs(emg_5_jerks))
    emg_5_median_jerk = np.median(np.abs(emg_5_jerks))
    emg_5_sigma = np.std(np.abs(emg_5_jerks))
    emg_5_max_jerk = np.max(np.abs(emg_5_jerks))
    
    emg_5_mean_jerk_emg = np.mean(np.abs(emg_5_jerks_emg))
    emg_5_median_jerk_emg = np.median(np.abs(emg_5_jerks_emg))
    emg_5_sigma_emg = np.std(np.abs(emg_5_jerks_emg))
    emg_5_max_jerk_emg = np.max(np.abs(emg_5_jerks_emg))
    
    emg_6_mean_jerk = np.mean(np.abs(emg_6_jerks))
    emg_6_median_jerk = np.median(np.abs(emg_6_jerks))
    emg_6_sigma = np.std(np.abs(emg_6_jerks))
    emg_6_max_jerk = np.max(np.abs(emg_6_jerks))

    emg_7_mean_jerk = np.mean(np.abs(emg_7_jerks))
    emg_7_median_jerk = np.median(np.abs(emg_7_jerks))
    emg_7_sigma = np.std(np.abs(emg_7_jerks))
    emg_7_max_jerk = np.max(np.abs(emg_7_jerks))
    
    emg_8_mean_jerk = np.mean(np.abs(emg_8_jerks))
    emg_8_median_jerk = np.median(np.abs(emg_8_jerks))
    emg_8_sigma = np.std(np.abs(emg_8_jerks))
    emg_8_max_jerk = np.max(np.abs(emg_8_jerks))
    
    emg_9_mean_jerk = np.mean(np.abs(emg_9_jerks))
    emg_9_median_jerk = np.median(np.abs(emg_9_jerks))
    emg_9_sigma = np.std(np.abs(emg_9_jerks))
    emg_9_max_jerk = np.max(np.abs(emg_9_jerks))
    
    emg_9_mean_jerk_emg = np.mean(np.abs(emg_9_jerks_emg))
    emg_9_median_jerk_emg = np.median(np.abs(emg_9_jerks_emg))
    emg_9_sigma_emg = np.std(np.abs(emg_9_jerks_emg))
    emg_9_max_jerk_emg = np.max(np.abs(emg_9_jerks_emg))
    
    emg_10_mean_jerk = np.mean(np.abs(emg_10_jerks))
    emg_10_median_jerk = np.median(np.abs(emg_10_jerks))
    emg_10_sigma = np.std(np.abs(emg_10_jerks))
    emg_10_max_jerk = np.max(np.abs(emg_10_jerks))
    
    emg_10_mean_jerk_emg = np.mean(np.abs(emg_10_jerks_emg))
    emg_10_median_jerk_emg = np.median(np.abs(emg_10_jerks_emg))
    emg_10_sigma_emg = np.std(np.abs(emg_10_jerks_emg))
    emg_10_max_jerk_emg = np.max(np.abs(emg_10_jerks_emg))
    
    imu_1_mean_jerk = np.mean(np.abs(imu_1_jerks))
    imu_1_median_jerk = np.median(np.abs(imu_1_jerks))
    imu_1_sigma = np.std(np.abs(imu_1_jerks))
    imu_1_max_jerk = np.max(np.abs(imu_1_jerks))
    
    imu_2_mean_jerk = np.mean(np.abs(imu_2_jerks))
    imu_2_median_jerk = np.median(np.abs(imu_2_jerks))
    imu_2_sigma = np.std(np.abs(imu_2_jerks))
    imu_2_max_jerk = np.max(np.abs(imu_2_jerks))
    
    imu_2_mean_jerk_emg = np.mean(np.abs(imu_2_jerks_emg))
    imu_2_median_jerk_emg = np.median(np.abs(imu_2_jerks_emg))
    imu_2_sigma_emg = np.std(np.abs(imu_2_jerks_emg))
    imu_2_max_jerk_emg = np.max(np.abs(imu_2_jerks_emg))
    
    imu_3_mean_jerk = np.mean(np.abs(imu_3_jerks))
    imu_3_median_jerk = np.median(np.abs(imu_3_jerks))
    imu_3_sigma = np.std(np.abs(imu_3_jerks))
    imu_3_max_jerk = np.max(np.abs(imu_3_jerks))
    
    imu_3_mean_jerk_emg = np.mean(np.abs(imu_3_jerks_emg))
    imu_3_median_jerk_emg = np.median(np.abs(imu_3_jerks_emg))
    imu_3_sigma_emg = np.std(np.abs(imu_3_jerks_emg))
    imu_3_max_jerk_emg = np.max(np.abs(imu_3_jerks_emg))
    
    imu_4_mean_jerk = np.mean(np.abs(imu_4_jerks))
    imu_4_median_jerk = np.median(np.abs(imu_4_jerks))
    imu_4_sigma = np.std(np.abs(imu_4_jerks))
    imu_4_max_jerk = np.max(np.abs(imu_4_jerks))
    
    imu_4_mean_jerk_emg = np.mean(np.abs(imu_4_jerks_emg))
    imu_4_median_jerk_emg = np.median(np.abs(imu_4_jerks_emg))
    imu_4_sigma_emg = np.std(np.abs(imu_4_jerks_emg))
    imu_4_max_jerk_emg = np.max(np.abs(imu_4_jerks_emg))

    imu_5_mean_jerk = np.mean(np.abs(imu_5_jerks))
    imu_5_median_jerk = np.median(np.abs(imu_5_jerks))
    imu_5_sigma = np.std(np.abs(imu_5_jerks))
    imu_5_max_jerk = np.max(np.abs(imu_5_jerks))
    
    imu_5_mean_jerk_emg = np.mean(np.abs(imu_5_jerks_emg))
    imu_5_median_jerk_emg = np.median(np.abs(imu_5_jerks_emg))
    imu_5_sigma_emg = np.std(np.abs(imu_5_jerks_emg))
    imu_5_max_jerk_emg = np.max(np.abs(imu_5_jerks_emg))

    imu_6_mean_jerk = np.mean(np.abs(imu_6_jerks))
    imu_6_median_jerk = np.median(np.abs(imu_6_jerks))
    imu_6_sigma = np.std(np.abs(imu_6_jerks))
    imu_6_max_jerk = np.max(np.abs(imu_6_jerks))

    imu_6_mean_jerk_emg = np.mean(np.abs(imu_6_jerks_emg))
    imu_6_median_jerk_emg = np.median(np.abs(imu_6_jerks_emg))
    imu_6_sigma_emg = np.std(np.abs(imu_6_jerks_emg))
    imu_6_max_jerk_emg = np.max(np.abs(imu_6_jerks_emg))

    imu_7_mean_jerk = np.mean(np.abs(imu_7_jerks))
    imu_7_median_jerk = np.median(np.abs(imu_7_jerks))
    imu_7_sigma = np.std(np.abs(imu_7_jerks))
    imu_7_max_jerk = np.max(np.abs(imu_7_jerks))

    imu_7_mean_jerk_emg = np.mean(np.abs(imu_7_jerks_emg))
    imu_7_median_jerk_emg = np.median(np.abs(imu_7_jerks_emg))
    imu_7_sigma_emg = np.std(np.abs(imu_7_jerks_emg))
    imu_7_max_jerk_emg = np.max(np.abs(imu_7_jerks_emg))

    mean_jerks = {
        "emg_1": emg_1_mean_jerk,
        "emg_2": emg_2_mean_jerk,
        # "emg_2_emg": emg_2_mean_jerk_emg,
        "emg_3": emg_3_mean_jerk,
        # "emg_3_emg": emg_3_mean_jerk_emg,
        "emg_4": emg_4_mean_jerk,
        # "emg_4_emg": emg_4_mean_jerk_emg,
        "emg_5": emg_5_mean_jerk,
        # "emg_5_emg": emg_5_mean_jerk_emg,
        # "emg_6": emg_6_mean_jerk,
        # "emg_7": emg_7_mean_jerk,
        "emg_8": emg_8_mean_jerk,
        "emg_9": emg_9_mean_jerk,
        # "emg_9_emg": emg_9_mean_jerk_emg,
        # "emg_10": emg_10_mean_jerk,
        # "emg_10_emg": emg_10_mean_jerk_emg,
        "imu_1": imu_1_mean_jerk,
        "imu_2": imu_2_mean_jerk,
        # "imu_2_imu": imu_2_mean_jerk_emg,
        "imu_3": imu_3_mean_jerk,
        # "imu_3_imu": imu_3_mean_jerk_emg,
        "imu_4": imu_4_mean_jerk,
        # "imu_4_imu": imu_4_mean_jerk_emg,
        "imu_5": imu_5_mean_jerk,
        # "imu_5_imu": imu_5_mean_jerk_emg,
        "imu_6": imu_6_mean_jerk,
        # "imu_6_imu": imu_6_mean_jerk_emg,
        "imu_7": imu_7_mean_jerk,
        # "imu_7_imu": imu_7_mean_jerk_emg
    }

    median_jerks = {
        "emg_1": emg_1_median_jerk,
        "emg_2": emg_2_median_jerk,
        # "emg_2_emg": emg_2_median_jerk_emg,
        "emg_3": emg_3_median_jerk,
        # "emg_3_emg": emg_3_median_jerk_emg,
        "emg_4": emg_4_median_jerk,
        # "emg_4_emg": emg_4_median_jerk_emg,
        "emg_5": emg_5_median_jerk,
        # "emg_5_emg": emg_5_median_jerk_emg,
        # "emg_6": emg_6_median_jerk,
        # "emg_7": emg_7_median_jerk,
        "emg_8": emg_8_median_jerk,
        "emg_9": emg_9_median_jerk,
        # "emg_9_emg": emg_9_median_jerk_emg,
        # "emg_10": emg_10_median_jerk,
        # "emg_10_emg": emg_10_median_jerk_emg,
        "imu_1": imu_1_median_jerk,
        "imu_2": imu_2_median_jerk,
        # "imu_2_imu": imu_2_median_jerk_emg,
        "imu_3": imu_3_median_jerk,
        # "imu_3_imu": imu_3_median_jerk_emg,
        "imu_4": imu_4_median_jerk,
        # "imu_4_imu": imu_4_median_jerk_emg,
        "imu_5": imu_5_median_jerk,
        # "imu_5_imu": imu_5_median_jerk_emg,
        "imu_6": imu_6_median_jerk,
        # "imu_6_imu": imu_6_median_jerk_emg,
        "imu_7": imu_7_median_jerk,
        # "imu_7_imu": imu_7_median_jerk_emg
    }

    sigmas = {
        "emg_1": emg_1_sigma,
        "emg_2": emg_2_sigma,
        # "emg_2_emg": emg_2_sigma_emg,
        "emg_3": emg_3_sigma,
        # "emg_3_emg": emg_3_sigma_emg,
        "emg_4": emg_4_sigma,
        # "emg_4_emg": emg_4_sigma_emg,
        "emg_5": emg_5_sigma,
        # "emg_5_emg": emg_5_sigma_emg,
        # "emg_6": emg_6_sigma,
        # "emg_7": emg_7_sigma,
        "emg_8": emg_8_sigma,
        "emg_9": emg_9_sigma,
        # "emg_9_emg": emg_9_sigma_emg,
        # "emg_10": emg_10_sigma,
        # "emg_10_emg": emg_10_sigma_emg,
        "imu_1": imu_1_sigma,
        "imu_2": imu_2_sigma,
        # "imu_2_imu": imu_2_sigma_emg,
        "imu_3": imu_3_sigma,
        # "imu_3_imu": imu_3_sigma_emg,
        "imu_4": imu_4_sigma,
        # "imu_4_imu": imu_4_sigma_emg,
        "imu_5": imu_5_sigma,
        # "imu_5_imu": imu_5_sigma_emg,
        "imu_6": imu_6_sigma,
        # "imu_6_imu": imu_6_sigma_emg,
        "imu_7": imu_7_sigma,
        # "imu_7_imu": imu_7_sigma_emg
    }

    max_jerks = {
        "emg_1": emg_1_max_jerk,
        "emg_2": emg_2_max_jerk,
        # "emg_2_emg": emg_2_max_jerk_emg,
        "emg_3": emg_3_max_jerk,
        # "emg_3_emg": emg_3_max_jerk_emg,
        "emg_4": emg_4_max_jerk,
        # "emg_4_emg": emg_4_max_jerk_emg,
        "emg_5": emg_5_max_jerk,
        # "emg_5_emg": emg_5_max_jerk_emg,
        # "emg_6": emg_6_max_jerk,
        # "emg_7": emg_7_max_jerk,
        "emg_8": emg_8_max_jerk,
        "emg_9": emg_9_max_jerk,
        # "emg_9_emg": emg_9_max_jerk_emg,
        # "emg_10": emg_10_max_jerk,
        # "emg_10_emg": emg_10_max_jerk_emg,
        "imu_1": imu_1_max_jerk,
        "imu_2": imu_2_max_jerk,
        # "imu_2_emg": imu_2_max_jerk_emg,
        "imu_3": imu_3_max_jerk,
        # "imu_3_emg": imu_3_max_jerk_emg,
        "imu_4": imu_4_max_jerk,
        # "imu_4_emg": imu_4_max_jerk_emg,
        "imu_5": imu_5_max_jerk,
        # "imu_5_emg": imu_5_max_jerk_emg,
        "imu_6": imu_6_max_jerk,
        # "imu_6_emg": imu_6_max_jerk_emg,
        "imu_7": imu_7_max_jerk,
        # "imu_7_emg": imu_7_max_jerk_emg
    }

    print("=====================================================================")

    smallest_name = min(mean_jerks, key=mean_jerks.get)
    smallest_value = mean_jerks[smallest_name]
    print("Smallest mean jerk:", smallest_name, "=", smallest_value)

    for name, value in sorted(mean_jerks.items(), key=lambda x: x[1]):
        print(f"{name}: {value}")

    print("=====================================================================")

    smallest_name = min(median_jerks, key=median_jerks.get)
    smallest_value = median_jerks[smallest_name]
    print("Smallest median jerk:", smallest_name, "=", smallest_value)

    for name, value in sorted(median_jerks.items(), key=lambda x: x[1]):
        print(f"{name}: {value}")

    print("=====================================================================")

    smallest_name = min(sigmas, key=sigmas.get)
    smallest_value = sigmas[smallest_name]
    print("Smallest sigma:", smallest_name, "=", smallest_value)

    for name, value in sorted(sigmas.items(), key=lambda x: x[1]):
        print(f"{name}: {value}")

    print("=====================================================================")

    smallest_name = min(max_jerks, key=max_jerks.get)
    smallest_value = max_jerks[smallest_name]
    print("Smallest max jerk:", smallest_name, "=", smallest_value)

    for name, value in sorted(max_jerks.items(), key=lambda x: x[1]):
        print(f"{name}: {value}")

    #==============================================================================

    # Make a bar plot
    trajectory_names = [
        "Raw EMG", 
        "Optimizer 1 (EMG)", 
        # "Optimizer 1 (EMG) - prediction",
        "Optimizer 2 (EMG)", 
        # "Optimizer 2 (EMG) - prediction",
        "Optimizer 3 (EMG)", 
        # "Optimizer 3 (EMG) - prediction",
        "Optimizer 4 (EMG)", 
        # "Optimizer 4 (EMG) - prediction",
        # "pDMP",
        # "pDMP coupled",
        "pDMP Omega", 
        "Optimizer 5 (EMG)", 
        # "Optimizer 5 (EMG) - prediction",
        # "Optimzer 7 (EMG)",
        "Raw IMU",  
        "Optimizer 1 (IMU)", 
        # "Optimizer 1 (IMU) - prediction",
        "Optimizer 2 (IMU)", 
        # "Optimizer 2 (IMU) - prediction",
        "Optimizer 3 (IMU)", 
        # "Optimizer 3 (IMU) - prediction",
        "Optimizer 4 (IMU)", 
        # "Optimizer 4 (IMU) - prediction",
        "Optimizer 5 (IMU)", 
        # "Optimizer 5 (IMU) - prediction",
        "Optimizer 6 (IMU)",
        # "Optimizer 6 (IMU) - prediction"
    ]

    x = np.arange(len(trajectory_names))

    plt.figure(figsize=(12, 8))
    plt.bar(x, [mean_jerks[name] for name in mean_jerks], yerr=[sigmas[name] for name in sigmas], capsize=5)
    # plt.bar(x, [median_jerks[name] for name in median_jerks], yerr=[sigmas[name] for name in sigmas], capsize=5)
    plt.scatter(x, [max_jerks[name] for name in max_jerks], color='red', label='Max Jerk')
    plt.yscale("log")
    plt.xticks(x, trajectory_names, rotation=45)
    plt.ylabel("Jerk")
    plt.title("Trajectory smoothness comparison (mean jerk)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    # plt.bar(x, [mean_jerks[name] for name in mean_jerks], yerr=[sigmas[name] for name in sigmas], capsize=5)
    plt.bar(x, [median_jerks[name] for name in median_jerks], yerr=[sigmas[name] for name in sigmas], capsize=5)
    plt.scatter(x, [max_jerks[name] for name in max_jerks], color='red', label='Max Jerk')
    plt.yscale("log")
    plt.xticks(x, trajectory_names, rotation=45)
    plt.ylabel("Jerk")
    plt.title("Trajectory smoothness comparison (median jerk)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Make a box plot
    jerk_data = [
        np.abs(emg_1_jerks),
        np.abs(emg_2_jerks),
        # np.abs(emg_2_jerks_emg),
        np.abs(emg_3_jerks),
        # np.abs(emg_3_jerks_emg),
        np.abs(emg_4_jerks),
        # np.abs(emg_4_jerks_emg),
        np.abs(emg_5_jerks),
        # np.abs(emg_5_jerks_emg),
        # np.abs(emg_6_jerks),
        # np.abs(emg_7_jerks),
        np.abs(emg_8_jerks),
        np.abs(emg_9_jerks),
        # np.abs(emg_9_jerks_emg),
        # np.abs(emg_10_jerks),
        # np.abs(emg_10_jerks_emg),
        np.abs(imu_1_jerks),
        np.abs(imu_2_jerks),
        # np.abs(imu_2_jerks_emg),
        np.abs(imu_3_jerks),
        # np.abs(imu_3_jerks_emg),
        np.abs(imu_4_jerks),
        # np.abs(imu_4_jerks_emg),
        np.abs(imu_5_jerks),
        # np.abs(imu_5_jerks_emg),
        np.abs(imu_6_jerks),
        # np.abs(imu_6_jerks_emg),
        np.abs(imu_7_jerks),
        # np.abs(imu_7_jerks_emg)
    ]

    plt.figure(figsize=(12, 8))
    plt.boxplot(jerk_data, tick_labels=trajectory_names)
    plt.yscale("log")
    plt.ylabel("Absolute jerk")
    plt.title("Distribution of jerk across trajectories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Save needed data to .CSV
    summary_data = []

    for name in mean_jerks:
        summary_data.append({
            "trajectory": name,
            "mean_jerk": mean_jerks[name],
            "median_jerk": median_jerks[name],
            "std_jerk": sigmas[name],
            "max_jerk": max_jerks[name]
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("Outputs/OptimizerStatistics/jerk_summary.csv", index=False)

    # TODO: Potentially save all the jerk data into new processed .csv files that can then be used for plotting later