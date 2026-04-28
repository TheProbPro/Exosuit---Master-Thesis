import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
file_emg = "Outputs/IMU_EMG_MoCap_Test/desired_angles.csv"   # has unix timestamps from time.time()
# file_emg = "Outputs/IMU_EMG_MoCap_Test/WithMotor.csv"   # has unix timestamps from time.time()
# file_emg = "Outputs/IMU_EMG_MoCap_Test/imu_angles.csv"   # has unix timestamps from time.time()

file_mocap = "Outputs/MoCap/EMGMocapTest.csv"   # has unix timestamps from time.time()
# file_mocap = "Outputs/MoCap/WithMotor.csv"   # has unix timestamps from time.time()
# file_mocap = "Outputs/MoCap/IMUEMGTest.csv"   # has unix timestamps from time.time()


# Change these column names to match your CSVs
file_emg_time_col = "Timestamp"
# file_emg_time_col = "Time"
# file_emg_time_col = "IMU_Timestamp"
file_emg_data_col = "Desired_Angle"
# file_emg_data_col = "Actual_Angle"
# file_emg_data_col = "Elbow_Angle"

file_mocap_time_col = "Timestamp"
file_mocap_data_col = "Elbow_Angle_Rad"
# file_mocap_data_col = "Elbow_Flexion_Rad"

CLOCK_OFFSET = 0.1753559112548828

if __name__ == "__main__":
    # Load EMG and MoCap data
    df_emg = pd.read_csv(file_emg)
    df_mocap = pd.read_csv(file_mocap)

    emg_timestamps = df_emg[file_emg_time_col].to_numpy()
    emg_timestamps = emg_timestamps - CLOCK_OFFSET  # Adjust for clock offset
    emg_angles = df_emg[file_emg_data_col].to_numpy()

    # plt.plot(emg_timestamps, emg_angles, label="EMG")
    # plt.show()

    print(f"Range of motion: {emg_angles.min():.2f} to {emg_angles.max():.2f} radians")

    mocap_timestamps = df_mocap[file_mocap_time_col].to_numpy()
    mocap_angles = df_mocap[file_mocap_data_col].to_numpy()

    print(f"Range of motion: {mocap_angles.min():.2f} to {mocap_angles.max():.2f} radians")

    print("EMG time range:", emg_timestamps[0], "→", emg_timestamps[-1])
    print("MoCap time range:", mocap_timestamps[0], "→", mocap_timestamps[-1])

    start_overlap = max(mocap_timestamps[0], emg_timestamps[0])
    end_overlap   = min(mocap_timestamps[-1], emg_timestamps[-1])

    print("Overlap:", start_overlap, "to", end_overlap)

    emg_mask = (emg_timestamps >= start_overlap) & (emg_timestamps <= end_overlap)
    mocap_mask = (mocap_timestamps >= start_overlap) & (mocap_timestamps <= end_overlap)

    emg_t = emg_timestamps[emg_mask]
    emg_y = emg_angles[emg_mask]

    mocap_t = mocap_timestamps[mocap_mask]
    mocap_y = mocap_angles[mocap_mask]

    for i in range(10):
        print(f"{emg_t[i]:.6f} | {mocap_t[i]:.6f}")

    print(f"EMG and MoCap timestamps: {emg_t[:10]}, {mocap_t[:10]}")

    emg_on_mocap = np.interp(mocap_t, emg_t, emg_y)

    for i in range(10):
        t = mocap_t[i]

        # find closest EMG samples used for interpolation
        idx = np.searchsorted(emg_t, t)

        t1 = emg_t[idx - 1]
        t2 = emg_t[idx]

        print(f"\nMoCap t: {t:.6f}")
        print(f"EMG neighbors: {t1:.6f}, {t2:.6f}")
        print(f"Interpolated EMG value: {emg_on_mocap[i]:.4f}")

    plt.figure(figsize=(12, 6))
    plt.title("Elbow Angle Comparison: EMG + IMU vs MoCap")
    plt.plot(mocap_t, mocap_y, label="MoCap")
    plt.plot(mocap_t, emg_on_mocap, label="EMG (aligned)")
    plt.xlabel("Time (s)")
    plt.ylabel("Elbow Angle (rad)")
    plt.legend()
    plt.show()

    a = mocap_y - np.mean(mocap_y)
    b = emg_on_mocap - np.mean(emg_on_mocap)

    corr = np.correlate(b, a, mode='full')
    lags = np.arange(-len(b)+1, len(a))

    lag = lags[np.argmax(corr)]
    lag_time = lag * (mocap_t[1] - mocap_t[0])

    print("Lag (seconds):", lag_time)