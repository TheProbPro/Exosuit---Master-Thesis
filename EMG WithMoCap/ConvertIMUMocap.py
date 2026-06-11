import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
from pathlib import Path

# EMG processing imports
from SignalProcessing.IMUProcessing import IMUProcessing
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_IMU_optimizer, EMG_IMU_optimizer_2

SAVEPATH = "Outputs/Results/"

IMU_OPTIMIZERS = [
    "None",
    "optimizer_1",
    "optimizer_2",
    "optimizer_4",
    "optimizer_5_pd",
    "optimizer_6",
    "EMG_IMU_optimizer",
    "EMG_IMU_optimizer_2"
]

INPUT_MOCAP_DATA = [
    # "Outputs/MoCap/ExoIMUTest1_002.csv",
    "Outputs/MoCap/ExoIMUTest1_003.csv"
]

INPUT_EMG_DATA = [
    # "Outputs/IMUMocap2/emg_data.csv",
    "Outputs/IMUMocap3/emg_data.csv"
]

INPUT_IMU_DATA = [
    # "Outputs/IMUMocap2/imu_data.csv",
    "Outputs/IMUMocap3/imu_data.csv"
]

INPUT_IMU_CALIBRATION_DATA = [
    # "Outputs/IMUMocap2/imu_calibration_data.csv",
    "Outputs/IMUMocap3/imu_calibration_data.csv"
]

def process_mocap(file):
    # Read start time
    with open(file, "r") as f:
        first_line = f.readline()

    parts = [p.strip() for p in first_line.split(",")]

    # Find the index of "Capture Start Time"
    idx = parts.index("Capture Start Time")

    # The value is right after it
    start_time = parts[idx + 1]

    print("Start time:", start_time)

    # Load the CSV
    df = pd.read_csv(
        file,
        skiprows=3,        # skip metadata before row 4
        header=[0, 3]      # row 4 (markers) + row 7 (axes)
    )

    # load data into numpy arrays
    print(f"Columns in the CSV: {df.columns.tolist()}")

    # arrange data into relevant numpy arrays
    timestamps = df[("Name", "Time (Seconds)")].to_numpy()
    forearm_lower = df["ForeArm:Lower"][["X", "Y", "Z"]].to_numpy()
    forearm_upper = df["ForeArm:Upper"][["X", "Y", "Z"]].to_numpy()
    upperarm_middle = df["UpperArm:Middle"][["X", "Y", "Z"]].to_numpy()
    upperarm_upper = df["UpperArm:Upper"][["X", "Y", "Z"]].to_numpy()
    upperarm_elbow = df["UpperArm:Elbow"][["X", "Y", "Z"]].to_numpy()

    # Convert timestamps into unix timestamps
    # Fix format (replace dots in time part)
    start_time_clean = start_time.replace(".", ":", 2)  # only replace first two dots
    print("Cleaned start time:", start_time_clean)
    
    # Parse to datetime
    start_dt = datetime.strptime(start_time_clean, "%Y-%m-%d %H:%M:%S.%f")
    # If you know this should be PM rather than AM
    start_dt = start_dt + timedelta(hours=12)
    print("Parsed start datetime:", start_dt)
    
    # Convert to unix timestamp
    start_unix = start_dt.timestamp()
    print("Start time as unix timestamp:", start_unix)

    # add start_unix to relative timestamps to get absolute unix timestamps
    absolute_timestamps = start_unix + timestamps

    # Calculate elbow angle using upperarm_elbow, upperarm_upper, and forearm_lower
    # Calculate vectors
    v_upper = upperarm_upper - upperarm_elbow
    v_forearm = forearm_lower - upperarm_elbow

    # Calculate dot product
    dot = np.sum(v_upper * v_forearm, axis=1)

    # Calculate norms
    norm_upper = np.linalg.norm(v_upper, axis=1)
    norm_forearm = np.linalg.norm(v_forearm, axis=1)

    # Avoid divide-by-zero
    cos_theta = dot / (norm_upper * norm_forearm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Angle in radians
    elbow_angle_rad = np.arccos(cos_theta)

    # Angle in degrees
    elbow_angle_deg = np.degrees(elbow_angle_rad)
    elbow_flexion_deg = 180 - elbow_angle_deg  # Assuming full extension is 0 degrees
    elbow_flexion_rad = np.radians(elbow_flexion_deg)

    # print("Calculated elbow angles (degrees):", elbow_angle_deg)
    print("Calculated elbow flexion angles (degrees):", elbow_flexion_deg)

    return absolute_timestamps, elbow_angle_rad, elbow_flexion_rad, timestamps, start_time

def parse_emg_string(s):
    s = s.replace('[', '').replace(']', '')
    return np.fromstring(s, sep=' ')

def process_emg(file):
    EMG_FS = 2000  # Hz
    dt = 1 / EMG_FS

    # # Extract start time
    # with open(file, "r") as f:
    #     lines = f.readlines()

    # start_time_line = lines[4].strip()
    # start_time = start_time_line.split(",")[1].strip()
    
    # print("EMG Start time:", start_time)
    
    # load csv
    df = pd.read_csv(file)

    print(df.head())
    print([repr(col) for col in df.columns])

    # Load data into numpy arrays
    timestamps = df["timestamp_emg"].to_numpy()
    # EMG_data = df["emg_data"].to_numpy()
    EMG_data = df["emg_data"].apply(parse_emg_string).to_numpy()
    EMG_data = np.vstack(EMG_data)  # convert to (N,2)

    # Save emg_data[x, 0] as BicepValues and emg_data[x, 1] as TricepValues¨
    BicepValues = EMG_data[:, 0]
    TricepValues = EMG_data[:, 1]

    # Convert EMG timestamps to absolute unix timestamps
    # Convert timestamps into unix timestamps
    # Fix format (replace dots in time part)
    # start_time_clean = start_time.replace(".", ":", 2)  # only replace first two dots
    # print("Cleaned start time:", start_time_clean)
    
    # # Parse to datetime
    # start_dt = datetime.strptime(start_time, "%d %b %Y %H:%M:%S %z")
    # start_unix = start_dt.timestamp()
    # absolute_timestamps = start_unix + timestamps
    # print("Start time as unix timestamp:", start_unix)

    # # add start_unix to relative timestamps to get absolute unix timestamps
    # absolute_timestamps = start_unix + timestamps

    # emg_absolute_timestamps = start_unix + np.arange(len(BicepValues)) * dt

    # Process EMG signals
    # EMG Processing parameters
    THETA_MIN = np.deg2rad(0)
    THETA_MAX = np.deg2rad(140)
    THETA_RANGE = THETA_MAX - THETA_MIN
    USER_NAME = "VictorBNielsen"
    filter_bicep = rt_filtering(EMG_FS, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_FS, 450, 20, 2)
    net_a_lowpass = rt_desired_Angle_lowpass(EMG_FS, lp_cutoff=2, order=2)
    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    bicep_rms_queue = queue.Queue(maxsize=50)
    tricep_rms_queue = queue.Queue(maxsize=50)
    filtered_net_a_values = []

    for bicep, tricep in zip(BicepValues, TricepValues):
        filtered_bicep = filter_bicep.bandpass(np.array([bicep]))[0]
        filtered_tricep = filter_tricep.bandpass(np.array([tricep]))[0]

        if bicep_rms_queue.full():
            bicep_rms_queue.get()
        bicep_rms_queue.put(filtered_bicep)
        if tricep_rms_queue.full():
            tricep_rms_queue.get()
        tricep_rms_queue.put(filtered_tricep)

        bicep_rms = np.sqrt(np.mean(np.array(list(bicep_rms_queue.queue))**2))
        tricep_rms = np.sqrt(np.mean(np.array(list(tricep_rms_queue.queue))**2))

        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(bicep_rms))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(tricep_rms))[0])

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]
        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])
        filtered_net_a_values.append(filtered_net_a)

    return filtered_net_a_values, timestamps

def parse_imu_string(s):
    # Remove brackets and split
    s = s.replace('[', '').replace(']', '')
    values = np.fromstring(s, sep=' ')
    return values

def process_imu_data(file, calibration_file):
    imuProcessor = IMUProcessing()
    imu_lowpass = rt_desired_Angle_lowpass(148, lp_cutoff=5)

    # Load calibration data
    calib_df = pd.read_csv(calibration_file)
    imu_zero = calib_df["IMU_zero"].to_numpy()
    imu_gyro_bias_upper = calib_df["IMU_gyro_bias_upper"].to_numpy()
    imu_gyro_bias_lower = calib_df["IMU_gyro_bias_lower"].to_numpy()

    imuProcessor.set_gyro_bias(imu_gyro_bias_upper, imu_gyro_bias_lower)
    imuProcessor.set_zero(imu_zero)

    # Load IMU data
    imu_df = pd.read_csv(file)
    timestamps = imu_df["timestamp_imu"].to_numpy()
    # imu_data = imu_df["imu_data"].to_numpy()
    imu_data = imu_df["imu_data"].apply(parse_imu_string).to_numpy()

    filtered_elbow_angles = []
    elbow_angles = []

    for data in imu_data:
        # Split imu data into acc, gyro, and mag (assuming they are concatenated in the string)
        # imu = np.asarray(data, dtype=float).reshape(-1)

        # Extract accelerometer and gyroscope data for upper and lower arm
        acc_upper = data[0:3]
        gyr_upper = data[3:6]
        acc_lower = data[18:21]
        gyr_lower = data[21:24]

        # Process imu data to get quaternions and elbow angle
        quat_upper, quat_lower = imuProcessor.calculate_quarternions(acc_upper, gyr_upper, acc_lower, gyr_lower)
        elbow_angle = np.deg2rad(imuProcessor.calculate_elbow_angle(quat_upper, quat_lower))
        elbow_angle_filtered = imu_lowpass.lowpass(np.atleast_1d(elbow_angle))[0]

        filtered_elbow_angles.append(elbow_angle_filtered)
        elbow_angles.append(elbow_angle)
    
    return filtered_elbow_angles, elbow_angles, timestamps

def synchronize_data(mocap_timestamps, imu_timestamps, emg_timestamps):
    start_overlap = max(mocap_timestamps[0], imu_timestamps[0], emg_timestamps[0])
    end_overlap   = min(mocap_timestamps[-1], imu_timestamps[-1], emg_timestamps[-1])

    emg_mask = (emg_timestamps >= start_overlap) & (emg_timestamps <= end_overlap)
    imu_mask = (imu_timestamps >= start_overlap) & (imu_timestamps <= end_overlap)
    mocap_mask = (mocap_timestamps >= start_overlap) & (mocap_timestamps <= end_overlap)

    return emg_mask, imu_mask, mocap_mask

def upward_crossings(t, y, threshold):
    idx = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    crossings = []

    for i in idx:
        # linear interpolation for sub-sample crossing time
        t_cross = t[i] + (threshold - y[i]) * (t[i+1] - t[i]) / (y[i+1] - y[i])
        crossings.append(t_cross)

    return np.array(crossings)

if __name__ == "__main__":
    IMU_dt = 1/148
    EMG_dt = 1/2000
    mocap_dt = 1/200

    THETA_MIN = np.deg2rad(0)
    THETA_MAX = np.deg2rad(140)

    USER_NAME = "VictorBNielsen"

    # CLOCK_OFFSET = 1.3   
    CLOCK_OFFSET = 1.4#1.4 # Set to 0 for now since we are using absolute timestamps

    interpreter = PMC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)

    for mocap_file, emg_file, imu_file, calib_file in zip(INPUT_MOCAP_DATA, INPUT_EMG_DATA, INPUT_IMU_DATA, INPUT_IMU_CALIBRATION_DATA):
        print(f"Processing MoCap file: {mocap_file}")
        mocap_timestamps, elbow_angle_rad, elbow_flexion_rad, mocap_relative_timestamps, mocap_start_time = process_mocap(mocap_file)

        print(f"Processing IMU file: {imu_file}")
        filtered_elbow_angles, elbow_angles, imu_timestamps = process_imu_data(imu_file, calib_file)

        print(f"Processing EMG file: {emg_file}")
        filtered_net_a_values, emg_timestamps = process_emg(emg_file)

        print("Synchronizing data...")
        # subtract time dif from
        # emg_timestamps = emg_timestamps - CLOCK_OFFSET * 2
        # imu_timestamps = imu_timestamps - CLOCK_OFFSET * 2
        # mocap_timestamps = mocap_timestamps + CLOCK_OFFSET * 2
        emg_timestamps = emg_timestamps - CLOCK_OFFSET
        imu_timestamps = imu_timestamps - CLOCK_OFFSET

        emg_mask, imu_mask, mocap_mask = synchronize_data(mocap_timestamps, imu_timestamps, emg_timestamps)

        emg_t = emg_timestamps[emg_mask]
        mocap_t = mocap_timestamps[mocap_mask]
        imu_t = imu_timestamps[imu_mask]

        print("First EMG timestamp:  ", emg_t[0])
        print("First IMU timestamp:  ", imu_t[0])
        print("First MoCap timestamp:", mocap_t[0])

        # emg_y = optimized_angle_values[emg_mask]
        emg_a = np.array(filtered_net_a_values)[emg_mask]
        mocap_y = elbow_flexion_rad[mocap_mask]
        imu_y = np.array(filtered_elbow_angles)[imu_mask]

        # Sample the corresponding EMG values to the corresponding IMU samples
        # Find indices where t_a would be inserted in t_b
        indices = np.searchsorted(emg_t, imu_t, side='right') - 1

        # Handle edge case: if t_a is earlier than first t_b
        indices[indices < 0] = 0

        # Get aligned samples
        # emg_y_aligned = emg_y[indices]
        emg_a_aligned = emg_a[indices]

        for optimizer in IMU_OPTIMIZERS:
            # Optimization parameters
            q = 0
            v = 0
            optimized_angle = 0
            delta_q_prev = 0
            a_prev = 0
            q_next = 0
            imu_prev = 0

            # Containers
            filtered_net_a_values = []
            optimized_angle_values = []

            if optimizer != "None":
                optimized_angle_values.append(q)

            for filtered_net_a, imu in zip(emg_a_aligned, imu_y):
                if optimizer == "None":
                    optimized_angle_values.append(imu)
                    # optimized_angle_values.append(interpreter.compute_angle(filtered_net_a))
                elif optimizer == "optimizer_1":
                    if emg_file == "Outputs/IMUMocap2/emg_data.csv":
                        k = 1.8 * np.pi
                    elif emg_file == "Outputs/IMUMocap3/emg_data.csv":
                        k = 1.8 * np.pi
                    optimized_angle = optimize_1(k, filtered_net_a, IMU_dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
                    # optimized_angle = optimize_1((2*np.pi), filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
                    optimized_angle_values.append(optimized_angle)
                elif optimizer == "optimizer_2":
                    if emg_file == "Outputs/IMUMocap2/emg_data.csv":
                        k = 4 * np.pi
                    elif emg_file == "Outputs/IMUMocap3/emg_data.csv":
                        k = 4 * np.pi
                    optimized_angle = optimize_2(k, filtered_net_a, IMU_dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
                    optimized_angle_values.append(optimized_angle)
                elif optimizer == "optimizer_4":
                    if emg_file == "Outputs/IMUMocap2/emg_data.csv":
                        k = 1.8 * np.pi
                    elif emg_file == "Outputs/IMUMocap3/emg_data.csv":
                        k = 1.8 * np.pi
                    optimized_angle, delta_q_prev = optimize_4(k, filtered_net_a, IMU_dt, optimized_angle_values[-1], delta_q_prev, THETA_MIN, THETA_MAX)
                    optimized_angle_values.append(optimized_angle)
                elif optimizer == "optimizer_5_pd":
                    if emg_file == "Outputs/IMUMocap2/emg_data.csv":
                        k = 18
                        b = 0.02
                    elif emg_file == "Outputs/IMUMocap3/emg_data.csv":
                        k = 18
                        b = 0.02
                    optimized_angle, v = optimize_5_pd(filtered_net_a, v, IMU_dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, k, b)#4, 0.01)
                    optimized_angle_values.append(optimized_angle)
                elif optimizer == "optimizer_6":
                    if emg_file == "Outputs/IMUMocap2/emg_data.csv":
                        k = 2 * 10 * np.pi
                        b = 5.0
                    elif emg_file == "Outputs/IMUMocap3/emg_data.csv":
                        k = 2 * 10 * np.pi
                        b = 8.0
                    optimized_angle, v, acc = optimizer_6(filtered_net_a, v, IMU_dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, b, k)
                    optimized_angle_values.append(optimized_angle)
                elif optimizer == "EMG_IMU_optimizer":
                    d_a = filtered_net_a - a_prev
                    a_prev = filtered_net_a
                    omega = (imu - imu_prev)
                    imu_prev = imu
                    q_next, v, acc = EMG_IMU_optimizer(filtered_net_a, d_a, v, omega, 8, 2, 5, 2, q_next, imu, THETA_MIN, THETA_MAX, np.pi, IMU_dt)
                    optimized_angle_values.append(q_next)
                elif optimizer == "EMG_IMU_optimizer_2":
                    d_a = filtered_net_a - a_prev
                    a_prev = filtered_net_a
                    omega = (imu - imu_prev)
                    imu_prev = imu
                    q_next, v = EMG_IMU_optimizer_2(filtered_net_a, d_a, omega, 8, 2, q_next, THETA_MIN, THETA_MAX, np.pi, IMU_dt)
                    optimized_angle_values.append(q_next)
                else:
                    print(f"Unknown optimizer: {optimizer}")

            if optimizer != "None":
                optimized_angle_values.remove(optimized_angle_values[0])  # remove initial value to align with timestamps

            # interpolate overlapping optimized angle values and mocap values to the same timestamps (mocap timestamps)
            print(len(mocap_t))
            print(len(optimized_angle_values))
            emg_on_mocap = np.interp(mocap_t, imu_t, optimized_angle_values)
            a_on_mocap = np.interp(mocap_t, imu_t, emg_a_aligned)

            # create relative time vector
            t0 = mocap_t[0]
            mocap_t_rel = mocap_t - t0

            # Plot the data
            plt.figure(figsize=(12, 6))
            plt.title(f"Elbow Angle Comparison: {optimizer}")
            plt.plot(mocap_t_rel, mocap_y, label="MoCap")
            plt.plot(mocap_t_rel, emg_on_mocap, label=f"Optimized ({optimizer})")
            plt.xlabel("Time (s)")
            plt.ylabel("Elbow Angle (rad)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            valid_mask = (np.isfinite(emg_on_mocap) & np.isfinite(mocap_y))

            # Perform statistics
            np_optimized_angle_values = np.interp(mocap_t_rel, mocap_t_rel[valid_mask], np.array(emg_on_mocap)[valid_mask])
            mocap_y_valid = np.interp(mocap_t_rel, mocap_t_rel[valid_mask], mocap_y[valid_mask])

            # Calculate MAE
            mae = np.mean(np.abs(np_optimized_angle_values - mocap_y_valid))
            print(f"Mean Absolute Error for {optimizer}: {mae:.4f} radians")

            # Caclulate RMSE
            rmse = np.sqrt(np.mean((np_optimized_angle_values - mocap_y_valid)**2))
            print(f"Root Mean Square Error for {optimizer}: {rmse:.4f} radians")

            # Calculate Bias (mean error)
            bias = np.mean(np_optimized_angle_values - mocap_y_valid)
            print(f"Bias for {optimizer}: {bias:.4f} radians")

            # Calculate pearson correlation coefficient
            correlation = np.corrcoef(np_optimized_angle_values, mocap_y_valid)[0, 1]
            print(f"Pearson Correlation Coefficient for {optimizer}: {correlation:.4f}")

            # Calculate R-squared
            ss_res = np.sum((mocap_y_valid - np_optimized_angle_values) ** 2)
            ss_tot = np.sum((mocap_y_valid - np.mean(mocap_y_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"R-squared for {optimizer}: {r_squared:.4f}")

            # Calculate lag (cross-correlation)
            cross_corr = np.correlate(np_optimized_angle_values - np.mean(np_optimized_angle_values), mocap_y_valid - np.mean(mocap_y_valid), mode='full')
            lag = np.argmax(cross_corr) - (len(np_optimized_angle_values) - 1)
            lag_dt = np.mean(np.diff(mocap_t_rel))
            lag_time = lag * lag_dt
            if lag < 0:
                print(f"EMG leads MoCap by {lag_time:.2f} seconds")
            elif lag > 0:
                print(f"EMG lags behind MoCap by {abs(lag_time):.2f} seconds")
            else:
                print("No lag detected")

            # Calculate rising edge / onset lag
            # Use normalized threshold so amplitude differences matter less
            emg_norm = (np_optimized_angle_values - np.min(np_optimized_angle_values)) / (np.max(np_optimized_angle_values) - np.min(np_optimized_angle_values))
            mocap_norm = (mocap_y_valid - np.min(mocap_y_valid)) / (np.max(mocap_y_valid) - np.min(mocap_y_valid))

            emg_onsets = upward_crossings(mocap_t_rel, emg_norm, threshold=0.2)
            mocap_onsets = upward_crossings(mocap_t_rel, mocap_norm, threshold=0.2)

            # n = min(len(emg_onsets), len(mocap_onsets))
            # lags = emg_onsets[:n] - mocap_onsets[:n]

            lags = []
            for t_emg in emg_onsets:
                closest_idx = np.argmin(np.abs(mocap_onsets - t_emg))
                lags.append(t_emg - mocap_onsets[closest_idx])

            lags = np.array(lags)

            print("Mean onset lag:", np.mean(lags), "s")
            print("Median onset lag:", np.median(lags), "s")
            print("Std onset lag:", np.std(lags), "s")

            if np.median(lags) > 0:
                print(f"EMG lags MoCap by {np.median(lags):.3f} s")
            elif np.median(lags) < 0:
                print(f"EMG leads MoCap by {abs(np.median(lags)):.3f} s")

            # Calcualte peak to peak lag
            from scipy.signal import find_peaks
            emg_peaks, _ = find_peaks(emg_norm, distance=int(1.5/np.mean(np.diff(mocap_t_rel))))
            mocap_peaks, _ = find_peaks(mocap_norm, distance=int(1.5/np.mean(np.diff(mocap_t_rel))))

            # n = min(len(emg_peaks), len(mocap_peaks))
            # peak_lags = mocap_t_rel[emg_peaks[:n]] - mocap_t_rel[mocap_peaks[:n]]
            peak_lags = []
            for t_emg in mocap_t_rel[emg_peaks]:
                closest_idx = np.argmin(np.abs(mocap_t_rel[mocap_peaks] - t_emg))
                peak_lags.append(t_emg - mocap_t_rel[mocap_peaks][closest_idx])

            peak_lags = np.array(peak_lags)

            print("Median peak lag:", np.median(peak_lags), "s")


            # calculate ROM error (difference in range of motion)
            rom_error = (np.max(np_optimized_angle_values) - np.min(np_optimized_angle_values)) - (np.max(mocap_y_valid) - np.min(mocap_y_valid))
            print(f"Range of Motion Error for {optimizer}: {rom_error:.4f} radians")

            # shift optimized angle values by lag
            shifted_optimized_angle_values = np.roll(np_optimized_angle_values, -lag)
            # Recalculate statistics with shifted data
            shifted_mae = np.mean(np.abs(shifted_optimized_angle_values - mocap_y_valid))
            shifted_rmse = np.sqrt(np.mean((shifted_optimized_angle_values - mocap_y_valid)**2))
            print(f"Shifted Mean Absolute Error for {optimizer}: {shifted_mae:.4f} radians")
            print(f"Shifted Root Mean Square Error for {optimizer}: {shifted_rmse:.4f} radians")

            dt = 1/148
            vel = np.gradient(optimized_angle_values, dt)
            acc = np.gradient(vel, dt)
            jerk = np.gradient(acc, dt)

            mean_jerk = np.mean(np.abs(jerk))
            median_jerk = np.median(np.abs(jerk))
            sigma_jerk = np.std(jerk)
            max_jerk = np.max(np.abs(jerk))
            q25_jerk = np.percentile(np.abs(jerk), 25)
            q75_jerk = np.percentile(np.abs(jerk), 75)
            lower_median_quantile = median_jerk - q25_jerk
            upper_median_quantile = q75_jerk - median_jerk
            lower_mean_quantile = mean_jerk - q25_jerk
            upper_mean_quantile = q75_jerk - mean_jerk
            abs_jerk = np.abs(jerk)

            t = np.arange(len(optimized_angle_values)) * dt

            # # Plot the jerk profile
            # plt.figure(figsize=(12, 6))
            # plt.plot(t, jerk, label=f"Jerk of Optimized Angle ({optimizer})")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Jerk (rad/s^3)")
            # plt.title(f"Jerk Profile of Optimized Angle - {optimizer}")
            # plt.legend()
            # plt.grid()
            # plt.tight_layout()
            # plt.show()
        
            extension = Path(mocap_file).stem

            # Save statistics to a csv file
            stats_df = pd.DataFrame({
                "Optimizer": [optimizer],
                "MAE": [mae],
                "RMSE": [rmse],
                "Bias": [bias],
                "Correlation": [correlation],
                "R_squared": [r_squared],
                "Lag": [lag],
                "Lag_seconds": [lag_time],
                "Mean_Onset_Lag_sec": [np.mean(lags)],
                "Median_Onset_Lag_sec": [np.median(lags)],
                "Std_Onset_Lag_sec": [np.std(lags)],
                "Median_Peak_Lag_sec": [np.median(peak_lags)],
                "Mean_Peak_Lag_sec": [np.mean(peak_lags)],
                "ROM_Error": [rom_error],
                "Shifted_MAE": [shifted_mae],
                "Shifted_RMSE": [shifted_rmse]
            })
            stats_file = SAVEPATH + f"IMU_MoCap_stats_{optimizer}_{extension}.csv"
            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH)
            stats_df.to_csv(stats_file, index=False)

            # Save MoCap and IMU data to a csv file for further analysis
            mocap_imu_df = pd.DataFrame({
                "Time_sec": mocap_t_rel,
                "MoCap_Angle_rad": mocap_y,
                "Optimized_Angle_rad": emg_on_mocap,
                "Filtered_Net_A": a_on_mocap
            })
            mocap_imu_file = SAVEPATH + f"IMU_MoCap_results_{optimizer}_{extension}.csv"
            mocap_imu_df.to_csv(mocap_imu_file, index=False)

            # Save jerk stats to a csv file
            jerk_stats_df = pd.DataFrame({
                "Optimizer": [optimizer],
                "Mean_Jerk": [mean_jerk],
                "Median_Jerk": [median_jerk],
                "Sigma_Jerk": [sigma_jerk],
                "Max_Jerk": [max_jerk],
                "Q25_Jerk": [q25_jerk],
                "Q75_Jerk": [q75_jerk],
                "Lower_Median_Quantile": [lower_median_quantile],
                "Upper_Median_Quantile": [upper_median_quantile],
                "Lower_Mean_Quantile": [lower_mean_quantile],
                "Upper_Mean_Quantile": [upper_mean_quantile]
            })
            jerk_stats_file = SAVEPATH + f"IMU_MoCap_jerk_stats_{optimizer}_{extension}.csv"
            jerk_stats_df.to_csv(jerk_stats_file, index=False)

            # Save jerk profile to a csv file
            jerk_df = pd.DataFrame({
                "Time_sec": t,
                "Jerk": jerk,
                "abs_Jerk": abs_jerk
            })
            jerk_file = SAVEPATH + f"IMU_MoCap_jerk_{optimizer}_{extension}.csv"
            jerk_df.to_csv(jerk_file, index=False)
