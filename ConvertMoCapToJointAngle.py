import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time

# EMG processing imports
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer
from ProjectInRobotics.pDMP.pDMP_functions import pDMP, pDMPCoupling1, pDMPOmega

SAVEPATH = "Outputs/Results/"

EMG_OPTIMIZERS = [
    "None",
    "optimizer_1",
    "optimizer_2",
    "optimizer_4",
    "optimizer_5_pd",
    "optimizer_6",
    "EMG_Optimizer",
    "pDMP",
    "pDMP coupled",
    "pDMP omega"
]

INPUT_MOCAP_DATA = [
    "Outputs/MoCap/ExoTestReal1.csv",
    "Outputs/MoCap/ExoTestReal1_002.csv"
    # "Outputs/MoCapEMGData/ExoTest1.csv",
    # "Outputs/MoCapEMGData/ExoTest2.csv",
    # "Outputs/MoCapEMGData/ExoTest3.csv",
]

INPUT_EMG_DATA = [
    "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv",
    "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv",
    # "Outputs/MoCapEMGData/ExoTest1_Trigno_2801.csv",
    # "Outputs/MoCapEMGData/ExoTest2_Trigno_2801.csv",
    # "Outputs/MoCapEMGData/ExoTest3_Trigno_2801.csv",
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

def process_emg(file, optimizer):
    EMG_FS = 2000  # Hz
    dt = 1 / EMG_FS

    phi = 0
    tau = 0.5
    if optimizer == "pDMP omega":
        tau = 5
    omega0 = 2*np.pi/tau
    DMP = None

    if optimizer == "pDMP":
        # Teach DMPS
        DMP = pDMP(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
        # Teach DMP 0 trajectory for 2s
        y_old = 0
        dy_old = 0
        print("Teaching DMP 0 trajectory for 3s")
        start_time = time.time()
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
        print("DMP teaching completed")

    elif optimizer == "pDMP coupled":
        DMP = pDMPCoupling1(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
        # Teach DMP 0 trajectory for 3s
        y_old = 0
        dy_old = 0
        print("Teaching pDMP coupling 1 with 0 trajectory for 2s")
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

    elif optimizer == "pDMP omega":
        mid = np.deg2rad(70)
        DMP = pDMPOmega(DOF=1, N=25, alpha=8, beta=2, lambd=0.999, dt=dt)
        DMP.set_frequency([omega0])
        # Teach DMP 0 trajectory for 3s
        y_old = 0
        dy_old = 0
        print("Teaching pDMP omega with 0 trajectory for 5s")
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

    # Extract start time
    with open(file, "r") as f:
        lines = f.readlines()

    start_time_line = lines[4].strip()
    start_time = start_time_line.split(",")[1].strip()
    
    print("EMG Start time:", start_time)
    
    # load csv
    df = pd.read_csv(file, skiprows=14)  # skip metadata

    print(df.head())
    print([repr(col) for col in df.columns])

    # Load data into numpy arrays
    timestamps = df[" MocapTime"].to_numpy()
    BicepValues = df["Bicep"].to_numpy()
    TricepValues = df["Tricep"].to_numpy()

    # Convert EMG timestamps to absolute unix timestamps
    # Convert timestamps into unix timestamps
    # Fix format (replace dots in time part)
    start_time_clean = start_time.replace(".", ":", 2)  # only replace first two dots
    print("Cleaned start time:", start_time_clean)
    
    # Parse to datetime
    start_dt = datetime.strptime(start_time, "%d %b %Y %H:%M:%S %z")
    start_unix = start_dt.timestamp()
    absolute_timestamps = start_unix + timestamps
    print("Start time as unix timestamp:", start_unix)

    # add start_unix to relative timestamps to get absolute unix timestamps
    absolute_timestamps = start_unix + timestamps

    emg_absolute_timestamps = start_unix + np.arange(len(BicepValues)) * dt

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

    # Optimization parameters
    q = 0
    v = 0
    optimized_angle = 0
    delta_q_prev = 0
    a_prev = 0

    # Containers
    filtered_net_a_values = []
    optimized_angle_values = []

    if optimizer != "None":
        optimized_angle_values.append(q)


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

        if optimizer == "None":
            optimized_angle_values.append(interpreter.compute_angle(filtered_net_a))
        elif optimizer == "optimizer_1":
            optimized_angle = optimize_1((3*np.pi), filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            # optimized_angle = optimize_1((2*np.pi), filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_2":
            optimized_angle = optimize_2((4*np.pi), filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_4":
            optimized_angle, delta_q_prev = optimize_4((2*np.pi), filtered_net_a, dt, optimized_angle_values[-1], delta_q_prev, THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_5_pd":
            # optimized_angle, v = optimize_5_pd(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, 4, 0.1)
            optimized_angle, v = optimize_5_pd(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, 4, 0.01)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_6":
            optimized_angle, v, acc = optimizer_6(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, b=10.0, k=np.pi*10.0*2)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "EMG_Optimizer":
            a_d = (filtered_net_a - a_prev) / dt
            a_prev = filtered_net_a
            optimized_angle, v, acc = EMG_Optimizer(filtered_net_a, a_d, v, 10.0, 2.0, 2.0, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, dt)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "pDMP":
            v = np.pi/10 #np.pi/22
            DMP.set_phase(np.array([phi]))
            DMP.set_period(np.array([tau]))

            U = np.asarray([filtered_net_a*v])  # EMG activation as input
            DMP.update(U)
            DMP.integration()
            x, dx, ph, ta = DMP.get_state()
            optimized_angle_values.append(x[0])

        elif optimizer == "pDMP coupled":
            DMP.set_phase(np.array([phi]))
            DMP.set_period(np.array([tau]))

            DMP.repeat()

            DMP.integration(np.array([filtered_net_a]))

            x, dx, ph, ta = DMP.get_state()
            optimized_angle_values.append(x[0])
                
        elif optimizer == "pDMP omega":
            k = 1.0
            omega = omega0 * (1 + k * filtered_net_a)
            DMP.set_frequency([omega])
            DMP.repeat()
            DMP.integration()
            x, dx, ph, ta = DMP.get_state()
            optimized_angle_values.append(x[0])

        else:
            print(f"Unknown optimizer: {optimizer}")

    if optimizer != "None":
        optimized_angle_values.remove(optimized_angle_values[0])  # remove initial value to align with timestamps

    return filtered_net_a_values, optimized_angle_values, absolute_timestamps, timestamps, start_time, emg_absolute_timestamps

if __name__ == "__main__":
    for mocap_file, emg_file in zip(INPUT_MOCAP_DATA, INPUT_EMG_DATA):
        print(f"Processing MoCap file: {mocap_file}")
        absolute_timestamps, elbow_angle_rad, elbow_flexion_deg, relative_timestamps, mocap_start_time = process_mocap(mocap_file)

        for optimizer in EMG_OPTIMIZERS:
            print(f"Processing EMG file: {emg_file} with optimizer: {optimizer}")
            filtered_net_a_values, optimized_angle_values, absolute_timestamps2, timestamps, start_time, emg_absolute_timestamps = process_emg(emg_file, optimizer)

            # Plot mocap and emg data overlapping
            # Interpolate mocap to EMG time
            print(len(absolute_timestamps))
            print(len(elbow_angle_rad))
            mocap_interp = np.interp(
                emg_absolute_timestamps,
                absolute_timestamps,
                elbow_flexion_deg
                # elbow_angle_rad
            )
            # Convert to relative time
            t0 = emg_absolute_timestamps[0]
            time_sec = emg_absolute_timestamps - t0

            plt.figure(figsize=(12, 6))
            plt.plot(time_sec, optimized_angle_values, label=f"Optimized Angle ({optimizer})")
            plt.plot(time_sec, mocap_interp, label="MoCap Elbow Angle")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (rad)")
            plt.title(f"MoCap vs EMG - {optimizer}")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

            ################# Statistics #################
            valid_mask = (np.isfinite(optimized_angle_values)) & (np.isfinite(mocap_interp))

            np_optimized_angle_values = np.array(optimized_angle_values)[valid_mask]
            mocap_interp_valid = mocap_interp[valid_mask]

            # Calculate MAE
            mae = np.mean(np.abs(np_optimized_angle_values - mocap_interp_valid))
            print(f"Mean Absolute Error for {optimizer}: {mae:.2f} radians")

            # Caclulate RMSE
            rmse = np.sqrt(np.mean((np_optimized_angle_values - mocap_interp_valid)**2))
            print(f"Root Mean Square Error for {optimizer}: {rmse:.2f} radians")

            # Calculate Bias (mean error)
            bias = np.mean(np_optimized_angle_values - mocap_interp_valid)
            print(f"Bias for {optimizer}: {bias:.2f} radians")

            # Calculate pearson correlation coefficient
            correlation = np.corrcoef(np_optimized_angle_values, mocap_interp_valid)[0, 1]
            print(f"Pearson Correlation Coefficient for {optimizer}: {correlation:.2f}")

            # Calculate R-squared
            ss_res = np.sum((np_optimized_angle_values - mocap_interp_valid) ** 2)
            ss_tot = np.sum((mocap_interp_valid - np.mean(mocap_interp_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"R-squared for {optimizer}: {r_squared:.2f}")

            # Calculate lag (cross-correlation)
            cross_corr = np.correlate(np_optimized_angle_values - np.mean(np_optimized_angle_values), mocap_interp_valid - np.mean(mocap_interp_valid), mode='full')
            lag = np.argmax(cross_corr) - (len(mocap_interp) - 1)
            lag_time = lag * (time_sec[1] - time_sec[0])
            if lag > 0:
                print(f"EMG leads MoCap by {lag_time:.2f} seconds")
            elif lag < 0:
                print(f"EMG lags behind MoCap by {abs(lag_time):.2f} seconds")
            else:
                print("No lag detected")

            # calculate ROM error (difference in range of motion)
            rom_error = (np.max(np_optimized_angle_values) - np.min(np_optimized_angle_values)) - (np.max(mocap_interp_valid) - np.min(mocap_interp_valid))
            print(f"Range of Motion Error for {optimizer}: {rom_error:.2f} radians")

            # Shift EMG by lag
            shifted_optimized_angle_values = np.roll(np_optimized_angle_values, -lag)
            # Recalculate statistics with shifted data
            shifted_mae = np.mean(np.abs(shifted_optimized_angle_values - mocap_interp_valid))
            shifted_rmse = np.sqrt(np.mean((shifted_optimized_angle_values - mocap_interp_valid)**2))
            print(f"Shifted Mean Absolute Error for {optimizer}: {shifted_mae:.2f} radians")
            print(f"Shifted Root Mean Square Error for {optimizer}: {shifted_rmse:.2f} radians")

            # Calculate jerk of the optimized angle
            dt = 1/2000
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

            # # Plot the jerk profile
            # plt.figure(figsize=(12, 6))
            # plt.plot(time_sec, jerk, label=f"Jerk of Optimized Angle ({optimizer})")
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
                "Lag_time_sec": [lag_time],
                "ROM_error": [rom_error],
                "Shifted_MAE": [shifted_mae],
                "Shifted_RMSE": [shifted_rmse]
            })
            stats_file = SAVEPATH + f"EMG_MoCap_stats_{optimizer}_{extension}.csv"
            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH)
            stats_df.to_csv(stats_file, index=False)

            # Save moCap and EMG data to a csv file for further analysis
            results_df = pd.DataFrame({
                "Time_sec": time_sec,
                "MoCap_Elbow_Angle": mocap_interp,
                "Optimized_Angle": optimized_angle_values,
                "Filtered_Net_A": filtered_net_a_values
            })
            results_file = SAVEPATH + f"EMG_MoCap_results_{optimizer}_{extension}.csv"
            results_df.to_csv(results_file, index=False)

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
            jerk_stats_file = SAVEPATH + f"EMG_MoCap_jerk_stats_{optimizer}_{extension}.csv"
            jerk_stats_df.to_csv(jerk_stats_file, index=False)
            
            # Save jerk profile to a csv file
            jerk_df = pd.DataFrame({
                "Time_sec": time_sec,
                "Jerk": jerk,
                "abs_Jerk": abs_jerk
            })
            jerk_file = SAVEPATH + f"EMG_MoCap_jerk_{optimizer}_{extension}.csv"
            jerk_df.to_csv(jerk_file, index=False)

