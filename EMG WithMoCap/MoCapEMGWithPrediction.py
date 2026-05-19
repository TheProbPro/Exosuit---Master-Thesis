import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

# EMG processing imports
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimizer_6, optimize_2
import AdaptiveEmbodiedControlSystems.ESN as ESN
import AdaptiveEmbodiedControlSystems.LSTM as LSTM

class ESNWithActivation(nn.Module):
    def __init__(self, esn_model, activation='softplus'):
        super(ESNWithActivation, self).__init__()
        self.esn = esn_model
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
    
    def forward(self, x, state=None):
        outputs, final_state = self.esn(x, state)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs, final_state

class WindowedESNWithActivation(nn.Module):
    def __init__(self, esn_model, activation='softplus'):
        super(WindowedESNWithActivation, self).__init__()
        self.esn = esn_model
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None
    
    def forward(self, x, state=None):
        # WindowedESN只返回一个值，不是两个
        output = self.esn(x, state)
        if self.activation is not None:
            output = self.activation(output)
        return output

SAVEPATH = "Outputs/PredictionResults/New/"

ESN_SAVEPATH = "Outputs/models/ESN/Windowed_ESN_80.pth"
LSTM_SAVEPATH = "Outputs/models/LSTM/Windowed_LSTM_80.pth"

PREDICTION_MODELS = [
    "ESN",
    "LSTM"
]

INPUT_MOCAP_DATA = [
    # "Outputs/MoCap/ExoTestReal1.csv",
    # "Outputs/MoCap/ExoTestReal1_002.csv"
    "Outputs/NewMoCap/EMGTest6s_001.csv",
]

INPUT_EMG_DATA = [
    # "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv",
    # "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv"
    "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv"
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
    # start_dt = start_dt + timedelta(hours=12)
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

    window = queue.Queue(maxsize=100)  # 50 samples at 2000Hz = 25ms window

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if optimizer == "ESN":
        base_model = ESN.WindowedESN(
            input_size=1, 
            reservoir_size=100,
            output_size=1,
            spectral_radius=0.9,
            leaking_rate=0.7,
            connectivity=0.1
        )
        model = WindowedESNWithActivation(base_model, activation='softplus').to(device)
        model.load_state_dict(torch.load(ESN_SAVEPATH, map_location=device))
        model.eval()
    elif optimizer == "LSTM":
        model = LSTM.LSTMModel(input_size=1, hidden_size=64, output_size=1, num_layers=1, batch_first=True).to(device)
        model.load_state_dict(torch.load(LSTM_SAVEPATH, map_location=device))
        model.eval()
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

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
    optimized_angle_before = []
    optimized_angle_values = []
    optimized_angle_before.append(optimized_angle)

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

        # optimized_angle, v, acc = optimizer_6(filtered_net_a, v, dt, optimized_angle_before[-1], THETA_MIN, THETA_MAX, np.pi, b=10.0, k=np.pi*10.0*2)
        optimized_angle = optimize_2(
            np.pi*0.9, filtered_net_a, dt,
            optimized_angle_before[-1], THETA_MIN, THETA_MAX
        )

        optimized_angle_before.append(optimized_angle)

        try:
            window.put_nowait(optimized_angle)
        except queue.Full:
            window.get()
            window.put_nowait(optimized_angle)

        if optimizer == "ESN":
            with torch.no_grad():
                input_tensor = torch.tensor(list(window.queue), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                # t = time.time()
                esn_output = model(input_tensor)
                # print(f"processing time for ESN input tensor creation: {time.time() - t:.4f} seconds")
                optimized_angle = esn_output.item()
        elif optimizer == "LSTM":
            with torch.no_grad():
                input_tensor = torch.tensor(list(window.queue), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                # t = time.time()
                lstm_output = model(input_tensor)
                # print(f"processing time for LSTM input tensor creation: {time.time() - t:.4f} seconds")
                optimized_angle = lstm_output.item()
        
        optimized_angle_values.append(optimized_angle)
    
    optimized_angle_before = optimized_angle_before[1:]  # remove initial value

        

    return filtered_net_a_values, optimized_angle_values, optimized_angle_before, absolute_timestamps, timestamps, start_time, emg_absolute_timestamps

if __name__ == "__main__":
    for mocap_file, emg_file in zip(INPUT_MOCAP_DATA, INPUT_EMG_DATA):
        print(f"Processing MoCap file: {mocap_file}")
        absolute_timestamps, elbow_angle_rad, elbow_flexion_deg, relative_timestamps, mocap_start_time = process_mocap(mocap_file)

        for optimizer in PREDICTION_MODELS:
            print(f"Processing EMG file: {emg_file} with optimizer: {optimizer}")
            filtered_net_a_values, optimized_angle_values, optimized_angle_before, absolute_timestamps2, timestamps, start_time, emg_absolute_timestamps = process_emg(emg_file, optimizer)

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
            plt.plot(time_sec, optimized_angle_before, label=f"Optimized Angle Before Prediction ({optimizer})", linestyle='--')
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
            np_optimized_angle_before = np.array(optimized_angle_before)[valid_mask]
            mocap_interp_valid = mocap_interp[valid_mask]

            # Calculate MAE
            mae = np.mean(np.abs(np_optimized_angle_values - mocap_interp_valid))
            print(f"Mean Absolute Error for {optimizer}: {mae:.4f} radians")
            mae_before = np.mean(np.abs(np_optimized_angle_before - mocap_interp_valid))
            print(f"Mean Absolute Error Before Prediction for {optimizer}: {mae_before:.4f} radians")
            improvement = mae_before - mae
            improvement_percent = (mae_before - mae) / mae_before
            print(f"Improvement for {optimizer}: {improvement}, {improvement_percent:.2%}")

            # Caclulate RMSE
            rmse = np.sqrt(np.mean((np_optimized_angle_values - mocap_interp_valid)**2))
            print(f"Root Mean Square Error for {optimizer}: {rmse:.4f} radians")
            rmse_before = np.sqrt(np.mean((np_optimized_angle_before - mocap_interp_valid)**2))
            print(f"Root Mean Square Error Before Prediction for {optimizer}: {rmse_before:.4f} radians")
            rmse_improvement = rmse_before - rmse
            rmse_improvement_percent = (rmse_before - rmse) / rmse_before
            print(f"RMSE Improvement for {optimizer}: {rmse_improvement}, {rmse_improvement_percent:.2%}")

            # Calculate Bias (mean error)
            bias = np.mean(np_optimized_angle_values - mocap_interp_valid)
            print(f"Bias for {optimizer}: {bias:.4f} radians")

            # Calculate pearson correlation coefficient
            correlation = np.corrcoef(np_optimized_angle_values, mocap_interp_valid)[0, 1]
            print(f"Pearson Correlation Coefficient for {optimizer}: {correlation:.4f}")

            correlation_before = np.corrcoef(np_optimized_angle_before, mocap_interp_valid)[0, 1]
            print(f"Pearson Correlation Coefficient Before Prediction for {optimizer}: {correlation_before:.2f}")
            correlation_improvement = correlation - correlation_before
            print(f"Correlation Improvement for {optimizer}: {correlation_improvement:.4f}")

            # Calculate R-squared
            ss_res = np.sum((np_optimized_angle_values - mocap_interp_valid) ** 2)
            ss_tot = np.sum((mocap_interp_valid - np.mean(mocap_interp_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"R-squared for {optimizer}: {r_squared:.4f}")

            # Calculate lag (cross-correlation)
            cross_corr = np.correlate(np_optimized_angle_values - np.mean(np_optimized_angle_values), mocap_interp_valid - np.mean(mocap_interp_valid), mode='full')
            lag = np.argmax(cross_corr) - (len(mocap_interp) - 1)
            lag_time = lag * (time_sec[1] - time_sec[0])
            if lag > 0:
                print(f"EMG leads MoCap by {lag_time:.4f} seconds")
            elif lag < 0:
                print(f"EMG lags behind MoCap by {abs(lag_time):.4f} seconds")
            else:
                print("No lag detected")

            cross_corr_before = np.correlate(np_optimized_angle_before - np.mean(np_optimized_angle_before), mocap_interp_valid - np.mean(mocap_interp_valid), mode='full')
            lag_before = np.argmax(cross_corr_before) - (len(mocap_interp) - 1)
            lag_time_before = lag_before * (time_sec[1] - time_sec[0])
            if lag_before > 0:
                print(f"Before Prediction: EMG leads MoCap by {lag_time_before:.4f} seconds")
            elif lag_before < 0:
                print(f"Before Prediction: EMG lags behind MoCap by {abs(lag_time_before):.4f} seconds")
            else:
                print("Before Prediction: No lag detected")
            
            lag_improvement = lag_before - lag
            lag_improvement_time = lag_improvement * (time_sec[1] - time_sec[0])
            print(f"Lag Improvement for {optimizer}: {lag_improvement}")
            print(f"Lag Improvement Time for {optimizer}: {lag_improvement_time:.4f} seconds")

            # calculate ROM error (difference in range of motion)
            rom_error = (np.max(np_optimized_angle_values) - np.min(np_optimized_angle_values)) - (np.max(mocap_interp_valid) - np.min(mocap_interp_valid))
            print(f"Range of Motion Error for {optimizer}: {rom_error:.4f} radians")

            # # Shift EMG by lag
            # shifted_optimized_angle_values = np.roll(np_optimized_angle_values, -lag)
            # # Recalculate statistics with shifted data
            # shifted_mae = np.mean(np.abs(shifted_optimized_angle_values - mocap_interp_valid))
            # shifted_rmse = np.sqrt(np.mean((shifted_optimized_angle_values - mocap_interp_valid)**2))
            # print(f"Shifted Mean Absolute Error for {optimizer}: {shifted_mae:.2f} radians")
            # print(f"Shifted Root Mean Square Error for {optimizer}: {shifted_rmse:.2f} radians")

            # Calculate jerk of the optimized angle
            dt = 1/2000
            vel = np.gradient(optimized_angle_values, dt)
            acc = np.gradient(vel, dt)
            jerk = np.gradient(acc, dt)

            vel_before = np.gradient(optimized_angle_before, dt)
            acc_before = np.gradient(vel_before, dt)
            jerk_before = np.gradient(acc_before, dt)

            jerk_improvement = np.mean(np.abs(jerk_before)) - np.mean(np.abs(jerk))
            print(f"Mean Jerk Improvement for {optimizer}: {jerk_improvement:.2f} radians/s^3")

            jerk_ratio = np.mean(np.abs(jerk)) / (np.mean(np.abs(jerk_before)) + 1e-8)
            print(f"Mean Jerk Ratio for {optimizer}: {np.mean(jerk_ratio):.2f}")

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

            # # Save statistics to a csv file
            # stats_df = pd.DataFrame({
            #     "Optimizer": [optimizer],
            #     "MAE": [mae],
            #     "RMSE": [rmse],
            #     "Bias": [bias],
            #     "Correlation": [correlation],
            #     "R_squared": [r_squared],
            #     "Lag": [lag],
            #     "Lag_time_sec": [lag_time],
            #     "ROM_error": [rom_error],
            #     "Shifted_MAE": [shifted_mae],
            #     "Shifted_RMSE": [shifted_rmse]
            # })
            # stats_file = SAVEPATH + f"EMG_MoCap_stats_{optimizer}_{extension}.csv"
            # if not os.path.exists(SAVEPATH):
            #     os.makedirs(SAVEPATH)
            # stats_df.to_csv(stats_file, index=False)

            # Save moCap and EMG data to a csv file for further analysis
            results_df = pd.DataFrame({
                "Time_sec": time_sec,
                "MoCap_Elbow_Angle": mocap_interp,
                "Optimized_Angle": optimized_angle_values,
                "Optimized_Angle_Before": optimized_angle_before,
                "Filtered_Net_A": filtered_net_a_values
            })
            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH)
            results_file = SAVEPATH + f"EMG_MoCap_results_{optimizer}_{extension}.csv"
            results_df.to_csv(results_file, index=False)

            # # Save jerk stats to a csv file
            # jerk_stats_df = pd.DataFrame({
            #     "Optimizer": [optimizer],
            #     "Mean_Jerk": [mean_jerk],
            #     "Median_Jerk": [median_jerk],
            #     "Sigma_Jerk": [sigma_jerk],
            #     "Max_Jerk": [max_jerk],
            #     "Q25_Jerk": [q25_jerk],
            #     "Q75_Jerk": [q75_jerk],
            #     "Lower_Median_Quantile": [lower_median_quantile],
            #     "Upper_Median_Quantile": [upper_median_quantile],
            #     "Lower_Mean_Quantile": [lower_mean_quantile],
            #     "Upper_Mean_Quantile": [upper_mean_quantile]
            # })
            # jerk_stats_file = SAVEPATH + f"EMG_MoCap_jerk_stats_{optimizer}_{extension}.csv"
            # jerk_stats_df.to_csv(jerk_stats_file, index=False)
            
            # # Save jerk profile to a csv file
            # jerk_df = pd.DataFrame({
            #     "Time_sec": time_sec,
            #     "Jerk": jerk,
            #     "abs_Jerk": abs_jerk
            # })
            # jerk_file = SAVEPATH + f"EMG_MoCap_jerk_{optimizer}_{extension}.csv"
            # jerk_df.to_csv(jerk_file, index=False)

