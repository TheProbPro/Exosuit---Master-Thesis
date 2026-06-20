import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.signal import find_peaks
from itertools import combinations

# EMG processing imports
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer
from ProjectInRobotics.pDMP.pDMP_functions import pDMP, pDMPCoupling1, pDMPOmega

SAVEPATH = "Outputs/ResultsV2/"

EMG_OPTIMIZERS = [
    "None",
    "optimizer_1",
    "optimizer_2",
    "optimizer_4",
    "optimizer_5_pd",
    "optimizer_6",
    "EMG_Optimizer",
    "pDMP",
    "pDMP coupled"
]

INPUT_MOCAP_DATA = [
    # "Outputs/MoCap/ExoTestReal1.csv",
    # "Outputs/MoCap/ExoTestReal1_002.csv",
    # # "Outputs/MoCapEMGData/ExoTest1.csv",
    # # "Outputs/MoCapEMGData/ExoTest2.csv",
    # # "Outputs/MoCapEMGData/ExoTest3.csv",
    "Outputs/NewMoCap/EMGTest6s_001.csv",
]

INPUT_EMG_DATA = [
    # "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv",
    # "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv",
    # # "Outputs/MoCapEMGData/ExoTest1_Trigno_2801.csv",
    # # "Outputs/MoCapEMGData/ExoTest2_Trigno_2801.csv",
    # # "Outputs/MoCapEMGData/ExoTest3_Trigno_2801.csv",
    "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv",
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
    # print(f"Columns in the CSV: {df.columns.tolist()}")

    # arrange data into relevant numpy arrays
    timestamps = df[("Name", "Time (Seconds)")].to_numpy()
    forearm_lower = df["ForeArm:Lower"][["X", "Y", "Z"]].to_numpy()
    forearm_upper = df["ForeArm:Upper"][["X", "Y", "Z"]].to_numpy()
    upperarm_middle = df["UpperArm:Middle"][["X", "Y", "Z"]].to_numpy()
    upperarm_upper = df["UpperArm:Upper"][["X", "Y", "Z"]].to_numpy()
    upperarm_elbow = df["UpperArm:Elbow"][["X", "Y", "Z"]].to_numpy()

    # Convert NaN to 0
    # forearm_lower = np.nan_to_num(forearm_lower, nan=0.0)
    # forearm_upper = np.nan_to_num(forearm_upper, nan=0.0)
    # upperarm_middle = np.nan_to_num(upperarm_middle, nan=0.0)
    # upperarm_upper = np.nan_to_num(upperarm_upper, nan=0.0)
    # upperarm_elbow = np.nan_to_num(upperarm_elbow, nan=0.0)

    # Convert timestamps into unix timestamps
    # Fix format (replace dots in time part)
    start_time_clean = start_time.replace(".", ":", 2)  # only replace first two dots
    print("Cleaned start time:", start_time_clean)
    
    # Parse to datetime
    start_dt = datetime.strptime(start_time_clean, "%Y-%m-%d %H:%M:%S.%f")
    # If you know this should be PM rather than AM
    if not file == "Outputs/NewMoCap/EMGTest6s_001.csv":  # This file is from the morning, the others are from the afternoon
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

    THETA_MIN = np.deg2rad(0)
    THETA_MAX = np.deg2rad(140)
    THETA_RANGE = THETA_MAX - THETA_MIN

    phi = 0
    tau = 0.5
    if optimizer == "pDMP omega":
        tau = 5
    omega0 = 2*np.pi/tau
    DMP = None

    if optimizer == "pDMP":
        # Teach DMPS
        DMP = pDMP(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
        DMP.set_output_limits(THETA_MIN, THETA_MAX, squash_gain=1.0)
        DMP.set_output_state(np.array([0.0]))
        # Teach DMP 0 trajectory for 2s
        y_old = 0
        dy_old = 0
        print("Teaching DMP 0 trajectory for 3s")
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
            DMP.integration()

            # old values	
            y_old = y
            dy_old = dy
            
            # store data for plotting
            x, dx, ph, ta = DMP.get_state()
        print("DMP teaching completed")

    elif optimizer == "pDMP coupled":
        DMP = pDMPCoupling1(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
        DMP.set_output_limits(THETA_MIN, THETA_MAX, squash_gain=1.0)
        DMP.set_output_state(np.array([0.0]))
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

    # print(df.head())
    # print([repr(col) for col in df.columns])

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
    dif_a_prev = 0
    net_a_prev = 0

    # Containers
    filtered_net_a_values = []
    # dif_filtered_net_a_values = []
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
            # k=3*np.pi
            if file == "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv":
                k = 2.4 * np.pi
            elif file == "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv":
                k = 2.8 * np.pi # File 2
            elif file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                k = (1.6*np.pi) / 3
            optimized_angle = optimize_1(k, filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            # optimized_angle = optimize_1((2*np.pi), filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_2":
            # k = 4 * np.pi
            if file == "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv":
                k = 4.5 * np.pi
            elif file == "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv":
                k = 4.5 * np.pi # File 2
            elif file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                k = np.pi *1.1#* 0.9
            optimized_angle = optimize_2(k, filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_4":
            if file == "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv":
                k = 2.6 * np.pi # File 1
            elif file == "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv":
                k = 2.6 * np.pi # File 2
            elif file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                k = (2.2*np.pi) / 4
            optimized_angle, delta_q_prev = optimize_4(k, filtered_net_a, dt, optimized_angle_values[-1], delta_q_prev, THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_5_pd":
            # optimized_angle, v = optimize_5_pd(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, 4, 0.1)
            # k = 4
            k = 16 # File 1
            if file == "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv":
                k = 24
                b = 0.01 # file 1
            elif file == "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv":
                k = 24
                b = 0.02 # file 2
            elif file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                k = (1.6*np.pi) / 3
                b = 0.01 # 0.001
            optimized_angle, v = optimize_5_pd(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, k, b)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_6":
            k=np.pi*10.0*1.7 # File 1
            if file == "Outputs/MoCap/ExoTestReal1_Trigno_2801.csv":
                b = 3.0 # File 1
            elif file == "Outputs/MoCap/ExoTestReal1_002_Trigno_2801.csv":
                b = 3.6 # File 2
            elif file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                k = np.pi * 2.6
                b = 4
            optimized_angle, v, acc = optimizer_6(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, b, k)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "EMG_Optimizer":
            a_d = (filtered_net_a - a_prev) / dt
            a_prev = filtered_net_a
            kn = 4.0
            kd = 2.0
            b = 4.0
            if file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                kn = 5
                kd = 5
                b = 4
            optimized_angle, v, acc = EMG_Optimizer(filtered_net_a, a_d, v, kn, kd, b, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, dt)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "pDMP":
            v = np.pi/40 #np.pi/22
            if file == "Outputs/NewMoCap/EMGTest6s_001_Trigno_2801.csv":
                v = np.pi/14
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

    return filtered_net_a_values, optimized_angle_values, absolute_timestamps, timestamps, start_time, emg_absolute_timestamps#dif_filtered_net_a_values, optimized_angle_values, absolute_timestamps, timestamps, start_time, emg_absolute_timestamps

def upward_crossings(t, y, threshold):
    idx = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    crossings = []

    for i in idx:
        # linear interpolation for sub-sample crossing time
        t_cross = t[i] + (threshold - y[i]) * (t[i+1] - t[i]) / (y[i+1] - y[i])
        crossings.append(t_cross)

    return np.array(crossings)

def normalize_01(y):
    y = np.asarray(y, dtype=float)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    rng = ymax - ymin

    if rng < 1e-12:
        return np.zeros_like(y)

    return (y - ymin) / rng


def detect_cycles_from_mocap(time_sec, mocap_angle, threshold=0.2, min_cycle_duration=2.0):
    """
    Detect full flexion/extension cycles from the MoCap reference trajectory.

    A cycle is defined from one upward threshold crossing to the next upward
    threshold crossing of the normalized MoCap angle.
    """

    mocap_norm = normalize_01(mocap_angle)
    starts = upward_crossings(time_sec, mocap_norm, threshold)

    if len(starts) < 2:
        raise ValueError("Could not detect enough cycles. Try changing threshold or min_cycle_duration.")

    # Remove crossings that are unrealistically close together
    filtered_starts = [starts[0]]
    for s in starts[1:]:
        if s - filtered_starts[-1] >= min_cycle_duration:
            filtered_starts.append(s)

    cycles = []
    for i in range(len(filtered_starts) - 1):
        cycles.append((filtered_starts[i], filtered_starts[i + 1]))

    return cycles


def safe_corrcoef(x, y):
    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def safe_r2(pred, ref):
    ss_res = np.sum((pred - ref) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)

    if ss_tot < 1e-12:
        return np.nan

    return 1 - ss_res / ss_tot


def onset_lag_near_cycle_start(time_sec, pred, ref_cycle_start, search_radius=1.5, threshold=0.2):
    """
    Finds the predicted upward threshold crossing nearest to the MoCap cycle start.

    Negative lag means the predicted trajectory leads MoCap.
    Positive lag means the predicted trajectory lags MoCap.
    """

    mask = (time_sec >= ref_cycle_start - search_radius) & (time_sec <= ref_cycle_start + search_radius)

    if np.sum(mask) < 5:
        return np.nan

    t_win = time_sec[mask]
    pred_win = pred[mask]

    pred_norm = normalize_01(pred_win)
    pred_onsets = upward_crossings(t_win, pred_norm, threshold)

    if len(pred_onsets) == 0:
        return np.nan

    closest_pred_onset = pred_onsets[np.argmin(np.abs(pred_onsets - ref_cycle_start))]
    return closest_pred_onset - ref_cycle_start


def compute_cycle_metrics(time_sec, pred, ref, cycle_start, cycle_end, fs=None):
    """
    Compute the metrics for one method in one flexion/extension cycle.
    """

    time_sec = np.asarray(time_sec)
    pred = np.asarray(pred)
    ref = np.asarray(ref)

    mask = (time_sec >= cycle_start) & (time_sec < cycle_end)
    t = time_sec[mask]
    q = pred[mask]
    r = ref[mask]

    valid = np.isfinite(t) & np.isfinite(q) & np.isfinite(r)
    t = t[valid]
    q = q[valid]
    r = r[valid]

    if len(t) < 10:
        return None

    mae = np.mean(np.abs(q - r))
    rmse = np.sqrt(np.mean((q - r) ** 2))
    corr = safe_corrcoef(q, r)
    r2 = safe_r2(q, r)

    rom_pred = np.max(q) - np.min(q)
    rom_ref = np.max(r) - np.min(r)

    rom_error = rom_pred - rom_ref
    abs_rom_error = np.abs(rom_error)

    # Use actual time vector rather than assuming exactly 2000 Hz
    vel = np.gradient(q, t)
    acc = np.gradient(vel, t)
    jerk = np.gradient(acc, t)
    median_abs_jerk = np.median(np.abs(jerk))

    # Signed onset lag: negative = method leads MoCap
    onset_lag = onset_lag_near_cycle_start(
        time_sec=time_sec,
        pred=pred,
        ref_cycle_start=cycle_start,
        search_radius=1.5,
        threshold=0.2
    )

    # Peak lag inside this cycle
    try:
        pred_peak_time = t[np.argmax(q)]
        ref_peak_time = t[np.argmax(r)]
        peak_lag = pred_peak_time - ref_peak_time
    except Exception:
        peak_lag = np.nan

    return {
        "Median_Abs_Jerk": median_abs_jerk,
        "Pearson_r": corr,
        "R_squared": r2,
        "ROM_Error": rom_error,
        "Abs_ROM_Error": abs_rom_error,
        "MAE": mae,
        "RMSE": rmse,
        "Onset_Lag_sec": onset_lag,
        "Abs_Onset_Lag_sec": np.abs(onset_lag) if np.isfinite(onset_lag) else np.nan,
        "Peak_Lag_sec": peak_lag,
        "Abs_Peak_Lag_sec": np.abs(peak_lag) if np.isfinite(peak_lag) else np.nan,
        "Cycle_Duration_sec": cycle_end - cycle_start
    }

def add_weighted_score(cycle_metrics_df):
    """
    Adds a normalized weighted score to the cycle-metric dataframe.

    Higher score = better performance.
    Normalization is performed within each cycle across methods.
    """

    metric_weights = {
        "Median_Abs_Jerk": 1.5,
        "Pearson_r": 1.5,
        "R_squared": 1.2,
        "MAE": 1.0,
        "RMSE": 1.0,
        "Abs_ROM_Error": 1.2,
        "Abs_Onset_Lag_sec": 0.0 #1.0
    }

    higher_is_better = {
        "Median_Abs_Jerk": False,
        "Pearson_r": True,
        "R_squared": True,
        "MAE": False,
        "RMSE": False,
        "Abs_ROM_Error": False,
        "Abs_Onset_Lag_sec": False
    }

    scored_groups = []

    for cycle_id, group in cycle_metrics_df.groupby("Cycle"):
        group = group.copy()

        weighted_sum = np.zeros(len(group))
        weight_sum = np.zeros(len(group))

        for metric, weight in metric_weights.items():
            values = group[metric].astype(float).to_numpy()

            if np.all(~np.isfinite(values)):
                norm_values = np.full(len(values), np.nan)
            else:
                finite_values = values[np.isfinite(values)]
                vmin = np.min(finite_values)
                vmax = np.max(finite_values)

                if np.abs(vmax - vmin) < 1e-12:
                    norm_values = np.full(len(values), 0.5)
                else:
                    norm_values = (values - vmin) / (vmax - vmin)

                    if not higher_is_better[metric]:
                        norm_values = 1 - norm_values

            group[metric + "_norm"] = norm_values

            valid = np.isfinite(norm_values)
            weighted_sum[valid] += weight * norm_values[valid]
            weight_sum[valid] += weight

        group["Weighted_Score"] = weighted_sum / weight_sum
        scored_groups.append(group)

    return pd.concat(scored_groups, ignore_index=True)


def run_friedman_and_posthoc(cycle_metrics_df, metric="Weighted_Score", higher_is_better=True):
    """
    Runs a Friedman test across methods using cycles as repeated blocks.
    Then performs uncorrected pairwise Wilcoxon signed-rank post-hoc tests
    between all method pairs.

    Each row in the Friedman matrix is one cycle.
    Each column is one method.
    """

    pivot = cycle_metrics_df.pivot(index="Cycle", columns="Optimizer", values=metric)

    # Friedman requires complete paired blocks
    pivot = pivot.dropna(axis=0)

    if pivot.shape[0] < 2:
        raise ValueError(f"Not enough complete cycles for Friedman test on {metric}.")

    orig_cols = list(pivot.columns)
    data = [pivot[m].to_numpy() for m in orig_cols]

    # Create custom labels for methods (map pivot column -> friendly label)
    method_labels = {
        "None": "EMG to q_d",
        "optimizer_1": "Integrator 1",
        "optimizer_2": "Integrator 2",
        "optimizer_4": "Integrator 3",
        "optimizer_5_pd": "Integrator 4",
        "optimizer_6": "Integrator 5",
        "EMG_Optimizer": "Integrator 6",
        "pDMP omega": "pDMP omega",
        "pDMP coupled": "pDMP coupled",
        "pDMP": "pDMP weight update",
    }

    # Labels corresponding to each original column (same order)
    labels_for_cols = [method_labels.get(col, col) for col in orig_cols]

    # Desired plotting order (friendly labels)
    method_order = [
        "EMG to q_d",
        "Integrator 1",
        "Integrator 2",
        "Integrator 3",
        "Integrator 4",
        "Integrator 5",
        "Integrator 6",
        "pDMP weight update",
        "pDMP coupled"
    ]

    # Build ordered lists of original column names and labels
    ordered_cols = []
    ordered_labels = []
    for label in method_order:
        if label in labels_for_cols:
            idx = labels_for_cols.index(label)
            ordered_cols.append(orig_cols[idx])
            ordered_labels.append(label)

    # Append any remaining methods that weren't in method_order
    for col, lab in zip(orig_cols, labels_for_cols):
        if col not in ordered_cols:
            ordered_cols.append(col)
            ordered_labels.append(lab)

    # Create boxplot for visualization using ordered columns and labels
    plt.figure(figsize=(10, 6))
    ordered_data = [pivot[c].to_numpy() for c in ordered_cols]
    plt.boxplot(ordered_data, labels=ordered_labels)
    plt.title(f"Friedman Test Data for {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    for cycle_id, row in pivot.iterrows():
        y = row[ordered_cols].values
        plt.plot(ordered_labels, y, marker="o", alpha=0.5, label=f"Cycle {cycle_id}")

    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.title(f"Paired cycle-wise {metric} across methods")
    plt.tight_layout()
    plt.show()

    # Friedman omnibus test
    stat, p = friedmanchisquare(*data)

    n_cycles = pivot.shape[0]
    n_methods = pivot.shape[1]

    # Ranking: rank 1 is best
    if higher_is_better:
        ranks = pivot.rank(axis=1, ascending=False)
    else:
        ranks = pivot.rank(axis=1, ascending=True)

    mean_ranks = ranks.mean(axis=0).sort_values()

    print("\n==============================")
    print(f"Friedman test for: {metric}")
    print("==============================")
    print(f"Number of complete cycles: {n_cycles}")
    print(f"Number of methods: {n_methods}")
    print(f"Friedman chi-square = {stat:.4f}")
    print(f"p-value = {p:.6f}")
    print("\nMean ranks, lower is better:")
    print(mean_ranks)

    # Pairwise Wilcoxon signed-rank post-hoc tests
    posthoc_rows = []

    for col_a, col_b in combinations(ordered_cols, 2):
        x = pivot[col_a].to_numpy()
        y = pivot[col_b].to_numpy()

        try:
            w_stat, p_raw = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox"
            )
        except ValueError:
            w_stat, p_raw = np.nan, np.nan

        median_difference = np.median(x - y)

        posthoc_rows.append({
            "Method_A": method_labels.get(col_a, col_a),
            "Method_B": method_labels.get(col_b, col_b),
            "Metric": metric,
            "Wilcoxon_stat": w_stat,
            "p_value": p_raw,
            "Median_Difference_A_minus_B": median_difference,
            "Significant_p_less_0_1": p_raw < 0.1 if np.isfinite(p_raw) else False
        })

    posthoc_df = pd.DataFrame(posthoc_rows)

    print("\nPairwise Wilcoxon signed-rank post-hoc tests, uncorrected:")
    print(posthoc_df.sort_values("p_value"))

    friedman_summary = pd.DataFrame({
        "Metric": [metric],
        "N_cycles": [n_cycles],
        "N_methods": [n_methods],
        "Friedman_chi_square": [stat],
        "p_value": [p]
    })

    return friedman_summary, posthoc_df, mean_ranks

if __name__ == "__main__":
    for mocap_file, emg_file in zip(INPUT_MOCAP_DATA, INPUT_EMG_DATA):
        print(f"Processing MoCap file: {mocap_file}")
        absolute_timestamps, elbow_angle_rad, elbow_flexion_deg, relative_timestamps, mocap_start_time = process_mocap(mocap_file)

        method_results = {}
        reference_time_sec = None
        reference_mocap = None

        for optimizer in EMG_OPTIMIZERS:
            print(f"Processing EMG file: {emg_file} with optimizer: {optimizer}")
            # filtered_net_a_values, dif_filtered_net_a_values, optimized_angle_values, absolute_timestamps2, timestamps, start_time, emg_absolute_timestamps = process_emg(emg_file, optimizer)
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

            # plt.figure(figsize=(12, 6))
            # plt.plot(time_sec, optimized_angle_values, label=f"Optimized Angle ({optimizer})")
            # plt.plot(time_sec, mocap_interp, label="MoCap Elbow Angle")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Angle (rad)")
            # plt.title(f"MoCap vs EMG - {optimizer}")
            # plt.legend()
            # plt.grid()

            # plt.tight_layout()
            # plt.show()

            ################# Statistics #################
            valid_mask = (np.isfinite(optimized_angle_values)) & (np.isfinite(mocap_interp))
            # valid_emg = np.isfinite(optimized_angle_values)
            # valid_mocap = np.isfinite(mocap_interp)

            np_optimized_angle_values = np.interp(time_sec, time_sec[valid_mask], np.array(optimized_angle_values)[valid_mask])
            mocap_interp_valid = np.interp(time_sec, time_sec[valid_mask], mocap_interp[valid_mask])
            # np_optimized_angle_values = np.interp(time_sec, time_sec[valid_emg], np.array(optimized_angle_values)[valid_emg])
            # mocap_interp_valid = np.interp(time_sec, time_sec[valid_mocap], mocap_interp[valid_mocap])

            method_results[optimizer] = {
                "time_sec": time_sec,
                "pred": np_optimized_angle_values,
                "ref": mocap_interp_valid
            }

            reference_time_sec = time_sec
            reference_mocap = mocap_interp_valid

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
            lag = np.argmax(cross_corr) - (len(np_optimized_angle_values) - 1)
            lag_dt = np.mean(np.diff(time_sec))
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
            mocap_norm = (mocap_interp_valid - np.min(mocap_interp_valid)) / (np.max(mocap_interp_valid) - np.min(mocap_interp_valid))

            emg_onsets = upward_crossings(time_sec, emg_norm, threshold=0.2)
            mocap_onsets = upward_crossings(time_sec, mocap_norm, threshold=0.2)

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
            emg_peaks, _ = find_peaks(emg_norm, distance=int(1.5/np.mean(np.diff(time_sec))))
            mocap_peaks, _ = find_peaks(mocap_norm, distance=int(1.5/np.mean(np.diff(time_sec))))

            # n = min(len(emg_peaks), len(mocap_peaks))
            # peak_lags = time_sec[emg_peaks[:n]] - time_sec[mocap_peaks[:n]]
            peak_lags = []
            for t_emg in time_sec[emg_peaks]:
                closest_idx = np.argmin(np.abs(time_sec[mocap_peaks] - t_emg))
                peak_lags.append(t_emg - time_sec[mocap_peaks][closest_idx])

            peak_lags = np.array(peak_lags)

            print("Median peak lag:", np.median(peak_lags), "s")

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

            print(f"Mean Jerk for {optimizer}: {mean_jerk:.2f} rad/s^3")
            print(f"Median Jerk for {optimizer}: {median_jerk:.2f} rad/s^3")

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
                "Mean_Onset_Lag_sec": [np.mean(lags)],
                "Median_Onset_Lag_sec": [np.median(lags)],
                "Std_Onset_Lag_sec": [np.std(lags)],
                "Median_Peak_Lag_sec": [np.median(peak_lags)],
                "Mean_Peak_Lag_sec": [np.mean(peak_lags)],
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
        
        # ==========================================================
        # Cycle-based statistical analysis
        # ==========================================================

        print("\nDetecting flexion/extension cycles from MoCap reference...")

        cycles = detect_cycles_from_mocap(
            time_sec=reference_time_sec,
            mocap_angle=reference_mocap,
            threshold=0.2,
            min_cycle_duration=2.0
        )

        print(f"Detected {len(cycles)} complete flexion/extension cycles.")

        cycle_rows = []

        for cycle_id, (cycle_start, cycle_end) in enumerate(cycles, start=1):
            for optimizer, data in method_results.items():

                metrics = compute_cycle_metrics(
                    time_sec=data["time_sec"],
                    pred=data["pred"],
                    ref=data["ref"],
                    cycle_start=cycle_start,
                    cycle_end=cycle_end
                )

                if metrics is None:
                    continue

                row = {
                    "Cycle": cycle_id,
                    "Cycle_Start_sec": cycle_start,
                    "Cycle_End_sec": cycle_end,
                    "Optimizer": optimizer
                }

                row.update(metrics)
                cycle_rows.append(row)

        cycle_metrics_df = pd.DataFrame(cycle_rows)

        # Add normalized weighted score per cycle
        cycle_metrics_df = add_weighted_score(cycle_metrics_df)

        extension = Path(mocap_file).stem

        cycle_metrics_file = SAVEPATH + f"cycle_metrics_{extension}.csv"
        cycle_metrics_df.to_csv(cycle_metrics_file, index=False)

        print(f"\nSaved cycle-level metrics to: {cycle_metrics_file}")
        print(cycle_metrics_df.head())

        # Friedman test on weighted score
        friedman_summary, posthoc_df, mean_ranks = run_friedman_and_posthoc(
            cycle_metrics_df,
            metric="Weighted_Score",
            higher_is_better=True
        )

        friedman_file = SAVEPATH + f"friedman_summary_weighted_score_{extension}.csv"
        posthoc_file = SAVEPATH + f"posthoc_weighted_score_{extension}.csv"
        ranks_file = SAVEPATH + f"mean_ranks_weighted_score_{extension}.csv"

        friedman_summary.to_csv(friedman_file, index=False)
        posthoc_df.to_csv(posthoc_file, index=False)
        mean_ranks.to_csv(ranks_file, header=["Mean_Rank"])

        print(f"Saved Friedman summary to: {friedman_file}")
        print(f"Saved post-hoc results to: {posthoc_file}")
        print(f"Saved mean ranks to: {ranks_file}")

        # Optional: also run Friedman tests on individual metrics
        individual_tests = [
            ("Median_Abs_Jerk", False),
            ("Pearson_r", True),
            ("R_squared", True),
            ("MAE", False),
            ("RMSE", False),
            ("Abs_ROM_Error", False),
            ("Abs_Onset_Lag_sec", False)
        ]

        all_friedman = []
        all_posthoc = []

        for metric_name, hib in individual_tests:
            try:
                f_sum, p_hoc, ranks = run_friedman_and_posthoc(
                    cycle_metrics_df,
                    metric=metric_name,
                    higher_is_better=hib
                )

                all_friedman.append(f_sum)
                all_posthoc.append(p_hoc)

            except ValueError as e:
                print(f"Skipping {metric_name}: {e}")

        if len(all_friedman) > 0:
            all_friedman_df = pd.concat(all_friedman, ignore_index=True)
            all_friedman_df.to_csv(SAVEPATH + f"friedman_summary_all_metrics_{extension}.csv", index=False)

        if len(all_posthoc) > 0:
            all_posthoc_df = pd.concat(all_posthoc, ignore_index=True)
            all_posthoc_df.to_csv(SAVEPATH + f"posthoc_all_metrics_{extension}.csv", index=False)

