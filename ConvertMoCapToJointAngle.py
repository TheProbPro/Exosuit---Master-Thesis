import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt

# EMG processing imports
from SignalProcessing.Filtering import rt_filtering, rt_desired_Angle_lowpass
from SignalProcessing.Interpretors import ProportionalMyoelectricalControl as PMC
from Optimizations import optimize_1, optimize_2, optimize_4, optimize_5_pd, optimizer_6, EMG_Optimizer

# FILE_NAME = "C:\\Users\\nvigg\\Desktop\\MoCapData\\IMUEMGTest.csv"
# FILE_NAME = "C:\\Users\\nvigg\\Desktop\\MoCapData\\EMGMocapTest.csv"
# FILE_NAME = "C:\\Users\\nvigg\\Desktop\\MoCapData\\WithMotor.csv"

# output_file = "Outputs/MoCap/IMUEMGTest.csv"
# output_file = "Outputs/MoCap/EMGMocapTest.csv"
# output_file = "Outputs/MoCap/WithMotor.csv"

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
    "pDMP omega",
]

INPUT_MOCAP_DATA = [
    "Outputs/MoCapEMGData/ExoTest1.csv",
    "Outputs/MoCapEMGData/ExoTest2.csv",
    "Outputs/MoCapEMGData/ExoTest3.csv",
]

INPUT_EMG_DATA = [
    "Outputs/MoCapEMGData/ExoTest1_Trigno_2801.csv",
    "Outputs/MoCapEMGData/ExoTest2_Trigno_2801.csv",
    "Outputs/MoCapEMGData/ExoTest3_Trigno_2801.csv",
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
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_2":
            optimized_angle = optimize_2((4*np.pi), filtered_net_a, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_4":
            optimized_angle, delta_q_prev = optimize_4((2*np.pi), filtered_net_a, dt, optimized_angle_values[-1], delta_q_prev, THETA_MIN, THETA_MAX)
            optimized_angle_values.append(optimized_angle)
        elif optimizer == "optimizer_5_pd":
            optimized_angle, v = optimize_5_pd(filtered_net_a, v, dt, optimized_angle_values[-1], THETA_MIN, THETA_MAX, np.pi, 4, 0.1)
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
            print("pDMP optimization not implemented yet")
            pass
        elif optimizer == "pDMP coupled":
            print("pDMP coupled optimization not implemented yet")
            pass
        elif optimizer == "pDMP omega":
            print("pDMP omega optimization not implemented yet")
            pass
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
            plt.ylabel("Angle (degrees)")
            plt.title(f"MoCap vs EMG - {optimizer}")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()
            #

# if __name__ == "__main__":
#     # Read start time
#     with open(FILE_NAME, "r") as f:
#         first_line = f.readline()

#     parts = [p.strip() for p in first_line.split(",")]

#     # Find the index of "Capture Start Time"
#     idx = parts.index("Capture Start Time")

#     # The value is right after it
#     start_time = parts[idx + 1]

#     print("Start time:", start_time)

#     # Load the CSV
#     df = pd.read_csv(
#         FILE_NAME,
#         skiprows=3,        # skip metadata before row 4
#         header=[0, 3]      # row 4 (markers) + row 7 (axes)
#     )
#     # Load Data into numpy arrays
#     print(f"Columns in the CSV: {df.columns.tolist()}")

#     # Arrange data into relevant numpy arrays
#     timestamps = df[("Name", "Time (Seconds)")].to_numpy()
#     forearm_lower = df["Forearm:Lower"][["X", "Y", "Z"]].to_numpy()
#     forearm_upper = df["Forearm:Upper"][["X", "Y", "Z"]].to_numpy()
#     upperarm_middle = df["Upperarm:Middle"][["X", "Y", "Z"]].to_numpy()
#     upperarm_upper = df["Upperarm:Upper"][["X", "Y", "Z"]].to_numpy()
#     upperarm_elbow = df["Upperarm:Elbow"][["X", "Y", "Z"]].to_numpy()

#     #==========================================================================
#     # Convert timestamps into unix timestamps
#     # Fix format (replace dots in time part)
#     start_time_clean = start_time.replace(".", ":", 2)  # only replace first two dots
#     print("Cleaned start time:", start_time_clean)
    
#     # Parse to datetime
#     start_dt = datetime.strptime(start_time_clean, "%Y-%m-%d %H:%M:%S.%f")
#     # If you know this should be PM rather than AM
#     start_dt = start_dt + timedelta(hours=12)
#     # start_dt = start_dt.astimezone()
#     # start_dt = start_dt.replace(tzinfo=timezone.utc)
#     print("Parsed start datetime:", start_dt)
    
#     # Convert to unix timestamp
#     start_unix = start_dt.timestamp()
#     print("Start time as unix timestamp:", start_unix)

#     # add start_unix to relative timestamps to get absolute unix timestamps
#     absolute_timestamps = start_unix + timestamps

#     #==========================================================================
#     # Calculate elbow angle using upperarm_elbow, upperarm_upper, and forearm_lower
#     # Calculate vectors
#     v_upper = upperarm_upper - upperarm_elbow
#     v_forearm = forearm_lower - upperarm_elbow

#     # Calculate dot product
#     dot = np.sum(v_upper * v_forearm, axis=1)

#     # Calculate norms
#     norm_upper = np.linalg.norm(v_upper, axis=1)
#     norm_forearm = np.linalg.norm(v_forearm, axis=1)

#     # Avoid divide-by-zero
#     cos_theta = dot / (norm_upper * norm_forearm)
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)

#     # Angle in radians
#     elbow_angle_rad = np.arccos(cos_theta)

#     # Angle in degrees
#     elbow_angle_deg = np.degrees(elbow_angle_rad)
#     elbow_flexion_deg = 180 - elbow_angle_deg  # Assuming full extension is 0 degrees

#     # print("Calculated elbow angles (degrees):", elbow_angle_deg)
#     print("Calculated elbow flexion angles (degrees):", elbow_flexion_deg)

#     #==========================================================================
#     # Save results to a new CSV
#     output_df = pd.DataFrame({
#         "Timestamp": absolute_timestamps,
#         "Elbow_Angle_Rad": elbow_angle_rad,
#         "Elbow_Flexion_Rad": elbow_flexion_deg
#     })
#     if not os.path.exists("Outputs/MoCap"):
#         os.makedirs("Outputs/MoCap")
#     output_df.to_csv(output_file, index=False)
#     print(f"Saved calculated angles to {output_file}")