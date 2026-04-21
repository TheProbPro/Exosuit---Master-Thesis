import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os

# FILE_NAME = "C:\\Users\\nvigg\\Desktop\\MoCapData\\IMUEMGTest.csv"
# FILE_NAME = "C:\\Users\\nvigg\\Desktop\\MoCapData\\EMGMocapTest.csv"
FILE_NAME = "C:\\Users\\nvigg\\Desktop\\MoCapData\\WithMotor.csv"


# output_file = "Outputs/MoCap/IMUEMGTest.csv"
# output_file = "Outputs/MoCap/EMGMocapTest.csv"
output_file = "Outputs/MoCap/WithMotor.csv"

if __name__ == "__main__":
    # Read start time
    with open(FILE_NAME, "r") as f:
        first_line = f.readline()

    parts = [p.strip() for p in first_line.split(",")]

    # Find the index of "Capture Start Time"
    idx = parts.index("Capture Start Time")

    # The value is right after it
    start_time = parts[idx + 1]

    print("Start time:", start_time)

    # Load the CSV
    df = pd.read_csv(
        FILE_NAME,
        skiprows=3,        # skip metadata before row 4
        header=[0, 3]      # row 4 (markers) + row 7 (axes)
    )
    # Load Data into numpy arrays
    print(f"Columns in the CSV: {df.columns.tolist()}")

    # Arrange data into relevant numpy arrays
    timestamps = df[("Name", "Time (Seconds)")].to_numpy()
    forearm_lower = df["Forearm:Lower"][["X", "Y", "Z"]].to_numpy()
    forearm_upper = df["Forearm:Upper"][["X", "Y", "Z"]].to_numpy()
    upperarm_middle = df["Upperarm:Middle"][["X", "Y", "Z"]].to_numpy()
    upperarm_upper = df["Upperarm:Upper"][["X", "Y", "Z"]].to_numpy()
    upperarm_elbow = df["Upperarm:Elbow"][["X", "Y", "Z"]].to_numpy()

    #==========================================================================
    # Convert timestamps into unix timestamps
    # Fix format (replace dots in time part)
    start_time_clean = start_time.replace(".", ":", 2)  # only replace first two dots
    print("Cleaned start time:", start_time_clean)
    
    # Parse to datetime
    start_dt = datetime.strptime(start_time_clean, "%Y-%m-%d %H:%M:%S.%f")
    # If you know this should be PM rather than AM
    start_dt = start_dt + timedelta(hours=12)
    # start_dt = start_dt.astimezone()
    # start_dt = start_dt.replace(tzinfo=timezone.utc)
    print("Parsed start datetime:", start_dt)
    
    # Convert to unix timestamp
    start_unix = start_dt.timestamp()
    print("Start time as unix timestamp:", start_unix)

    # add start_unix to relative timestamps to get absolute unix timestamps
    absolute_timestamps = start_unix + timestamps

    #==========================================================================
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

    # print("Calculated elbow angles (degrees):", elbow_angle_deg)
    print("Calculated elbow flexion angles (degrees):", elbow_flexion_deg)

    #==========================================================================
    # Save results to a new CSV
    output_df = pd.DataFrame({
        "Timestamp": absolute_timestamps,
        "Elbow_Angle_Rad": elbow_angle_rad,
        "Elbow_Flexion_Rad": elbow_flexion_deg
    })
    if not os.path.exists("Outputs/MoCap"):
        os.makedirs("Outputs/MoCap")
    output_df.to_csv(output_file, index=False)
    print(f"Saved calculated angles to {output_file}")

