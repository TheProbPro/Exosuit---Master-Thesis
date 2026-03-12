import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------
file1 = "Outputs/IMU_EMG_MoCap_Test/imu_angles.csv"   # has unix timestamps from time.time()
file2 = "Outputs/IMU_EMG_MoCap_Test/emg_desired_angles.csv"   # has unix timestamps from time.time()
file3 = "Outputs/IMU_EMG_MoCap_Test/MoCap.csv"   # has start time + relative timestamps

# Change these column names to match your CSVs
file1_time_col = "Timestamp"
file1_data_col = "Elbow_Angle"

file2_time_col = "Timestamp"
file2_data_col = "Desired_Angle"

# For file3:
# Example columns:
# start_time, rel_time, value
file3_start_col = "start_time"
file3_rel_col = "rel_time"
file3_data_col = "value"


# ----------------------------
# HELPER: parse relative time
# ----------------------------
def parse_relative_time(rel_str):
    """
    Convert strings like:
      '0:005' -> 0.005 seconds
      '0:01'  -> 0.01 seconds
      '0:015' -> 0.015 seconds
      '1:250' -> 1.250 seconds
    into float seconds.
    """
    rel_str = str(rel_str).strip()
    sec_part, frac_part = rel_str.split(":")
    return float(sec_part) + float("0." + frac_part)


# ----------------------------
# HELPER: parse start time
# ----------------------------
def parse_start_time(start_str):
    """
    Convert '2026.03.12 12:23:00:00' into unix timestamp.

    Adjust the format string if your actual timestamp format differs.
    This assumes:
      year.month.day hour:minute:second:centisecond
    """
    start_str = str(start_str).strip()
    dt = datetime.strptime(start_str, "%Y.%m.%d %H:%M:%S:%f")
    return dt.timestamp()


# ----------------------------
# LOAD FILES
# ----------------------------
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Make sure numeric timestamps are floats
df1[file1_time_col] = pd.to_numeric(df1[file1_time_col], errors="coerce")
df2[file2_time_col] = pd.to_numeric(df2[file2_time_col], errors="coerce")

# ----------------------------
# CONVERT FILE 3 TO UNIX TIME
# ----------------------------
# If the start time is repeated on every row, just use the first one
start_unix = parse_start_time(df3[file3_start_col].iloc[0])

df3["rel_seconds"] = df3[file3_rel_col].apply(parse_relative_time)
df3["timestamp_unix"] = start_unix + df3["rel_seconds"]

# ----------------------------
# OPTIONAL: convert to relative time for plotting
# ----------------------------
# Use the earliest timestamp across all files as t=0
global_start = min(
    df1[file1_time_col].min(),
    df2[file2_time_col].min(),
    df3["timestamp_unix"].min()
)

df1["t_sync"] = df1[file1_time_col] - global_start
df2["t_sync"] = df2[file2_time_col] - global_start
df3["t_sync"] = df3["timestamp_unix"] - global_start

# ----------------------------
# PLOT
# ----------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df1["t_sync"], df1[file1_data_col], label="Sensor 1")
axes[0].set_ylabel("Value")
axes[0].set_title("Sensor 1")
axes[0].grid(True)

axes[1].plot(df2["t_sync"], df2[file2_data_col], label="Sensor 2")
axes[1].set_ylabel("Value")
axes[1].set_title("Sensor 2")
axes[1].grid(True)

axes[2].plot(df3["t_sync"], df3[file3_data_col], label="Sensor 3")
axes[2].set_xlabel("Time since first sample [s]")
axes[2].set_ylabel("Value")
axes[2].set_title("Sensor 3")
axes[2].grid(True)

plt.tight_layout()
plt.show()