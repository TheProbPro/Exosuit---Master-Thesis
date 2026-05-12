import numpy as np
import pandas as pd
from pathlib import Path

# RW exosuit bicep parameters
# Usernames
USERNAMES = ["VictorBNielsen", "Kally", "ZichenWang", "Nicklas", "Magnus"]

# Path to files
BasePath = Path("Outputs/RWExosuitResults/")

# Save Path
SavePath = Path("Outputs/RWExosuitResults/Processed/")

# Subfolders
Subfolders = ["1", "2", "3"]

# Files to be processed
Files = ["trial_1.csv", "trial_2.csv", "trial_3.csv"]

# # Path to files
# BasePath = Path("Outputs/RWExosuitResultsVic/VictorBNielsen")

# # Save path
# SavePath = Path("Outputs/RWExosuitResultsVic/Processed/")

# # Files to be processed
# Files = ["OIAC/trial_1.csv", "OIAC/trial_2.csv", "OIAC/trial_3.csv", "PID/trial_1.csv", "PID/trial_2.csv", "PID/trial_3.csv"]

def clean_numeric_column(col):
    return (
        col.astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .astype(float)
    )

if __name__ == "__main__":
    for username in USERNAMES:
        for file in Files:
            for subfolder in Subfolders:
                # Read the data
                input_file = BasePath / username / subfolder / file
                if not input_file.exists():
                    print(f"File {input_file} does not exist. Skipping.")
                    continue

                data = pd.read_csv(input_file)

                #Load the data into numpy arrays
                time = data['t_qd'].astype(float).values
                qd = clean_numeric_column(data['qd_rad']).values
                q = clean_numeric_column(data['q_rad']).values
                tau = clean_numeric_column(data['tau']).values

                if len(time) < 1 or len(qd) < 1 or len(q) < 1 or len(tau) < 1:
                    print(f"File {input_file} has insufficient data. Skipping.")
                    print(f"time length: {len(time)}, qd length: {len(qd)}, q length: {len(q)}, tau length: {len(tau)}")
                    print(f"file Header: {data.columns}")
                    continue

                # Process the data
                # Calculate position error
                position_error = qd - q

                # Calculate velocity, acceleration, and jerk
                velocity = np.gradient(q, time)
                acceleration = np.gradient(velocity, time)
                jerk = np.gradient(acceleration, time)

                print(f"Mean jerk for {file}: {np.mean(np.abs(jerk))}")
                print(f"median jerk for {file}: {np.median(np.abs(jerk))}")

                velocity_d = np.gradient(qd, time)
                acceleration_d = np.gradient(velocity_d, time)
                jerk_d = np.gradient(acceleration_d, time)

                print(f"desired Mean jerk for {file}: {np.mean(np.abs(jerk_d))}")
                print(f"desired median jerk for {file}: {np.median(np.abs(jerk_d))}")

                # Save the processed data to a new CSV file
                processed_data = pd.DataFrame({
                    'time': time,
                    'qd': qd,
                    'q': q,
                    'tau': tau,
                    'position_error': position_error,
                    'jerk': jerk,
                    'jerk_d': jerk_d
                })
                if not SavePath.exists():
                    SavePath.mkdir(parents=True)
                output_file = SavePath / username / subfolder / file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                processed_data.to_csv(output_file, index=False)
