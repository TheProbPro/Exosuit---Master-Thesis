import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Load Initialize path to parent folder and file name
    # folder = Path("Outputs/PreplannedThresholdAANRAN/")
    # folder = Path("Outputs/PreplannedUpDownAANRAN/")
    # folder = Path("Outputs/EMGThresholdAANRAN/")
    # folder = Path("Outputs/EMGThresholdAANRAN-TrainedOnPreplanned/")
    # folder = Path("Outputs/EMGUpDownAANRAN/")
    folder = Path("Outputs/EMGUpDownAANRAN-TrainedOnPreplanned/")

    filename = "Final_Trial_Data.csv"

    time_vector = []
    control_mode = []
    Actual_Position_deg = []
    Desired_Position_deg = []
    position_error_deg = []
    Feedback_Torque_Nm = []
    Feedforward_Torque_Nm = []
    Total_Torque_Nm = []
    Jerk_deg_per_s3 = []
    n_files = 0

    # Open the subfolders of the parent folder and read the csv file
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            file_path = subfolder / filename
            if file_path.exists():
                n_files += 1
                df = pd.read_csv(file_path)
                time_vector.append(df["Time_s"].values)
                control_mode.append(df["Control_Mode"].values)
                Actual_Position_deg.append(df["Actual_Position_deg"].values)
                Desired_Position_deg.append(df["Desired_Position_deg"].values)
                position_error_deg.append(df["Position_Error_deg"].values)
                Feedback_Torque_Nm.append(df["Feedback_Torque_Nm"].values)
                Feedforward_Torque_Nm.append(df["Feedforward_Torque_Nm"].values)
                Total_Torque_Nm.append(df["Total_Torque_Nm"].values)
                Jerk_deg_per_s3.append(df["Jerk_deg_per_s3"].values)
                print(f"Found file: {file_path}")
            else:
                print(f"File {filename} not found in {subfolder.name}")
    
    # Check length of lists
    shortest = 200000
    for i in range(n_files):
        length = len(time_vector[i])
        print(f"length of vector {i}: {length} samples")
        if length < shortest:
            shortest = length

    # Clip data to the shortest length across all trials
    for i in range(n_files):
        time_vector[i] = time_vector[i][:shortest]
        control_mode[i] = control_mode[i][:shortest]
        Actual_Position_deg[i] = Actual_Position_deg[i][:shortest]
        Desired_Position_deg[i] = Desired_Position_deg[i][:shortest]
        position_error_deg[i] = position_error_deg[i][:shortest]
        Feedback_Torque_Nm[i] = Feedback_Torque_Nm[i][:shortest]
        Feedforward_Torque_Nm[i] = Feedforward_Torque_Nm[i][:shortest]
        Total_Torque_Nm[i] = Total_Torque_Nm[i][:shortest]
        Jerk_deg_per_s3[i] = Jerk_deg_per_s3[i][:shortest]


    # Create a alternate vector containing radian values instead of degrees for the relevant variables
    Actual_Position_rad = [np.deg2rad(pos) for pos in Actual_Position_deg]
    Desired_Position_rad = [np.deg2rad(pos) for pos in Desired_Position_deg]
    position_error_rad = [np.deg2rad(err) for err in position_error_deg]
    jerk_rad_per_s3 = [np.deg2rad(jerk) for jerk in Jerk_deg_per_s3]
    
    # Calulate average value, and standard deviation, for each timestep across all trials
    actual_position_deg_matrix = np.vstack(Actual_Position_deg)
    average_actual_position_deg = np.mean(actual_position_deg_matrix, axis=0)
    standard_deviation_actual_position_deg = np.std(actual_position_deg_matrix, axis=0)

    actual_position_rad_matrix = np.vstack(Actual_Position_rad)
    average_actual_position_rad = np.mean(actual_position_rad_matrix, axis=0)
    standard_deviation_actual_position_rad = np.std(actual_position_rad_matrix, axis=0)

    Desired_Position_deg_matrix = np.vstack(Desired_Position_deg)
    average_desired_position_deg = np.mean(Desired_Position_deg_matrix, axis=0)
    standard_deviation_desired_position_deg = np.std(Desired_Position_deg_matrix, axis=0)

    Desired_Position_rad_matrix = np.vstack(Desired_Position_rad)
    average_desired_position_rad = np.mean(Desired_Position_rad_matrix, axis=0)
    standard_deviation_desired_position_rad = np.std(Desired_Position_rad_matrix, axis=0)

    position_error_deg_matrix = np.vstack(position_error_deg)
    average_position_error_deg = np.mean(position_error_deg_matrix, axis=0)
    standard_deviation_position_error_deg = np.std(position_error_deg_matrix, axis=0)

    position_error_rad_matrix = np.vstack(position_error_rad)
    average_position_error_rad = np.mean(position_error_rad_matrix, axis=0)
    standard_deviation_position_error_rad = np.std(position_error_rad_matrix, axis=0)

    Feedback_Torque_Nm_matrix = np.vstack(Feedback_Torque_Nm)
    average_Feedback_Torque_Nm = np.mean(Feedback_Torque_Nm_matrix, axis=0)
    standard_deviation_feedback_torque_Nm = np.std(Feedback_Torque_Nm_matrix, axis=0)

    Feedforward_Torque_Nm_matrix = np.vstack(Feedforward_Torque_Nm)
    average_Feedforward_Torque_Nm = np.mean(Feedforward_Torque_Nm_matrix, axis=0)
    standard_deviation_feedforward_torque_Nm = np.std(Feedforward_Torque_Nm_matrix, axis=0)

    Total_Torque_Nm_matrix = np.vstack(Total_Torque_Nm)
    average_Total_Torque_Nm = np.mean(Total_Torque_Nm_matrix, axis=0)
    standard_deviation_total_torque_Nm = np.std(Total_Torque_Nm_matrix, axis=0)

    jerk_deg_per_s3_matrix = np.vstack(Jerk_deg_per_s3)
    average_jerk_deg_per_s3 = np.mean(jerk_deg_per_s3_matrix, axis=0)
    standard_deviation_jerk_deg_per_s3 = np.std(jerk_deg_per_s3_matrix, axis=0)
    
    jerk_rad_per_s3_matrix = np.vstack(jerk_rad_per_s3)
    average_jerk_rad_per_s3 = np.mean(jerk_rad_per_s3_matrix, axis=0)
    standard_deviation_jerk_rad_per_s3 = np.std(jerk_rad_per_s3_matrix, axis=0)

    # Create a new average dataframe to store the values in and save it as a csv file in the parent folder
    average_df = pd.DataFrame({
        "Time_s": time_vector[0],
        "Average_Actual_Position_deg": average_actual_position_deg,
        "Standard_Deviation_Actual_Position_deg": standard_deviation_actual_position_deg,
        "Average_Desired_Position_deg": average_desired_position_deg,
        "Standard_Deviation_Desired_Position_deg": standard_deviation_desired_position_deg,
        "Average_Position_Error_deg": average_position_error_deg,
        "Standard_Deviation_Position_Error_deg": standard_deviation_position_error_deg,
        "Average_Actual_Position_rad": average_actual_position_rad,
        "Standard_Deviation_Actual_Position_rad": standard_deviation_actual_position_rad,
        "Average_Desired_Position_rad": average_desired_position_rad,
        "Standard_Deviation_Desired_Position_rad": standard_deviation_desired_position_rad,
        "Average_Position_Error_rad": average_position_error_rad,
        "Standard_Deviation_Position_Error_rad": standard_deviation_position_error_rad,
        "Average_Feedback_Torque_Nm": average_Feedback_Torque_Nm,
        "Standard_Deviation_Feedback_Torque_Nm": standard_deviation_feedback_torque_Nm,
        "Average_Feedforward_Torque_Nm": average_Feedforward_Torque_Nm,
        "Standard_Deviation_Feedforward_Torque_Nm": standard_deviation_feedforward_torque_Nm,
        "Average_Total_Torque_Nm": average_Total_Torque_Nm,
        "Standard_Deviation_Total_Torque_Nm": standard_deviation_total_torque_Nm,
        "Average_Jerk_deg_per_s3": average_jerk_deg_per_s3,
        "Standard_Deviation_Jerk_deg_per_s3": standard_deviation_jerk_deg_per_s3,
        "Average_Jerk_rad_per_s3": average_jerk_rad_per_s3,
        "Standard_Deviation_Jerk_rad_per_s3": standard_deviation_jerk_rad_per_s3
    })
    average_df.to_csv(folder / "Average_Trial_Data.csv", index=False)