import numpy as np
import pandas as pd
from pathlib import Path

Folders = [
    Path("Outputs/PreplannedThresholdAANRAN/"),
    Path("Outputs/PreplannedUpDownAANRAN/"),
    Path("Outputs/EMGThresholdAANRAN/"),
    Path("Outputs/EMGThresholdAANRAN-TrainedOnPreplanned/"),
    Path("Outputs/EMGUpDownAANRAN/"),
    Path("Outputs/EMGUpDownAANRAN-TrainedOnPreplanned/")
    ]

filename = "Average_Trial_Data.csv"

if __name__ == "__main__":
    for folder in Folders:

        file_path = folder / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            position_errors_deg = df["Average_Position_Error_deg"].values
            position_errors_rad = df["Average_Position_Error_rad"].values
            Jerks_deg_per_s3 = df["Average_Jerk_deg_per_s3"].values
            Jerks_rad_per_s3 = df["Average_Jerk_rad_per_s3"].values
            Total_Torques_Nm = df["Average_Total_Torque_Nm"].values
            Feedback_Torques_Nm = df["Average_Feedback_Torque_Nm"].values
            Feedforward_Torques_Nm = df["Average_Feedforward_Torque_Nm"].values

            # Calculate averages
            avg_position_error_deg = np.mean(position_errors_deg)
            avg_position_error_rad = np.mean(position_errors_rad)
            avg_jerk_deg_per_s3 = np.mean(Jerks_deg_per_s3)
            avg_jerk_rad_per_s3 = np.mean(Jerks_rad_per_s3)
            avg_total_torque_Nm = np.mean(Total_Torques_Nm)
            avg_feedback_torque_Nm = np.mean(Feedback_Torques_Nm)
            avg_feedforward_torque_Nm = np.mean(Feedforward_Torques_Nm)

            # Calclate standard deviations
            std_position_error_deg = np.std(position_errors_deg)
            std_position_error_rad = np.std(position_errors_rad)
            std_jerk_deg_per_s3 = np.std(Jerks_deg_per_s3)
            std_jerk_rad_per_s3 = np.std(Jerks_rad_per_s3)
            std_total_torque_Nm = np.std(Total_Torques_Nm)
            std_feedback_torque_Nm = np.std(Feedback_Torques_Nm)
            std_feedforward_torque_Nm = np.std(Feedforward_Torques_Nm)

            print("#" * 50)

            print(f"Folder: {folder.name}")
            
            print(f"Average Position Error: {avg_position_error_deg} deg, {avg_position_error_rad} rad")
            print(f"Std Dev Position Error: {std_position_error_deg} deg, {std_position_error_rad} rad")

            print(f"Average Jerk: {avg_jerk_deg_per_s3} deg/s³, {avg_jerk_rad_per_s3} rad/s³")
            print(f"Std Dev Jerk: {std_jerk_deg_per_s3} deg/s³, {std_jerk_rad_per_s3} rad/s³")
            
            print(f"Average Total Torque: {avg_total_torque_Nm} Nm")
            print(f"Std Dev Total Torque: {std_total_torque_Nm} Nm")
            
            print(f"Average Feedback Torque: {avg_feedback_torque_Nm} Nm")
            print(f"Std Dev Feedback Torque: {std_feedback_torque_Nm} Nm")
            
            print(f"Average Feedforward Torque: {avg_feedforward_torque_Nm} Nm")
            print(f"Std Dev Feedforward Torque: {std_feedforward_torque_Nm} Nm")

            print("#" * 50 + "\n\n")


