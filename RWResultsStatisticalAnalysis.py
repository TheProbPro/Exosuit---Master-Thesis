import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Load Initialize path to parent folder and file name
    # folder = Path("Outputs/RWExosuitResults/Processed")
    folder = Path("Outputs/IEEE/Processed")
    # usernames = ["VictorBNielsen", "Kally", "ZichenWang", "Nicklas"]#, "Magnus"]
    usernames = ["VictorBNielsen", "Kally", "ZichenWang", "Valentina", "Cavan", "Annonomous"]
    # subfolders = ["1", "2", "3"]
    subfolders = ["Optim2", "Optim6", "pDMP"]
    subsubfolers = ["OIAC", "PID"]
    subsubsubfolders = ["NonePeriodic", "Periodic"]
    filenames = ["trial_1.csv", "trial_2.csv", "trial_3.csv"]

    time_vector = []
    Actual_Position_rad = []
    Desired_Position_rad = []
    position_error_rad = []
    Total_Torque_Nm = []
    Jerk_rad_per_s3 = []
    Desired_Jerk_rad_per_s3 = []
    n_files = 0

    # Open the subfolders of the parent folder and read the csv file
    for filename in filenames:
        for subfolder in subfolders:
            for subsubfolder in subsubfolers:
                for subsubsubfolder in subsubsubfolders:
                    for username in usernames:
                        file_path = folder / subfolder / subsubfolder / username / subsubsubfolder / filename
                        if file_path.exists():
                            n_files += 1
                            df = pd.read_csv(file_path)
                            time_vector.append(df["time"].values)
                            Actual_Position_rad.append(df["q"].values)
                            Desired_Position_rad.append(df["qd"].values)
                            position_error_rad.append(df["position_error"].values)
                            Total_Torque_Nm.append(df["tau"].values)
                            Jerk_rad_per_s3.append(df["jerk"].values)
                            Desired_Jerk_rad_per_s3.append(df["jerk_d"].values)
                            print(f"Found file: {file_path}")
                        else:
                            print(f"File {filename} not found in {username}/{subfolder}")

                    if n_files == 0:
                        raise FileNotFoundError("No trial files were found. Check your folder path and filenames.")
                    
                    # Check length of lists
                    shortest = 200000
                    for i in range(n_files):
                        length = len(time_vector[i])
                        print(f"length of vector {i}: {length} samples")
                        if length < shortest:
                            shortest = length
                    print(f"Shortest vector length: {shortest} samples")

                    for i in range(n_files):
                        time_vector[i] = time_vector[i][:shortest]
                        Actual_Position_rad[i] = Actual_Position_rad[i][:shortest]
                        Desired_Position_rad[i] = Desired_Position_rad[i][:shortest]
                        position_error_rad[i] = position_error_rad[i][:shortest]
                        Total_Torque_Nm[i] = Total_Torque_Nm[i][:shortest]
                        Jerk_rad_per_s3[i] = Jerk_rad_per_s3[i][:shortest]
                        Desired_Jerk_rad_per_s3[i] = Desired_Jerk_rad_per_s3[i][:shortest]

                    # Calulate average value, and standard deviation, for each timestep across all trials
                    actual_position_rad_matrix = np.vstack(Actual_Position_rad)
                    average_actual_position_rad = np.mean(actual_position_rad_matrix, axis=0)
                    standard_deviation_actual_position_rad = np.std(actual_position_rad_matrix, axis=0)

                    Desired_Position_rad_matrix = np.vstack(Desired_Position_rad)
                    average_desired_position_rad = np.mean(Desired_Position_rad_matrix, axis=0)
                    standard_deviation_desired_position_rad = np.std(Desired_Position_rad_matrix, axis=0)

                    position_error_rad_matrix = np.vstack(position_error_rad)
                    average_position_error_rad = np.mean(position_error_rad_matrix, axis=0)
                    standard_deviation_position_error_rad = np.std(position_error_rad_matrix, axis=0)

                    Total_Torque_Nm_matrix = np.vstack(Total_Torque_Nm)
                    average_Total_Torque_Nm = np.mean(Total_Torque_Nm_matrix, axis=0)
                    standard_deviation_total_torque_Nm = np.std(Total_Torque_Nm_matrix, axis=0)

                    jerk_rad_per_s3_matrix = np.vstack(Jerk_rad_per_s3)
                    average_jerk_rad_per_s3 = np.mean(jerk_rad_per_s3_matrix, axis=0)
                    standard_deviation_jerk_rad_per_s3 = np.std(jerk_rad_per_s3_matrix, axis=0)
                    
                    Desired_jerk_rad_per_s3_matrix = np.vstack(Desired_Jerk_rad_per_s3)
                    average_desired_jerk_rad_per_s3 = np.mean(Desired_jerk_rad_per_s3_matrix, axis=0)
                    standard_deviation_desired_jerk_rad_per_s3 = np.std(Desired_jerk_rad_per_s3_matrix, axis=0)

                    # Create new timevector to ensure it still spans 10 seconds
                    time_vector = np.linspace(0, 30, shortest)

                    average_df = pd.DataFrame({
                        "Time_s": time_vector,
                        "Average_Actual_Position_rad": average_actual_position_rad,
                        "Standard_Deviation_Actual_Position_rad": standard_deviation_actual_position_rad,
                        "Average_Desired_Position_rad": average_desired_position_rad,
                        "Standard_Deviation_Desired_Position_rad": standard_deviation_desired_position_rad,
                        "Average_Position_Error_rad": average_position_error_rad,
                        "Standard_Deviation_Position_Error_rad": standard_deviation_position_error_rad,
                        "Average_Total_Torque_Nm": average_Total_Torque_Nm,
                        "Standard_Deviation_Total_Torque_Nm": standard_deviation_total_torque_Nm,
                        "Average_Desired_Jerk_rad_per_s3": average_desired_jerk_rad_per_s3,
                        "Standard_Deviation_Desired_Jerk_rad_per_s3": standard_deviation_desired_jerk_rad_per_s3,
                        "Average_Jerk_rad_per_s3": average_jerk_rad_per_s3,
                        "Standard_Deviation_Jerk_rad_per_s3": standard_deviation_jerk_rad_per_s3
                    })
                    average_df.to_csv(folder / f"Average_{Path(filename).stem}_{subfolder}_{subsubfolder}_{subsubsubfolder}_Data.csv", index=False)

                    # clear vectors
                    time_vector = []
                    Actual_Position_rad.clear()
                    Desired_Position_rad.clear()
                    position_error_rad.clear()
                    Total_Torque_Nm.clear()
                    Jerk_rad_per_s3.clear()
                    Desired_Jerk_rad_per_s3.clear()
                    n_files = 0

    
    # for username in usernames:
    #     for subfolder in subfolders:
    #         for filename in filenames:
    #             file_path = folder / username / subfolder / filename

    #             if file_path.exists():
    #                 n_files += 1
    #                 df = pd.read_csv(file_path)
    #                 time_vector.append(df["time"].values)
    #                 Actual_Position_rad.append(df["q"].values)
    #                 Desired_Position_rad.append(df["qd"].values)
    #                 position_error_rad.append(df["position_error"].values)
    #                 Total_Torque_Nm.append(df["tau"].values)
    #                 Jerk_rad_per_s3.append(df["jerk"].values)
    #                 Desired_Jerk_rad_per_s3.append(df["jerk_d"].values)
    #                 print(f"Found file: {file_path}")
    #             else:
    #                 print(f"File {filename} not found in {username}/{subfolder}")

    # if n_files == 0:
    #     raise FileNotFoundError("No trial files were found. Check your folder path and filenames.")
    
    # Check length of lists
    # shortest = 200000
    # for i in range(n_files):
    #     length = len(time_vector[i])
    #     print(f"length of vector {i}: {length} samples")
    #     if length < shortest:
    #         shortest = length
    # print(f"Shortest vector length: {shortest} samples")

    # Clip data to the shortest length across all trials
    # for i in range(n_files):
    #     time_vector[i] = time_vector[i][:shortest]
    #     Actual_Position_rad[i] = Actual_Position_rad[i][:shortest]
    #     Desired_Position_rad[i] = Desired_Position_rad[i][:shortest]
    #     position_error_rad[i] = position_error_rad[i][:shortest]
    #     Total_Torque_Nm[i] = Total_Torque_Nm[i][:shortest]
    #     Jerk_rad_per_s3[i] = Jerk_rad_per_s3[i][:shortest]
    #     Desired_Jerk_rad_per_s3[i] = Desired_Jerk_rad_per_s3[i][:shortest]

    # # Calulate average value, and standard deviation, for each timestep across all trials
    # actual_position_rad_matrix = np.vstack(Actual_Position_rad)
    # average_actual_position_rad = np.mean(actual_position_rad_matrix, axis=0)
    # standard_deviation_actual_position_rad = np.std(actual_position_rad_matrix, axis=0)

    # Desired_Position_rad_matrix = np.vstack(Desired_Position_rad)
    # average_desired_position_rad = np.mean(Desired_Position_rad_matrix, axis=0)
    # standard_deviation_desired_position_rad = np.std(Desired_Position_rad_matrix, axis=0)

    # position_error_rad_matrix = np.vstack(position_error_rad)
    # average_position_error_rad = np.mean(position_error_rad_matrix, axis=0)
    # standard_deviation_position_error_rad = np.std(position_error_rad_matrix, axis=0)

    # Total_Torque_Nm_matrix = np.vstack(Total_Torque_Nm)
    # average_Total_Torque_Nm = np.mean(Total_Torque_Nm_matrix, axis=0)
    # standard_deviation_total_torque_Nm = np.std(Total_Torque_Nm_matrix, axis=0)

    # jerk_rad_per_s3_matrix = np.vstack(Jerk_rad_per_s3)
    # average_jerk_rad_per_s3 = np.mean(jerk_rad_per_s3_matrix, axis=0)
    # standard_deviation_jerk_rad_per_s3 = np.std(jerk_rad_per_s3_matrix, axis=0)
    
    # Desired_jerk_rad_per_s3_matrix = np.vstack(Desired_Jerk_rad_per_s3)
    # average_desired_jerk_rad_per_s3 = np.mean(Desired_jerk_rad_per_s3_matrix, axis=0)
    # standard_deviation_desired_jerk_rad_per_s3 = np.std(Desired_jerk_rad_per_s3_matrix, axis=0)

    # # Create new timevector to ensure it still spans 10 seconds
    # time_vector = np.linspace(0, 30, shortest)

    # Create a new average dataframe to store the values in and save it as a csv file in the parent folder
    # average_df = pd.DataFrame({
    #     "Time_s": time_vector,
    #     "Average_Actual_Position_rad": average_actual_position_rad,
    #     "Standard_Deviation_Actual_Position_rad": standard_deviation_actual_position_rad,
    #     "Average_Desired_Position_rad": average_desired_position_rad,
    #     "Standard_Deviation_Desired_Position_rad": standard_deviation_desired_position_rad,
    #     "Average_Position_Error_rad": average_position_error_rad,
    #     "Standard_Deviation_Position_Error_rad": standard_deviation_position_error_rad,
    #     "Average_Total_Torque_Nm": average_Total_Torque_Nm,
    #     "Standard_Deviation_Total_Torque_Nm": standard_deviation_total_torque_Nm,
    #     "Average_Desired_Jerk_rad_per_s3": average_desired_jerk_rad_per_s3,
    #     "Standard_Deviation_Desired_Jerk_rad_per_s3": standard_deviation_desired_jerk_rad_per_s3,
    #     "Average_Jerk_rad_per_s3": average_jerk_rad_per_s3,
    #     "Standard_Deviation_Jerk_rad_per_s3": standard_deviation_jerk_rad_per_s3
    # })
    # average_df.to_csv(folder / "Average_Trial_Data.csv", index=False)