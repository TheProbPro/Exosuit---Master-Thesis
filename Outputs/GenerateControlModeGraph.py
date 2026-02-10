import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

Folders = [
    Path("Outputs/PreplannedThresholdAANRAN/"),
    Path("Outputs/PreplannedUpDownAANRAN/"),
    Path("Outputs/EMGThresholdAANRAN/"),
    Path("Outputs/EMGThresholdAANRAN-TrainedOnPreplanned/"),
    Path("Outputs/EMGUpDownAANRAN/"),
    Path("Outputs/EMGUpDownAANRAN-TrainedOnPreplanned/")
    ]

SAVE_PATH = Path("Outputs/PythonGraphs/ControlMode/")

class ControlMode:
    """控制模式定义"""
    AAN = "assist_as_needed"  # 辅助模式
    RAN = "resist_as_needed"  # 阻力模式

if __name__ == "__main__":
    # For plotting later
    time_vectors = []
    control_modes = []
    desired_positions = []
    actual_positions = []
    pathnames = []
    n_folders = 0
    n_subfolders = []

    folder_start_idx = []  ### FIX: start index into the flat arrays for each folder

    # Load all data into the vectors
    filename = "Final_Trial_Data.csv"
    for folder in Folders:
        folder_start_idx.append(len(time_vectors))  ### FIX: record where this folder's datasets begin
        n_subfolder = 0
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                file_path = subfolder / filename
                if file_path.exists():
                    n_subfolder += 1

                    df = pd.read_csv(file_path)
                    time_vector = df["Time_s"].values
                    control_mode = df["Control_Mode"].values
                    desired_position = np.deg2rad(df["Desired_Position_deg"].values)
                    actual_position = np.deg2rad(df["Actual_Position_deg"].values)

                    time_vectors.append(time_vector)
                    control_modes.append(control_mode)
                    desired_positions.append(desired_position)
                    actual_positions.append(actual_position)
        pathnames.append(folder.name)
        n_subfolders.append(n_subfolder)
        n_folders += 1

    print(f"Loaded data from {n_folders} folders with subfolder counts: {n_subfolders}")

    # Set font sizes
    plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    })

    for i in range(n_folders):
        fig, axs = plt.subplots(n_subfolders[i], 1, sharex=True, figsize=(10, 6), constrained_layout=True)
        axs[0].set_title(f"Control Mode: AAN=Blue, RAN=Red")
        base = folder_start_idx[i]  ### FIX: starting index for this folder in the flat lists
        for j in range(n_subfolders[i]):
            idx = base + j          ### FIX: correct dataset index for (folder i, subfolder j)
            current_mode = control_modes[idx][0]
            start = 0
            for k in range(len(control_modes[idx])):
                if control_modes[idx][k] != current_mode:
                    color = 'lightblue' if current_mode == ControlMode.AAN else 'lightcoral'
                    axs[j].axvspan(time_vectors[idx][start], time_vectors[idx][k], color=color, alpha=0.3)
                    current_mode = control_modes[idx][k]
                    start = k
            axs[j].axvspan(time_vectors[idx][start], time_vectors[idx][-1], color='lightblue' if current_mode == ControlMode.AAN else 'lightcoral', alpha=0.3)
            
            axs[j].plot(time_vectors[idx], actual_positions[idx], label='Actual', linewidth=2, color='red')
            axs[j].plot(time_vectors[idx], desired_positions[idx], label='Desired', linestyle='--', linewidth=2, color='blue')
            axs[j].set_ylabel('Position [rad]')
            axs[j].legend(loc='best', frameon=False, framealpha=0.8)
        axs[n_subfolders[i]-1].set_xlabel('Time [s]')
        axs[n_subfolders[i]-1].set_xlim(0, 10)
        
        # Save the figure to savepath
        if not SAVE_PATH.exists():
            SAVE_PATH.mkdir(parents=True)
        save_filename = f"{SAVE_PATH}/{pathnames[i]}.pdf"
        plt.savefig(save_filename, format='pdf', bbox_inches='tight', pad_inches=0.02)

        plt.show()
