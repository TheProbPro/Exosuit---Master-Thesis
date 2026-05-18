import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    
    'font.size': 10,          # default text size
    'axes.titlesize': 14,     # title
    'axes.labelsize': 12,     # x and y labels
    'xtick.labelsize': 10,    # x tick labels
    'ytick.labelsize': 10,    # y tick labels
    'legend.fontsize': 10,    
    'figure.titlesize': 16
})

RESULTS_PATH = "Outputs/Results/"
VEUSZ_CSV_PATH = "Outputs/Results/VeuszPlotting/"
GRAPH_SAVE_PATH = "Outputs/Results/Graphs/"

df_radar = pd.DataFrame({
    "Integrator": [],
    "MAE": [],
    "RMSE": [],
    "Correlation": [],
    "R_squared": [],
    "ROM_error": [],
    "median_jerk": [],
    "Median_Onset_Lag_sec": [],
    "Median_Peak_Lag_sec": []
})

OPTIMIZERS = [
    "optimizer_1",
    "optimizer_2",
    "optimizer_3",
    "optimizer_4",
    "optimizer_5",
    "optimizer_6",
    "EMG_Optimizer",
    "pDMP",
    "pDMP coupled",
    "pDMP omega",
    "EMG_IMU_optimizer",
    "EMG_IMU_optimizer_2"
]

SCRIPTS = [
    "EMG_MoCap",
    "IMU_MoCap"
]

ORDER = [
    "EMG to $q_d$",
    "IMU to $q_d$",
    "Integrator 1",
    "Integrator 2",
    "Integrator 3",
    "Integrator 4",
    "Integrator 5",
    "Integrator 6",
    "Integrator 7",
    "Integrator 8",
    "pDMP Weight update",
    "pDMP coupling term",
    "pDMP omega"
]

# Optimizer,Mean_Jerk,Median_Jerk,Sigma_Jerk,Max_Jerk,Q25_Jerk,Q75_Jerk,Lower_Median_Quantile,Upper_Median_Quantile,Lower_Mean_Quantile,Upper_Mean_Quantile

def print_best(metric, dataframe):
    idx = dataframe[metric].idxmin()
    row = dataframe.loc[idx]

    print(f"\nBest {metric}:")
    print(f"  Optimizer: {row['optimizer']}")
    print(f"  File:      {row['file']}")
    print(f"  Value:     {row[metric]:.6f}")

if __name__ == "__main__":
    # Group files
    jerk_stats_groups = {}
    stats_groups = {}
    jerk_groups = {}

    files = glob.glob(os.path.join(RESULTS_PATH, "*_jerk_stats_*.csv"))
    print(f"Found {len(files)} jerk stats files")

    for file in files:
        name = Path(file).stem
        parts = name.split("_")

        file_name = parts[-1]
        if file_name == "002" or file_name == "003":
            file_name = parts[-2] + "_" + parts[-1]

        if file_name not in jerk_stats_groups:
            jerk_stats_groups[file_name] = []
    
        jerk_stats_groups[file_name].append(file)
        
    files = glob.glob(os.path.join(RESULTS_PATH, "*_stats_*.csv"))
    # Remove the ones containing "jerk_stats"
    files = [f for f in files if "jerk_stats" not in f]
    print(f"Found {len(files)} files")

    for file in files:
        name = Path(file).stem
        parts = name.split("_")

        file_name = parts[-1]
        if file_name == "002" or file_name == "003":
            file_name = parts[-2] + "_" + parts[-1]

        if file_name not in stats_groups:
            stats_groups[file_name] = []
    
        stats_groups[file_name].append(file)

    files = glob.glob(os.path.join(RESULTS_PATH, "*_jerk_*.csv"))
    # Remove the ones containing "jerk_stats"
    files = [f for f in files if "jerk_stats" not in f]
    print(f"Found {len(files)} jerk files")

    for file in files:
        name = Path(file).stem
        parts = name.split("_")

        file_name = parts[-1]
        if file_name == "002" or file_name == "003":
            file_name = parts[-2] + "_" + parts[-1]

        if file_name not in jerk_groups:
            jerk_groups[file_name] = []
    
        jerk_groups[file_name].append(file)

    ######################### Jerk ###########################
    for file_name, files in jerk_stats_groups.items():
        print(f"\nProcessing jerk stats for {file_name} with {len(files)} files")

        filenames = []
        optimizer_name = []
        mean_jerk = []
        median_jerk = []
        sigma_jerk = []
        max_jerk = []
        q25_jerk = []
        q75_jerk = []
        lower_median_quantile = []
        upper_median_quantile = []
        lower_mean_quantile = []
        upper_mean_quantile = []

        for file in files:
            df = pd.read_csv(file, keep_default_na=False)
            filenames.append(os.path.basename(file))
            optimizer_name.append(df["Optimizer"][0])
            mean_jerk.append(df["Mean_Jerk"][0])
            median_jerk.append(df["Median_Jerk"][0])
            sigma_jerk.append(df["Sigma_Jerk"][0])
            max_jerk.append(df["Max_Jerk"][0])
            q25_jerk.append(df["Q25_Jerk"][0])
            q75_jerk.append(df["Q75_Jerk"][0])
            lower_median_quantile.append(df["Lower_Median_Quantile"][0])
            upper_median_quantile.append(df["Upper_Median_Quantile"][0])
            lower_mean_quantile.append(df["Lower_Mean_Quantile"][0])
            upper_mean_quantile.append(df["Upper_Mean_Quantile"][0])

        # Create optimizer labels
        labels = []
        for optimizer in optimizer_name:
            print("test", optimizer)
            # if optimizer contains "None"
            if "None" in optimizer:
                if "ExoTestReal1" in file_name or "001" in file_name:
                    labels.append("EMG to $q_d$")
                else:
                    labels.append("IMU to $q_d$")
            elif "EMG_IMU_optimizer_2" in optimizer:
                labels.append("Integrator 8")
            elif "EMG_IMU_optimizer" in optimizer:
                labels.append("Integrator 7")
            elif "optimizer_1" in optimizer:
                labels.append("Integrator 1")
            elif "optimizer_2" in optimizer:
                labels.append("Integrator 2")
            elif "optimizer_4" in optimizer:
                labels.append("Integrator 3")
            elif "optimizer_5" in optimizer:
                labels.append("Integrator 4")
            elif "optimizer_6" in optimizer:
                labels.append("Integrator 5")
            elif "EMG_Optimizer" in optimizer:
                labels.append("Integrator 6")
            elif "pDMP coupled" in optimizer:
                labels.append("pDMP coupling term")
            elif "pDMP omega" in optimizer:
                labels.append("pDMP omega")
            elif "pDMP" in optimizer:
                labels.append("pDMP Weight update")
            else:
                print("Ups ", optimizer)
                labels.append(optimizer)
            print(labels[-1])

        df_all = pd.DataFrame({
            "file": filenames,
            "optimizer": optimizer_name,
            "labels": labels,
            "mean": mean_jerk,
            "median": median_jerk,
            "sigma": sigma_jerk,
            "max": max_jerk,
            "q25": q25_jerk,
            "q75": q75_jerk,
            "lower_median_q": lower_median_quantile,
            "upper_median_q": upper_median_quantile,
            "lower_mean_q": lower_mean_quantile,
            "upper_mean_q": upper_mean_quantile
        })
        # Save to CSV for Veusz
        df_all.to_csv(os.path.join(VEUSZ_CSV_PATH, f"{file_name}_jerk_summary.csv"), index=False)

        df_all["labels"] = pd.Categorical(df_all["labels"], categories=ORDER, ordered=True)
        df_all = df_all.sort_values("labels")

        print(df_all[["labels", "optimizer"]])

        # statistics
        print_best("mean", df_all)
        print_best("median", df_all)
        print_best("sigma", df_all)
        print_best("max", df_all)

        # print(df_all["optimizer"])
        # print(df_all["optimizer"].apply(type))  
        # Generate bar plot for mean jerk
        plt.figure(figsize=(7, 4))
        plt.bar(df_all["labels"], df_all["mean"], yerr=[df_all["lower_median_q"], df_all["upper_median_q"]], color='skyblue')
        plt.scatter(df_all["labels"], df_all["max"], color='red', label='Max Jerk')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.xlabel("Optimizer")
        plt.ylabel("Mean Jerk (log scale)")
        # plt.title("Mean Jerk")
        plt.legend()
        plt.tight_layout()
        plt.show()

        df_bar_mean = df_all[["labels", "mean", "lower_median_q", "upper_mean_q"]].copy()

        # Generate bar plot for median jerk
        plt.figure(figsize=(7, 4))
        plt.bar(df_all["labels"], df_all["median"], yerr=[df_all["lower_median_q"], df_all["upper_median_q"]], color='lightgreen')
        plt.scatter(df_all["labels"], df_all["max"], color='red', label='Max Jerk')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.xlabel("Optimizer")
        plt.ylabel("Median Jerk (log scale)")
        # plt.title("Median Jerk for Different Optimizers")
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    for file_name, files in jerk_groups.items():
        print(f"\nProcessing stats for {file_name} with {len(files)} files")

        abs_jerk_data = []
        filenames = []
        optimizer_name = []

        for file in files:
            df = pd.read_csv(file)
            abs_jerk_data.append(df["abs_Jerk"].values)
            filename = os.path.basename(file)
            filenames.append(filename)

            # Remove extension
            name = filename.replace(".csv", "")

            # Split
            parts = name.split("_")

            # Extract optimizer (everything between "jerk" and last part)
            jerk_idx = parts.index("jerk")
            optimizer = "_".join(parts[jerk_idx + 1:-1])

            optimizer_name.append(optimizer)
        
        # Create optimizer labels
        labels = []
        for optimizer in optimizer_name:
            # if optimizer contains "None"
            if "None" in optimizer:
                if "ExoTestReal1" in file_name or "001" in file_name:
                    labels.append("EMG to $q_d$")
                else:
                    labels.append("IMU to $q_d$")
            elif "EMG_IMU_optimizer_2" in optimizer:
                labels.append("Integrator 8")
            elif "EMG_IMU_optimizer" in optimizer:
                labels.append("Integrator 7")
            elif "optimizer_1" in optimizer:
                labels.append("Integrator 1")
            elif "optimizer_2" in optimizer:
                labels.append("Integrator 2")
            elif "optimizer_4" in optimizer:
                labels.append("Integrator 3")
            elif "optimizer_5" in optimizer:
                labels.append("Integrator 4")
            elif "optimizer_6" in optimizer:
                labels.append("Integrator 5")
            elif "EMG_Optimizer" in optimizer:
                labels.append("Integrator 6")
            elif "pDMP coupled" in optimizer:
                labels.append("pDMP coupling term")
            elif "pDMP omega" in optimizer:
                labels.append("pDMP omega")
            elif "pDMP" in optimizer:
                labels.append("pDMP Weight update")
            else:
                print(optimizer)
                labels.append(optimizer)
            print(labels[-1])

        df_jerk = pd.DataFrame({
            "filenames": filenames,
            "optimizer": optimizer_name,
            "labels": labels,
            "jerk_data": abs_jerk_data
        })
        # Save to CSV for Veusz
        df_jerk.to_csv(os.path.join(VEUSZ_CSV_PATH, f"{file_name}_jerk_data.csv"), index=False)

        df_jerk["labels"] = pd.Categorical(df_jerk["labels"], categories=ORDER, ordered=True)
        df_jerk = df_jerk.sort_values("labels")

        print(df_jerk[["labels", "optimizer"]])

        plt.figure(figsize=(7, 4))
        plt.boxplot(df_jerk["jerk_data"], labels=df_jerk["labels"], showfliers=False)#, whis=[5, 95])
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.xlabel("Optimizer")
        plt.ylabel("Absolute Jerk (log scale)")
        # plt.title("Distribution of Absolute Jerk for Different Optimizers")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        violin = plt.violinplot(
            df_jerk["jerk_data"], 
            showmeans=False,
            showmedians=True,
            showextrema=True
        )
        # plt.yscale('log')
        plt.xticks(
            ticks=range(1, len(df_jerk["labels"]) + 1),
            labels=df_jerk["labels"],
            rotation=45
        )
        plt.xlabel("Optimizer")
        plt.ylabel("Absolute Jerk (log scale)")
        plt.tight_layout()
        plt.show()

    # # Generate box plot for jerk distribution
    # files = glob.glob(os.path.join(RESULTS_PATH, "*_jerk_*.csv"))
    # # Remove the ones containing "jerk_stats"
    # files = [f for f in files if "jerk_stats" not in f]
    # print(f"Found {len(files)} files")

    

    for file_name, files in stats_groups.items():
        print(f"\nProcessing stats for {file_name} with {len(files)} files")

        filenames = []
        optimizer_name = []
        mae_array = []
        rmse_array = []
        bias_array = []
        correlation_array = []
        r_squared_array = []
        lag_array = []
        lag_time_sec_array = []
        mean_onset_lag_sec_array = []
        median_onset_lag_sec_array = []
        std_onset_lag_sec_array = []
        median_peak_lag_sec_array = []
        mean_peak_lag_sec_array = []
        rom_error_array = []
        shifted_mae_array = []
        shifted_rmse_array = []

        for file in files:
            df = pd.read_csv(file, keep_default_na=False)
            filenames.append(os.path.basename(file))
            optimizer = df["Optimizer"][0]
            mae = df["MAE"][0]
            rmse = df["RMSE"][0]
            bias = df["Bias"][0]
            correlation = df["Correlation"][0]
            r_squared = df["R_squared"][0]
            lag = df["Lag"][0]
            try:
                lag_time_sec = df["Lag_time_sec"][0]
            except:
                lag_time_sec = df["Lag_seconds"][0]
            mean_onset_lag = df["Mean_Onset_Lag_sec"][0]
            median_onset_lag = df["Median_Onset_Lag_sec"][0]
            std_onset_lag = df["Std_Onset_Lag_sec"][0]
            median_peak_lag = df["Median_Peak_Lag_sec"][0]
            mean_peak_lag = df["Mean_Peak_Lag_sec"][0]
            try:
                rom_error = df["ROM_error"][0]
            except:
                rom_error = df["ROM_Error"][0]
            shifted_mae = df["Shifted_MAE"][0]
            shifted_rmse = df["Shifted_RMSE"][0]

            optimizer_name.append(optimizer)
            mae_array.append(mae)
            rmse_array.append(rmse)
            bias_array.append(bias)
            correlation_array.append(correlation)
            r_squared_array.append(r_squared)
            lag_array.append(lag)
            lag_time_sec_array.append(lag_time_sec)
            mean_onset_lag_sec_array.append(mean_onset_lag)
            median_onset_lag_sec_array.append(median_onset_lag)
            std_onset_lag_sec_array.append(std_onset_lag)
            median_peak_lag_sec_array.append(median_peak_lag)
            mean_peak_lag_sec_array.append(mean_peak_lag)
            rom_error_array.append(rom_error)
            shifted_mae_array.append(shifted_mae)
            shifted_rmse_array.append(shifted_rmse)
        
        # Create optimizer labels
        labels = []
        for optimizer in optimizer_name:
            # if optimizer contains "None"
            if "None" in optimizer:
                if "ExoTestReal1" in file_name or "001" in file_name:
                    labels.append("EMG to $q_d$")
                else:
                    labels.append("IMU to $q_d$")
            elif "EMG_IMU_optimizer_2" in optimizer:
                labels.append("Integrator 8")
            elif "EMG_IMU_optimizer" in optimizer:
                labels.append("Integrator 7")
            elif "optimizer_1" in optimizer:
                labels.append("Integrator 1")
            elif "optimizer_2" in optimizer:
                labels.append("Integrator 2")
            elif "optimizer_4" in optimizer:
                labels.append("Integrator 3")
            elif "optimizer_5" in optimizer:
                labels.append("Integrator 4")
            elif "optimizer_6" in optimizer:
                labels.append("Integrator 5")
            elif "EMG_Optimizer" in optimizer:
                labels.append("Integrator 6")
            elif "pDMP coupled" in optimizer:
                labels.append("pDMP coupling term")
            elif "pDMP omega" in optimizer:
                labels.append("pDMP omega")
            elif "pDMP" in optimizer:
                labels.append("pDMP Weight update")
            else:
                print(optimizer)
                labels.append(optimizer)
            print(labels[-1])

        df_stats = pd.DataFrame({
            "file": filenames,
            "optimizer": optimizer_name,
            "labels": labels,
            "MAE": mae_array,
            "RMSE": rmse_array,
            "Bias": bias_array,
            "Correlation": correlation_array,
            "R_squared": r_squared_array,
            "Lag": lag_array,
            "Lag_time_sec": lag_time_sec_array,
            "Mean_Onset_Lag_sec": mean_onset_lag_sec_array,
            "Median_Onset_Lag_sec": median_onset_lag_sec_array,
            "Std_Onset_Lag_sec": std_onset_lag_sec_array,
            "Median_Peak_Lag_sec": median_peak_lag_sec_array,
            "Mean_Peak_Lag_sec": mean_peak_lag_sec_array,
            "ROM_error": rom_error_array,
            "Shifted_MAE": shifted_mae_array,
            "Shifted_RMSE": shifted_rmse_array
        })

        df_stats["labels"] = pd.Categorical(df_stats["labels"], categories=ORDER, ordered=True)
        df_stats = df_stats.sort_values("labels")

        ####################################
        # Find corresponding jerk stats for this file_name
        jerk_files = jerk_stats_groups[file_name]

        jerk_opt = []
        jerk_median = []

        for jf in jerk_files:
            df_j = pd.read_csv(jf, keep_default_na=False)
            jerk_opt.append(df_j["Optimizer"][0])
            jerk_median.append(df_j["Median_Jerk"][0])

        df_jerk_stats = pd.DataFrame({
            "optimizer": jerk_opt,
            "median_jerk": jerk_median
        })

        # Merge into df_stats
        df_stats = df_stats.merge(df_jerk_stats, on="optimizer", how="left")
        
        emg_integrators = [
            "EMG to $q_d$",
            "Integrator 1",
            "Integrator 2",
            "Integrator 3",
            "Integrator 4",
            "Integrator 5",
            "Integrator 6",
            "pDMP Weight update",
            "pDMP coupling term"
        ]

        imu_integrators = [
            "IMU to $q_d$",
            "Integrator 7",
            "Integrator 8"
        ]

        is_emgtest6s_001 = (
            "EMGTest6s_001" in file_name
            or df_stats["file"].astype(str).str.contains("EMGTest6s_001", regex=False).any()
        )

        is_exoimutest1_003 = (
            "ExoIMUTest1_003" in file_name
            or df_stats["file"].astype(str).str.contains("ExoIMUTest1_003", regex=False).any()
        )

        if is_emgtest6s_001:
            selected_labels = emg_integrators

        elif is_exoimutest1_003:
            selected_labels = imu_integrators

        else:
            selected_labels = []

        if selected_labels:
            df_selected = df_stats[
                df_stats["labels"].astype(str).isin(selected_labels)
            ].copy()

            df_selected = df_selected[[
                "labels",
                "MAE",
                "RMSE",
                "Correlation",
                "R_squared",
                "ROM_error",
                "median_jerk",
                "Median_Onset_Lag_sec",
                "Median_Peak_Lag_sec"
            ]]

            df_selected = df_selected.rename(columns={
                "labels": "Integrator"
            })

            df_radar = pd.concat(
                [df_radar, df_selected],
                ignore_index=True
            )

        ####################################

        # print optimizer with lowest MAE, RMSE, shifted MAE, shifted RMSE
        print_best("MAE", df_stats)
        print_best("RMSE", df_stats)
        print_best("Shifted_MAE", df_stats)
        print_best("Shifted_RMSE", df_stats)
        print_best("ROM_error", df_stats)

        # print optimizer with highest correlation and R_squared
        idx = df_stats["Correlation"].idxmax()
        row = df_stats.loc[idx]
        print(f"\nBest Correlation:")
        print(f"  Optimizer: {row['optimizer']}")
        print(f"  File:      {row['file']}")
        print(f"  Value:     {row['Correlation']:.6f}")

        idx = df_stats["R_squared"].idxmax()
        row = df_stats.loc[idx]
        print(f"\nBest R_squared:")
        print(f"  Optimizer: {row['optimizer']}")
        print(f"  File:      {row['file']}")
        print(f"  Value:     {row['R_squared']:.6f}")

        # ---------- Radar plot (ALL metrics) ----------

        # metrics = [
        #     "MAE","RMSE","Bias","Correlation","R_squared",
        #     "Lag","Lag_time_sec","ROM_error","Shifted_MAE","Shifted_RMSE", "median_jerk"
        # ]
        # metrics = [
        #     "MAE","RMSE","Bias","Correlation","R_squared",
        #     "ROM_error","Shifted_MAE","Shifted_RMSE", "median_jerk"
        # ]
        metrics = [
            "MAE","RMSE","Correlation","R_squared",
            "ROM_error", "median_jerk", "Median_Onset_Lag_sec", "Median_Peak_Lag_sec"
        ]
        metrics_labels = [
            "MAE","RMSE","Pearson correlation","$R^2$",
            "ROM error", "Median jerk", "Median onset lag (s)", "Median peak lag (s)"
        ]

        df_norm = df_stats.copy()

        # Exclude the pDMP optimizers
        df_norm = df_norm[~df_norm["optimizer"].str.contains("pDMP")]

        # Handle special cases first
        df_norm["Bias"] = df_norm["Bias"].abs()
        # df_norm["Lag"] = df_norm["Lag"].abs()
        # df_norm["Lag_time_sec"] = df_norm["Lag_time_sec"].abs()
        df_norm["ROM_error"] = df_norm["ROM_error"].abs()

        # vals = df_stats["Lag"].astype(float)
        # max_abs = np.max(np.abs(vals)) + 1e-8
        # # Normalize to [-1, 1]
        # norm = vals / max_abs
        # # Shift to [0, 1] (so radar works)
        # df_norm["Lag"] = (norm + 1) / 2

        # vals = df_stats["Lag_time_sec"].astype(float)
        # max_abs = np.max(np.abs(vals)) + 1e-8
        # norm = vals / max_abs
        # df_norm["Lag_time_sec"] = (norm + 1) / 2

        # Normalize all metrics to [0,1]
        for col in metrics:
            vals = df_norm[col].astype(float)
            min_v = vals.min()
            max_v = vals.max()
            
            # avoid divide by zero
            norm = (vals - min_v) / (max_v - min_v + 1e-8)
            
            # invert "error" metrics
            if col not in ["Correlation", "R_squared"]:
                norm = 1 - norm
            
            df_norm[col] = norm

        # Radar setup
        labels = metrics_labels
        num_vars = len(labels)

        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        plt.figure(figsize=(7,7))
        ax = plt.subplot(111, polar=True)

        # Plot each optimizer
        for i, row in df_norm.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, label=row["labels"])
            ax.fill(angles, values, alpha=0.05)

        # Axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)

        ax.set_ylim(0, 1)
        # ax.set_title(f"Radar Comparison – {file_name}", pad=20)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()

        # score the optimizers
        weights = {
            "MAE": 1.0,
            "RMSE": 1.0,
            # "Bias": 1.0,
            "Correlation": 1.5,
            "R_squared": 1.2,
            "ROM_error": 1.2,
            # "Shifted_MAE": 1.0,
            # "Shifted_RMSE": 1.0,
            "median_jerk": 1.5,
            "Median_Onset_Lag_sec": 0.0,
            "Median_Peak_Lag_sec": 0.0
        }

        df_norm["score"] = sum(df_norm[m] * weights[m] for m in metrics) / sum(weights.values())
        df_norm = df_norm.sort_values("score", ascending=False)
        for i, row in df_norm.iterrows():
            print(f"{row['optimizer']:<20} | Score: {row['score']:.3f}")

    # Sort radar dataframe in the desired final order
    radar_order = [
        "EMG to $q_d$",
        "IMU to $q_d$",
        "Integrator 1",
        "Integrator 2",
        "Integrator 3",
        "Integrator 4",
        "Integrator 5",
        "Integrator 6",
        "Integrator 7",
        "Integrator 8",
        "pDMP Weight update",
        "pDMP coupling term",
    ]

    df_radar["Integrator"] = pd.Categorical(
        df_radar["Integrator"],
        categories=radar_order,
        ordered=True
    )

    df_radar = df_radar.sort_values("Integrator")

    print("\nFinal radar dataframe:")
    print(df_radar)

    # df_radar.to_csv(
    #     os.path.join(VEUSZ_CSV_PATH, "combined_radar_dataframe.csv"),
    #     index=False
    # )

    df_norm = df_radar.copy()

    # Exclude the pDMP optimizers #TODO
    df_norm = df_norm[~df_norm["Integrator"].str.contains("pDMP")]
    # df_norm = df_norm[~df_norm["Integrator"].str.contains("pDMP omega")]

    # Handle special cases first
    # df_norm["Bias"] = df_norm["Bias"].abs()
    # df_norm["Lag"] = df_norm["Lag"].abs()
    # df_norm["Lag_time_sec"] = df_norm["Lag_time_sec"].abs()
    df_norm["ROM_error"] = df_norm["ROM_error"].abs()

    # vals = df_stats["Lag"].astype(float)
    # max_abs = np.max(np.abs(vals)) + 1e-8
    # # Normalize to [-1, 1]
    # norm = vals / max_abs
    # # Shift to [0, 1] (so radar works)
    # df_norm["Lag"] = (norm + 1) / 2

    # vals = df_stats["Lag_time_sec"].astype(float)
    # max_abs = np.max(np.abs(vals)) + 1e-8
    # norm = vals / max_abs
    # df_norm["Lag_time_sec"] = (norm + 1) / 2

    # Normalize all metrics to [0,1]
    for col in metrics:
        vals = df_norm[col].astype(float)
        min_v = vals.min()
        max_v = vals.max()
            
        # avoid divide by zero
        norm = (vals - min_v) / (max_v - min_v + 1e-8)
        
        # invert "error" metrics
        if col not in ["Correlation", "R_squared"]:
            norm = 1 - norm
            
        df_norm[col] = norm

    # Radar setup
    labels = metrics_labels
    num_vars = len(labels)

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))

    # Plot each optimizer
    cmap = mpl.colormaps["tab20"].resampled(len(df_norm))

    for j, (_, row) in enumerate(df_norm.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]

        color = cmap(j)

        ax.plot(
            angles,
            values,
            label=row["Integrator"],   # or row["labels"] if using df_stats
            color=color,
            linewidth=1.8
        )
        ax.fill(
            angles,
            values,
            color=color,
            alpha=0.05
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)

    # Move plot left and reserve space for legend
    fig.subplots_adjust(right=0.68)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        frameon=True
    )

    plt.show()
    
    # # Load stats
    # # Optimizer,MAE,RMSE,Bias,Correlation,R_squared,Lag,Lag_time_sec,ROM_error,Shifted_MAE,Shifted_RMSE
    # files = glob.glob(os.path.join(RESULTS_PATH, "*_stats_*.csv"))
    # files = [f for f in files if "jerk_stats" not in f]
    # print(f"Found {len(files)} files")

    



        


    
        