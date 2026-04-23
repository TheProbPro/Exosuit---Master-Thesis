import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

RESULTS_PATH = "Outputs/Results/"
VEUSZ_CSV_PATH = "Outputs/Results/Veusz_csv/"
GRAPH_SAVE_PATH = "Outputs/Results/Graphs/"

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

        df_all = pd.DataFrame({
            "file": filenames,
            "optimizer": optimizer_name,
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


        # statistics
        print_best("mean", df_all)
        print_best("median", df_all)
        print_best("sigma", df_all)
        print_best("max", df_all)

        # print(df_all["optimizer"])
        # print(df_all["optimizer"].apply(type))  

        # Generate bar plot for mean jerk
        plt.figure(figsize=(12, 6))
        plt.bar(df_all["optimizer"], df_all["mean"], yerr=[df_all["lower_median_q"], df_all["upper_median_q"]], color='skyblue')
        plt.scatter(df_all["optimizer"], df_all["max"], color='red', label='Max Jerk')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.xlabel("Optimizer")
        plt.ylabel("Mean Jerk (log scale)")
        plt.title("Mean Jerk for Different Optimizers")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Generate bar plot for median jerk
        plt.figure(figsize=(12, 6))
        plt.bar(df_all["optimizer"], df_all["median"], yerr=[df_all["lower_median_q"], df_all["upper_median_q"]], color='lightgreen')
        plt.scatter(df_all["optimizer"], df_all["max"], color='red', label='Max Jerk')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.xlabel("Optimizer")
        plt.ylabel("Median Jerk (log scale)")
        plt.title("Median Jerk for Different Optimizers")
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
        
        df_jerk = pd.DataFrame({
            "filenames": filenames,
            "optimizer": optimizer_name,
            "jerk_data": abs_jerk_data
        })

        plt.figure(figsize=(12, 6))
        plt.boxplot(df_jerk["jerk_data"], labels=df_jerk["optimizer"])
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.xlabel("Optimizer")
        plt.ylabel("Absolute Jerk (log scale)")
        plt.title("Distribution of Absolute Jerk for Different Optimizers")
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
            rom_error_array.append(rom_error)
            shifted_mae_array.append(shifted_mae)
            shifted_rmse_array.append(shifted_rmse)

        df_stats = pd.DataFrame({
            "file": filenames,
            "optimizer": optimizer_name,
            "MAE": mae_array,
            "RMSE": rmse_array,
            "Bias": bias_array,
            "Correlation": correlation_array,
            "R_squared": r_squared_array,
            "Lag": lag_array,
            "Lag_time_sec": lag_time_sec_array,
            "ROM_error": rom_error_array,
            "Shifted_MAE": shifted_mae_array,
            "Shifted_RMSE": shifted_rmse_array
        })

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

        metrics = [
            "MAE","RMSE","Bias","Correlation","R_squared",
            "Lag","Lag_time_sec","ROM_error","Shifted_MAE","Shifted_RMSE", "median_jerk"
        ]

        df_norm = df_stats.copy()

        # Handle special cases first
        df_norm["Bias"] = df_norm["Bias"].abs()
        # df_norm["Lag"] = df_norm["Lag"].abs()
        df_norm["Lag_time_sec"] = df_norm["Lag_time_sec"].abs()
        df_norm["ROM_error"] = df_norm["ROM_error"].abs()

        vals = df_stats["Lag"].astype(float)
        max_abs = np.max(np.abs(vals)) + 1e-8
        # Normalize to [-1, 1]
        norm = vals / max_abs
        # Shift to [0, 1] (so radar works)
        df_norm["Lag"] = (norm + 1) / 2

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
        labels = metrics
        num_vars = len(labels)

        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        plt.figure(figsize=(8,8))
        ax = plt.subplot(111, polar=True)

        # Plot each optimizer
        for i, row in df_norm.iterrows():
            values = [row[m] for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, label=row["optimizer"])
            ax.fill(angles, values, alpha=0.05)

        # Axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)

        ax.set_ylim(0, 1)
        ax.set_title(f"Radar Comparison – {file_name}", pad=20)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.show()
    
    # # Load stats
    # # Optimizer,MAE,RMSE,Bias,Correlation,R_squared,Lag,Lag_time_sec,ROM_error,Shifted_MAE,Shifted_RMSE
    # files = glob.glob(os.path.join(RESULTS_PATH, "*_stats_*.csv"))
    # files = [f for f in files if "jerk_stats" not in f]
    # print(f"Found {len(files)} files")

    



        


    
        