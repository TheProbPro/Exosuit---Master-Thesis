# import numpy as np
# import pandas as pd
# from pathlib import Path

# if __name__ == "__main__":
#     # Open file
#     file_path = Path("Outputs/EMG_Test/VictorBNielsen/EMG_Control_Loop_Data_VictorBNielsen.csv")
#     df = pd.read_csv(file_path)

#     # Create a time vector based on the length of the data and the recording time
#     recording_time = 10.0
#     num_samples = len(df)
#     time_vector = np.linspace(0, recording_time, num_samples, endpoint=False)

#     # calculate and print effective sample rate
#     fs_effective = num_samples / recording_time
#     print(f"Effective sample rate: {fs_effective:.2f} Hz")

#     # Add time vector to DataFrame
#     df.insert(0, 'Time', time_vector)
#     # Save updated DataFrame to a new CSV file
#     output_path = file_path
#     df.to_csv(output_path, index=False)

import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # ---- file paths ----
    file_path = Path("Outputs/EMG_Test/VictorBNielsen/EMG_Data_VictorBNielsen.csv")
    # file_path = Path("Outputs/EMG_Test/VictorBNielsen/EMG_Control_Loop_Data_VictorBNielsen.csv")

    # ---- load CSV as strings (important!) ----
    df = pd.read_csv(file_path, dtype=str)

    # ---- remove [ ] brackets from all cells ----
    df = df.applymap(
        lambda x: x.replace("[", "").replace("]", "") if isinstance(x, str) else x
    )

    # ---- convert columns to numeric where possible ----
    df = df.apply(pd.to_numeric, errors="ignore")

    # ---- create time vector ----
    recording_time = 10.0  # seconds
    num_samples = len(df)
    time_vector = np.linspace(0, recording_time, num_samples, endpoint=False)

    # ---- calculate and print effective sample rate ----
    fs_effective = num_samples / recording_time
    print(f"Effective sample rate: {fs_effective:.2f} Hz")

    # ---- insert time column ----
    if not "Time" in df.columns:
        df.insert(0, "Time", time_vector)

    # ---- save cleaned CSV (overwrite original) ----
    df.to_csv(file_path, index=False)

    print("CSV cleaned, time vector added, and file saved.")
