import numpy as np
import pandas as pd
from pathlib import Path

def clean_numeric_column(col):
    return (
        col.astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .astype(float)
    )

if __name__ == "__main__":
    #file
    # file = Path("Outputs/RWExosuitResults/Processed/VictorBNielsen/1/trial_1.csv")
    file = Path("Outputs/RecordedEMG/EMGData.csv")

    df = pd.read_csv(file)

    # Clean numeric columns
    for col in df.columns:
        if col not in ["timestamp"]:
            df[col] = clean_numeric_column(df[col])
    # Save cleaned data
    df.to_csv(file.with_name(file.stem + "_cleaned.csv"), index=False)