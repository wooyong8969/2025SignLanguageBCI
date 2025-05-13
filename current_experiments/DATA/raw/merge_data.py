import pandas as pd
import os

csv_files = [
    r"current_experiments\DATA\raw\experiment_001\SI_30(1).csv",
    r"current_experiments\DATA\raw\experiment_001\SI_30(3).csv",
    r"current_experiments\DATA\raw\experiment_001\SI_30(4).csv",
]

raw_data_combined = pd.concat(
    [pd.read_csv(f, sep='\t', header=None, engine='python') for f in csv_files],
    ignore_index=True
)

print(len(raw_data_combined))

raw_data_combined.to_csv(r"current_experiments\DATA\raw\experiment_001\SI_30(1-4).csv", sep='\t', header=False, index=False)
print("CSV 병합 완료.")


excel_files = [
    r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
    r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
    r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
]

epoch_combined = pd.concat(
    [pd.read_excel(f) for f in excel_files],
    ignore_index=True
)

epoch_combined.to_excel(r"current_experiments\DATA\video\experiment_001_90_epochs.xlsx", index=False)
print("Excel 병합 완료.")