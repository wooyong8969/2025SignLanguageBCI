import pandas as pd
import os

csv_files = [
    r"current_experiments\DATA\Final\eeg_sign_session1.csv",
    r"current_experiments\DATA\Final\eeg_sign_session2.csv",
    r"current_experiments\DATA\Final\eeg_sign_session3.csv",
]

raw_data_combined = pd.concat(
    [pd.read_csv(f, sep='\t', header=None, engine='python') for f in csv_files],
    ignore_index=True
)

print(len(raw_data_combined))

raw_data_combined.to_csv(r"current_experiments\DATA\Final\eeg_sign_session1-3.csv", sep='\t', header=False, index=False)
print("CSV 병합 완료.")


# excel_files = [
#     # r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
#     # r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
#     # r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
#     # r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
#     # r"current_experiments\DATA\video\experiment_001_30_epochs.xlsx",
#     r"current_experiments\DATA\video\experiment_002_10_epochs.xlsx",
#     r"current_experiments\DATA\video\experiment_002_10_epochs.xlsx",
#     r"current_experiments\DATA\video\experiment_002_10_epochs.xlsx",
#     r"current_experiments\DATA\video\experiment_002_10_epochs.xlsx",
# ]

# epoch_combined = pd.concat(
#     [pd.read_excel(f) for f in excel_files],
#     ignore_index=True
# )

# epoch_combined.to_excel(r"current_experiments\DATA\video\experiment_003_40_epochs.xlsx", index=False)
# print("Excel 병합 완료.")