
import pandas as pd
from pathlib import Path

# 파일 경로 설정
csv_path_1 = Path("D:/W00Y0NG/PRGM2/2025BCI/current_experiments/DATA/raw/experiment_001/(1).csv")
csv_path_2 = Path("D:/W00Y0NG/PRGM2/2025BCI/current_experiments/DATA/raw/experiment_001/(2).csv")
epoch_path_1 = r"current_experiments\DATA\video\experiment_001_epochs.xlsx"
epoch_path_2 = r"current_experiments\DATA\video\experiment_001_epochs.xlsx"

combined_csv_path = Path("D:/W00Y0NG/PRGM2/2025BCI/current_experiments/DATA/raw/experiment_001/combined.csv")
combined_epoch_path = Path("D:/W00Y0NG/PRGM2/2025BCI/current_experiments/DATA/video/experiment_001_epochs_combined.xlsx")

# CSV 불러오기 (탭 구분자)
df1 = pd.read_csv(csv_path_1, sep='\t', header=None, engine='python')
df2 = pd.read_csv(csv_path_2, sep='\t', header=None, engine='python')

# 엑셀 불러오기
epoch1 = pd.read_excel(epoch_path_1)
epoch2 = pd.read_excel(epoch_path_2)

# df2 시간 오프셋 조정
last_sample_time = df1.iloc[-1, 0]
df2.iloc[:, 0] += last_sample_time
df_combined = pd.concat([df1, df2], ignore_index=True)

# epoch2 시간 오프셋 조정
last_epoch_time = epoch1.iloc[-1, 1]
epoch2_offset = epoch2.copy()
epoch2_offset.iloc[:, 0] += last_epoch_time
epoch2_offset.iloc[:, 1] += last_epoch_time
epoch_combined = pd.concat([epoch1, epoch2_offset], ignore_index=True)

# 저장
df_combined.to_csv(combined_csv_path, sep='\t', header=False, index=False)
epoch_combined.to_excel(combined_epoch_path, index=False)

print("CSV와 xlsx 병합 완료")