import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ---------- 1. 경로 설정 ---------- #
data_path = "current_experiments/DATA/processed/experiment_001"
mat_path = os.path.join(data_path, "experiment_001(1-4)_cleaned.mat")
label_path = os.path.join(data_path, "experiment_001(1-4)_labels.csv")

# ---------- 2. 데이터 불러오기 (구조체 유지) ---------- #
mat = sio.loadmat(mat_path)
eeg_struct = mat["EEG_clean"]
eeg_data = eeg_struct["data"][0, 0]       # (epochs, samples, channels)
srate = eeg_struct["srate"][0, 0]         # 샘플링 주파수

labels = pd.read_csv(label_path, header=None).iloc[:, 0].values

# ---------- 3. 셔플 및 8:2 분할 ---------- #
# split 위해 sklearn은 (epochs, features...) 구조 필요 → 전치 필요
eeg_data_t = np.transpose(eeg_data, (0, 2, 1))  # (epochs, channels, samples)

X_train, X_test, y_train, y_test = train_test_split(
    eeg_data_t, labels, test_size=0.2, stratify=labels, random_state=42
)

# 전치 복원 → (epochs, samples, channels)
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# ---------- 4. 저장 ---------- #
sio.savemat(os.path.join(data_path, "experiment_001(1-4)_train_cleaned.mat"), {
    "EEG_clean": {
        "data": X_train.astype(np.float32),
        "srate": srate
    }
})
sio.savemat(os.path.join(data_path, "experiment_001(1-4)_test_cleaned.mat"), {
    "EEG_clean": {
        "data": X_test.astype(np.float32),
        "srate": srate
    }
})

pd.DataFrame(y_train).to_csv(
    os.path.join(data_path, "experiment_001(1-4)_train_labels.csv"), index=False
)
pd.DataFrame(y_test).to_csv(
    os.path.join(data_path, "experiment_001(1-4)_test_labels.csv"), index=False
)

print("구조체 유지한 EEG + 라벨 8:2 분할 저장 완료")
