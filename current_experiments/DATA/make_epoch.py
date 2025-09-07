import pandas as pd
import numpy as np

# ==== 1. 경로 설정 및 데이터 로드 ====
csv_path = 'current_experiments\DATA\Final\eeg_sign_session1-3.csv'
df = pd.read_csv(csv_path, sep='\t')

# ==== 2. EEG 데이터와 마커 분리 ====
eeg_data = df.iloc[:, 1:17].values       # 1~16채널 EEG 데이터
marker_values = df.iloc[:, -1].values    # 마지막 열은 마커
marker_values = marker_values.astype(int)

# ==== 3. 마커 기준 정보 ====
sfreq = 125               # 샘플링 주파수 (Hz)
epoch_sec = 3             # epoch 길이 (초)
epoch_samples = sfreq * epoch_sec  # 375샘플

# ==== 4. 100번대 마커 위치 찾기 ====
imagine_indices = np.where((marker_values >= 100) & (marker_values < 200))[0]
imagine_labels = marker_values[imagine_indices]

# ==== 5. 유효한 epoch만 추출 ====
valid_indices = []
valid_labels = []

for i, idx in enumerate(imagine_indices):
    if idx + epoch_samples <= eeg_data.shape[0]:
        valid_indices.append(idx)
        valid_labels.append(imagine_labels[i])

# ==== 6. epoch 추출 ====
epochs = np.array([
    eeg_data[start_idx:start_idx + epoch_samples, :]
    for start_idx in valid_indices
])

labels = np.array(valid_labels)

print(f"추출된 epoch 수: {len(epochs)}")    # 20
print(f"epoch shape: {epochs.shape}")      # (20, 375, 16)

print(labels)

# ==== 저장 ====
np.save('current_experiments\DATA\Final\eeg_sign_epochs.npy', epochs)
np.save('current_experiments\DATA\Final\eeg_sign_labels.npy', labels)
