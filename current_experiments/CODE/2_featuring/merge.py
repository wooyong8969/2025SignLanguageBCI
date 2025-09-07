import numpy as np
import os

# 파일 경로 정의
base_dir = "current_experiments/DATA/processed"

# 실험 002 데이터
data_002 = np.load(os.path.join(base_dir, "experiment_002", "experiment_002(4)_cleaned.npy"))
labels_002 = np.load(os.path.join(base_dir, "experiment_002", "experiment_002(4)_labels.npy"))

# 실험 003 데이터
data_003 = np.load(os.path.join(base_dir, "experiment_003", "experiment_003_cleaned.npy"))
labels_003 = np.load(os.path.join(base_dir, "experiment_003", "experiment_003_labels.npy"))

# 데이터 이어붙이기 (axis=0 기준으로 sample을 기준으로 결합)
combined_data = np.concatenate([data_002, data_003], axis=0)
combined_labels = np.concatenate([labels_002, labels_003], axis=0)

# 저장 경로 지정
save_data_path = os.path.join(base_dir, "experiment_003", "experiment_003(1-5)_cleaned.npy")
save_labels_path = os.path.join(base_dir, "experiment_003", "experiment_003(1-5)_labels.npy")

# 파일 저장
np.save(save_data_path, combined_data)
np.save(save_labels_path, combined_labels)

print(f"Combined data shape: {combined_data.shape}")
print(f"Combined labels shape: {combined_labels.shape}")
print("저장 완료!")
