import os
import numpy as np
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from eeg_dataset_maker import EEGDataset
from eeg_augmenter import EEGAugmenter
from feature_extractor2 import DWTFeatureExtractor

# ---------- 1. 경로 설정 ---------- #
mat_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(2)_cleaned.mat'
label_csv_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(2)_labels.csv'
feature_save_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(2)_TFS_features.npy'
label_save_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(2)_TFS_labels.npy'
label_encoder_path = 'label_encoder.joblib'

# ---------- 2. 데이터 불러오기 ---------- #
dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()
eeg, labels, fs = dataset.get_data()

le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
dump(le, label_encoder_path)

# ---------- 3. 데이터 증강 ---------- #
augmenter = EEGAugmenter(noise_level=0.01, max_shift=8)
aug_eeg, aug_labels = augmenter.augment(eeg, encoded_labels, num_augments=3)

print(f"증강된 EEG shape: {aug_eeg.shape}, 라벨 수: {len(aug_labels)}")

# ---------- 4. 특징 추출 ---------- #
extractor = DWTFeatureExtractor(fs=fs, wavelet='coif5', level=3)
time_feats, freq_feats = extractor.extract(aug_eeg)
csp_feats = extractor.extract_csp_features(aug_eeg, aug_labels, n_components=4)
riem_feats = extractor.extract_riemannian_logvar(aug_eeg)
features = np.concatenate([time_feats, freq_feats, csp_feats, riem_feats], axis=1)

# ---------- 5. 저장 ---------- #
np.save(feature_save_path, features)
np.save(label_save_path, aug_labels)
print("특징과 라벨 저장 완료")
