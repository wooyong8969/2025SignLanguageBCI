from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import os

mat_path = r'current_experiments\DATA\processed\experiment_001_cleaned.mat'
label_csv_path = r'current_experiments\DATA\processed\experiment_001_labels.csv'

# ---------- 1.  전처리 된 데이터셋 불러오기 ---------- #
dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)  # y가 ['left', 'right', 'left', ...] 등이라면 0, 1로 변환됨


# ---------- 2. 특징 추출하기 ---------- #
print("특징 추출 중...")
extractor = DWTFeatureExtractor(wavelet='coif1', level=5)
time_features, freq_features = extractor.extract(eeg)

flat_time = extractor.flatten_feature_dict(time_features, extractor.bands)
flat_freq = extractor.flatten_feature_dict(freq_features, extractor.bands)
csp_features = extractor.extract_csp_features(eeg, labels, n_components=4)
riemannian_features = extractor.extract_riemannian_features(eeg)

features = np.concatenate([flat_time, flat_freq, csp_features, riemannian_features], axis=1)
n_epochs = eeg.shape[0]
features = features.reshape(n_epochs, -1)

np.save(r'current_experiments\DATA\processed\experiment_001_processed_features.npy', features)
np.save(r'current_experiments\DATA\processed\experiment_001_encoded_labels.npy', encoded_labels)