from eeg_dataset_maker import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from eeg_augmenter import EEGAugmenter
import os
import scipy.io as sio
from joblib import dump, load

mat_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(7)_cleaned.mat'
label_csv_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(7)_labels.csv'

features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(7)_features.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(7)_labels.npy'


# ---------- 1. 전처리 된 데이터셋 불러오기 ---------- #

dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()
#le = LabelEncoder()
le = load(r'current_experiments\MODEL\label_encoder.joblib')
encoded_labels = le.fit_transform(labels)


# ---------- 2. 데이터 증강하기 ---------- #

augmenter = EEGAugmenter(noise_level=0.01, max_shift=8)
aug_eeg, aug_labels = augmenter.augment(eeg, encoded_labels, num_augments=1)

print("증강 후 데이터 shape:", aug_eeg.shape)
print("증강 후 라벨 수:", len(aug_labels))


# ---------- 2. 특징 추출하기 ---------- #

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("이미 파일이 저장되어 있습니다.")
else:
    print("특징 추출 중...")
    extractor = DWTFeatureExtractor(wavelet='coif1', level=5)
    time_features, freq_features = extractor.extract(aug_eeg)

    flat_time = extractor.flatten_feature_dict(time_features, extractor.bands)
    flat_freq = extractor.flatten_feature_dict(freq_features, extractor.bands)
    csp_features = extractor.extract_csp_features(aug_eeg, aug_labels, n_components=4)
    riemannian_features = extractor.extract_riemannian_features(aug_eeg)

    features = np.concatenate([flat_time, flat_freq, csp_features, riemannian_features], axis=1)
    n_epochs = aug_eeg.shape[0]
    features = features.reshape(n_epochs, -1)

    np.save(features_path, features)
    np.save(labels_path, aug_labels)

    print(features.shape)

# dump(le, 'label_encoder.joblib')


