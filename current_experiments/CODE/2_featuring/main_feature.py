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

# mat_path = r'current_experiments\DATA\open\BCI competition IV\experiment_iv_cleaned.mat'
# label_csv_path = r'current_experiments\DATA\open\BCI competition IV\experiment_iv_labels.csv'

# features_path = r'current_experiments\DATA\open\BCI competition IV\experiment_iv_cleaned.npy'
# labels_path = r'current_experiments\DATA\open\BCI competition IV\experiment_iv_labels.npy'

# mat_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(_)_cleaned.mat'
# label_csv_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(_)_labels.csv'

# features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(_)_cleaned.npy'
# labels_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(_)_labels.npy'

# mat_path = r'current_experiments\DATA\processed\experiment_003\experiment_003_cleaned.mat'
# label_csv_path = r'current_experiments\DATA\processed\experiment_003\experiment_003_labels.csv'

# features_path = r'current_experiments\DATA\processed\experiment_003\experiment_003_cleaned.npy'
# labels_path = r'current_experiments\DATA\processed\experiment_003\experiment_003_labels.npy'

mat_path = r'current_experiments\DATA\Final\eeg_sign_epochs.npy'
label_csv_path = r'current_experiments\DATA\Final\eeg_sign_epochs.npy'

features_path = r'current_experiments\DATA\Final\eeg_sign_labels.npy'
labels_path = r'current_experiments\DATA\Final\eeg_sign_labels.npy'



# ---------- 1. 전처리 된 데이터셋 불러오기 ---------- #

dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()
print(eeg.shape[0])
#le = LabelEncoder()
le = load(r'current_experiments\MODEL\label_encoder.joblib')
csp = r'current_experiments\MODEL\csp_filters_IV.joblib'
encoded_labels = le.fit_transform(labels)


# ---------- 2. 데이터 증강하기 ---------- #

augmenter = EEGAugmenter(noise_level=0.01, max_shift=8)
aug_eeg, aug_labels = augmenter.augment(eeg, encoded_labels, num_augments=1)

print("증강 후 데이터 shape:", aug_eeg.shape)
print("증강 후 라벨 수:", len(aug_labels))

# ---------- 2. 특징 추출하기 ---------- #

print("특징 추출 중...")
extractor = DWTFeatureExtractor(wavelet='coif1', level=5)
time_features, freq_features = extractor.extract(aug_eeg)

# flat_time = extractor.flatten_feature_dict(time_features, extractor.bands)
# flat_freq = extractor.flatten_feature_dict(freq_features, extractor.bands)
csp_features = extractor.extract_csp_features(aug_eeg, aug_labels, n_components=4, save_path=csp)
# riemannian_features = extractor.extract_riemannian_features(aug_eeg)

features = np.concatenate([csp_features], axis=1)
n_epochs = aug_eeg.shape[0]
features = features.reshape(n_epochs, -1)

np.save(features_path, features)
np.save(labels_path, aug_labels)

# np.save('6_8_dwt_time.npy', flat_time)
# np.save('6_8_dwt_freq.npy', flat_freq)
# np.save('6_8_csp.npy', csp_features)
# np.save('1_5_riemann.npy', riemannian_features)

print(features.shape)


# dump(le, r'current_experiments\MODEL\label_encoder.joblib')


