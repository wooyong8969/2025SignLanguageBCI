import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from feature_extractor import DWTFeatureExtractor

# ===== 1. 경로 설정 =====
epochs_path = r'current_experiments\DATA\Final\eeg_sign_epochs.npy'
npy_labels_path = r'current_experiments\DATA\Final\eeg_sign_labels.npy'

features_path = r'current_experiments\DATA\Final\eeg_sign_features.npy'
labels_path = r'current_experiments\DATA\Final\eeg_sign_features_labels.npy'

label_encoder_path = r'current_experiments\MODEL\label_encoder_100.joblib'

# ===== 2. 데이터 불러오기 =====
eeg = np.load(epochs_path)     # (n_epochs, samples, channels)
labels = np.load(npy_labels_path)  # (n_epochs,)

print("원본 데이터 shape:", eeg.shape)
print("라벨 shape:", labels.shape)

# ===== 3. 0 라벨 제거 =====
mask = labels != 0
eeg = eeg[mask]
labels = labels[mask]

print("0 라벨 제거 후 데이터 shape:", eeg.shape)
print("0 라벨 제거 후 라벨 shape:", labels.shape)

# ===== 4. 라벨 인코딩 (새로 생성) =====
# le = LabelEncoder()
# encoded_labels = le.fit_transform(labels)

# # 인코더 저장
# dump(le, label_encoder_path)
# print(f"라벨 인코더 저장 완료: {label_encoder_path}")

le = load(label_encoder_path)
encoded_labels = le.transform(labels)
print("라벨 클래스:", le.classes_)

# ===== 5. 특징 추출 (예: CSP) =====
extractor = DWTFeatureExtractor(wavelet='coif1', level=5)
csp_features = extractor.extract_csp_features(
    eeg, encoded_labels, n_components=4
)

# 특징 벡터 형태로 변환
features = csp_features.reshape(eeg.shape[0], -1)

# ===== 6. 저장 =====
np.save(features_path, features)
np.save(labels_path, encoded_labels)

print("저장 완료:", features.shape)
