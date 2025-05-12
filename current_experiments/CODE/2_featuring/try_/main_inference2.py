import os
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from eeg_dataset_maker import EEGDataset
from eeg_augmenter import EEGAugmenter
from feature_extractor2 import DWTFeatureExtractor

# ---------- 1. 경로 설정 ---------- #
mat_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(1)_cleaned.mat'
label_csv_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(1)_labels.csv'
label_encoder_path = 'label_encoder.joblib'

# ---------- 2. 데이터 불러오기 ---------- #
dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()
eeg, labels, fs = dataset.get_data()

if os.path.exists(label_encoder_path):
    le = load(label_encoder_path)
else:
    raise FileNotFoundError("label_encoder.joblib 파일이 없습니다.")

encoded_labels = le.transform(labels)

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
labels = np.array(aug_labels)

print("최종 feature shape:", features.shape)
print("클래스 분포:", Counter(labels))

# ---------- 5. 모델 불러오기 ---------- #
scaler = load('scaler.joblib')
selector = load('feature_selector.joblib')
pca = load('pca_reducer.joblib')
model = load('trained_model.joblib')

# ---------- 6. 전처리 및 예측 ---------- #
X_scaled = scaler.transform(features)
X_selected = selector.transform(X_scaled)
X_pca = pca.transform(X_selected)
y_pred = model.predict(X_pca)

# ---------- 7. 평가 ---------- #
acc = accuracy_score(labels, y_pred)
print("\n========== 새 데이터 평가 결과 (experiment_001(1)) ==========")
print(f"정확도: {acc:.4f}")
print("\n[Classification Report]")
print(classification_report(labels, y_pred))

cm = confusion_matrix(labels, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title("Confusion Matrix (experiment_001(1))")
plt.grid(False)
plt.tight_layout()
plt.show()
