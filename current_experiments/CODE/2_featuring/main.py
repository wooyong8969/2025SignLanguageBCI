from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pyriemann
import os

# X: 추출한 전체 특징 벡터
# encoded_labels: 라벨 (0 또는 1)

# 전처리 + L1 정규화 기반 특징 선택 파이프라인
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1))),
])

mat_path = r'current_experiments\DATA\processed\experiment_001_cleaned.mat'
label_csv_path = r'current_experiments\DATA\processed\experiment_001_labels.csv'

# ---------- 1.  전처리 된 데이터셋 load ---------- #
dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# ---------- 2. 특징 추출 ---------- #

if os.path.exists(r'current_experiments\CODE\2_featuring\processed_features_.npy') and os.path.exists(r'current_experiments\CODE\2_featuring\encoded_labels_.npy'):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(r'current_experiments\CODE\2_featuring\processed_features_.npy')
    encoded_labels = np.load(r'current_experiments\CODE\2_featuring\encoded_labels_.npy')
else:
    print("특징 추출 중...")
    extractor = DWTFeatureExtractor(wavelet='coif1', level=5)
    time_features, freq_features = extractor.extract(eeg)

    flat_time = extractor.flatten_feature_dict(time_features, extractor.bands)
    flat_freq = extractor.flatten_feature_dict(freq_features, extractor.bands)
    csp_features = extractor.extract_csp_features(eeg, labels, n_components=4)
    riemannian_features = extractor.extract_riemannian_features(eeg)

    features = np.concatenate([flat_time, flat_freq], axis=1)
    n_epochs = eeg.shape[0]
    features = features.reshape(n_epochs, -1)

    np.save(r'current_experiments\CODE\2_featuring\processed_features_.npy', features)
    np.save(r'current_experiments\CODE\2_featuring\encoded_labels_.npy', encoded_labels)

# ---------- 4. 특징 선택 ---------- #
selected_features = pipeline.fit_transform(features, encoded_labels)

print("기존 feature shape:", features.shape)
print("최종 feature shape:", selected_features.shape)
print("최종 라벨 개수:", len(labels))

# ---------- 3. 모델 학습 ---------- #

X_train, X_test, y_train, y_test = train_test_split(selected_features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))

clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"훈련 정확도: {train_acc:.4f}")
print(f"학습 정확도:  {test_acc:.4f}")