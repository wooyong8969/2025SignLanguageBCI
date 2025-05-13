from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
import os

# X: 추출한 전체 특징 벡터
# encoded_labels: 라벨 (0 또는 1)

# 전처리 + L1 정규화 기반 특징 선택 파이프라인
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1))),
])
features_path = r'current_experiments\DATA\processed\experiment_001_processed_features.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001_encoded_labels.npy'


# ---------- 1. 특징 load ---------- #

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(features_path)
    encoded_labels = np.load(labels_path)
else:
    print("저장된 feature 파일이 없습니다.")


# ---------- 2. 특징 선택 ---------- #

selected_features = pipeline.fit_transform(features, encoded_labels)

# # 특징 선택 적용 p-value는 실패함
# features = pipeline.fit_transform(features, encoded_labels)

# # ANOVA F-test를 기반으로 특징 선택
# selector = SelectKBest(score_func=f_classif, k='all')  # 전체 p-value 확인
# selector.fit(features, encoded_labels)

# # 각 특징에 대한 p-value 확인
# p_values = selector.pvalues_

# # p < 0.05인 특징만 선택
# mask = p_values < 0.1
# selected_features = selected_features[:, mask]

print("기존 feature shape:", features.shape)
print("최종 feature shape:", selected_features.shape)
print("최종 라벨 개수:", len(encoded_labels))

# ---------- 3. 모델 학습 ---------- #

X_train, X_test, y_train, y_test = train_test_split(selected_features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))

clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"훈련 정확도: {train_acc:.4f}")
print(f"학습 정확도:  {test_acc:.4f}")