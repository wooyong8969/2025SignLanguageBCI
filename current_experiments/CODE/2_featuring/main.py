from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os

# X: 추출한 전체 특징 벡터
# encoded_labels: 라벨 (0 또는 1)

# 전처리 + L1 정규화 기반 특징 선택 파이프라인
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1))),
    # ('pca', PCA(n_components=30)),
])

features_path = r'current_experiments\DATA\processed\experiment_001_augmented_features.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001_augmented_labels.npy'

# ---------- 1. 특징 load ---------- #

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(features_path)
    encoded_labels = np.load(labels_path)
else:
    print("저장된 feature 파일이 없습니다.")


# ---------- 2. 특징 선택 ---------- #

# L1 정규화
selected_features = pipeline.fit_transform(features, encoded_labels)

# LDA 적용
lda = LinearDiscriminantAnalysis()
selected_features = lda.fit_transform(selected_features, encoded_labels)

print("기존 feature shape:", features.shape)
print("LDA 적용 후 feature shape:", selected_features.shape)
print("최종 라벨 개수:", len(encoded_labels))


# ---------- 3. 모델 학습 ---------- #

X_train, X_test, y_train, y_test = train_test_split(selected_features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

print(f"훈련 정확도: {train_acc:.4f}")
print(f"학습 정확도:  {test_acc:.4f}")
