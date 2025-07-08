import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from joblib import dump
from collections import Counter

# ---------- 1. 파일 경로 ---------- #

features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-5)_cleaned.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-5)_labels.npy'

# ---------- 2. 특징 load ---------- #
if os.path.exists(features_path) and os.path.exists(labels_path):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(features_path)
    encoded_labels = np.load(labels_path)
else:
    raise FileNotFoundError("feature 또는 label 파일이 존재하지 않습니다.")

print("기존 feature shape:", features.shape)
print("클래스 분포:", Counter(encoded_labels))

# ---------- 3. Train/Test 분리 ---------- #
X_train, X_test, y_train, y_test = train_test_split(
    features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# ---------- 4. 표준화 + PCA 차원 축소 ---------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA 차원 축소 (n_components=3)
pca = PCA(n_components=3, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("PCA 적용 후 train feature shape:", X_train_pca.shape)

# ---------- 5. SVM(RBF) 모델 학습 ---------- #
clf_svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
clf_svm_rbf.fit(X_train_pca, y_train)

# ---------- 6. 평가 ---------- #
train_acc = clf_svm_rbf.score(X_train_pca, y_train)
test_acc = clf_svm_rbf.score(X_test_pca, y_test)

disp = ConfusionMatrixDisplay.from_estimator(clf_svm_rbf, X_test_pca, y_test)
plt.title("PCA(3) + SVM(RBF) Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

print(f"PCA+SVM(RBF) 훈련 정확도: {train_acc:.4f}")
print(f"PCA+SVM(RBF) 테스트 정확도: {test_acc:.4f}")

# ---------- 7. 모델 저장 ---------- #
dump(clf_svm_rbf, r'current_experiments\MODEL\trained_model_pca_svmrbf.joblib')
dump(scaler, r'current_experiments\MODEL\feature_selector.joblib')
dump(pca, r'current_experiments\MODEL\pca_reducer.joblib')

# ---------- 8. LDA 3D 시각화 ---------- #
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y_train):
    idx = y_train == label
    ax.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], X_train_pca[idx, 2],
                label=f"Class {label}", alpha=0.3)
ax.set_title("LDA Projection (3D)")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_zlabel("LD3")
ax.legend()
plt.tight_layout()
plt.show()
