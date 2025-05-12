import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump
from collections import Counter

# ---------- 1. 파일 경로 ---------- #
feature_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(2)_TFS_features.npy'
label_path = r'current_experiments/DATA/processed/experiment_001/experiment_001(2)_TFS_labels.npy'

# ---------- 2. 데이터 로딩 ---------- #
if os.path.exists(feature_path) and os.path.exists(label_path):
    features = np.load(feature_path)
    labels = np.load(label_path)
else:
    raise FileNotFoundError("특징 또는 라벨 파일이 존재하지 않습니다.")

print("특징 shape:", features.shape)
print("클래스 분포:", Counter(labels))

# ---------- 3. Train/Test 분리 ---------- #
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

# ---------- 4. 표준화 ---------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dump(scaler, 'scaler.joblib')

# ---------- 5. L1 기반 특징 선택 ---------- #
sfm = SelectFromModel(LogisticRegression(penalty='l1', C=0.5, solver='liblinear'))
sfm.fit(X_train_scaled, y_train)
X_train_sel = sfm.transform(X_train_scaled)
X_test_sel = sfm.transform(X_test_scaled)
dump(sfm, 'feature_selector.joblib')

print(f"선택된 특징 수: {X_train_sel.shape[1]}")

# ---------- 6. PCA 차원 축소 ---------- #
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_sel)
X_test_pca = pca.transform(X_test_sel)
dump(pca, 'pca_reducer.joblib')

# ---------- 7. 분류기 학습 ---------- #
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train_pca, y_train)
dump(clf, 'trained_model.joblib')

# ---------- 8. 평가 ---------- #
train_acc = clf.score(X_train_pca, y_train)
test_acc = clf.score(X_test_pca, y_test)
scores = cross_val_score(clf, X_train_pca, y_train, cv=5)
y_pred = clf.predict(X_test_pca)

print("\n========== 모델 성능 평가 ==========")
print(f"훈련 정확도: {train_acc:.4f}")
print(f"테스트 정확도: {test_acc:.4f}")
print(f"5-Fold CV 정확도: {scores.mean():.4f} ± {scores.std():.4f}")
print("\n[Classification Report]")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()