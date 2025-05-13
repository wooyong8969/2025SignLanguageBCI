import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os
from joblib import dump, load
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# ---------- 1. 파일 경로 ---------- #
<<<<<<< HEAD
features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(3)_features.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(3)_labels.npy'
=======
features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-6)_train_cleaned.npy'
labels_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-6)_train_labels.npy'
>>>>>>> wooyong8969-clean

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

# ---------- 4. 특징 선택 (L1 기반) ---------- #
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1)))
])

X_train_sel = pipeline.fit_transform(X_train, y_train)
X_test_sel = pipeline.transform(X_test)

# ---------- 5. LDA 차원 축소 ---------- #
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_sel, y_train)
X_test_lda = lda.transform(X_test_sel)

print("LDA 적용 후 train feature shape:", X_train_lda.shape)

# ---------- 6. 모델 학습 ---------- #
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))
clf.fit(X_train_lda, y_train)

# ---------- 7. 평가 ---------- #
train_acc = clf.score(X_train_lda, y_train)
test_acc = clf.score(X_test_lda, y_test)

disp = ConfusionMatrixDisplay.from_estimator(clf, X_test_lda, y_test)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

print(f"훈련 정확도: {train_acc:.4f}")
print(f"테스트 정확도: {test_acc:.4f}")

# ---------- 8. LDA 3D 시각화 ---------- #
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y_train):
    idx = y_train == label
    ax.scatter(X_train_lda[idx, 0], X_train_lda[idx, 1], X_train_lda[idx, 2],
                label=f"Class {label}", alpha=0.6)
ax.set_title("LDA Projection (3D)")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_zlabel("LD3")
ax.legend()
plt.tight_layout()
plt.show()

# ---------- 9. 모델 저장 ---------- #
dump(clf, r'current_experiments\MODEL\trained_model.joblib')
dump(pipeline, r'current_experiments\MODEL\feature_selector.joblib')
dump(lda, r'current_experiments\MODEL\lda_reducer.joblib')
