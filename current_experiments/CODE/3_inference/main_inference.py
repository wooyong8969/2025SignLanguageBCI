import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ---------- 1. 경로 설정 ---------- #
features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(2)_features.npy'  # raw 920차원
true_label_path = r'current_experiments\DATA\video\experiment_001_30_epochs.xlsx'
train_feat_path = r'current_experiments\DATA\processed\experiment_001\experiment_001_augmented3_features.npy'
train_label_path = r'current_experiments\DATA\processed\experiment_001\experiment_001_augmented3_labels.npy'

# ---------- 2. 모델 및 전처리기 로드 ---------- #
clf = load('trained_model.joblib')
pipeline = load('feature_selector.joblib')
lda = load('lda_reducer.joblib')
le = load('label_encoder.joblib')

# ---------- 3. 테스트 데이터 로드 및 전처리 ---------- #
test_features_raw = np.load(features_path)  # (n_samples, 920)
test_features_scaled = pipeline.transform(test_features_raw)  # (n_samples, 249)
test_features_lda = lda.transform(test_features_scaled)       # (n_samples, 3)

# ---------- 4. 실제 라벨 불러오기 ---------- #
df = pd.read_excel(true_label_path)
true_labels_full = df.iloc[:, 2].astype(str).tolist()
true_labels = [lbl for lbl in true_labels_full if lbl != 'Break']
true_encoded = le.transform(true_labels)

# ---------- 5. 예측 및 정확도 ---------- #
pred = clf.predict(test_features_lda)
accuracy = accuracy_score(true_encoded, pred)
print(f"정확도: {accuracy * 100:.2f}%")

# ---------- 6. 혼동 행렬 ---------- #
cmatrix = confusion_matrix(true_encoded, pred)
disp = ConfusionMatrixDisplay(cmatrix, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Break 제외)")
plt.tight_layout()
plt.show()

# ---------- 7. LDA 시각화 ---------- #
train_features_raw = np.load(train_feat_path)  # (n_train, 920)
train_labels = np.load(train_label_path)
train_features_scaled = pipeline.transform(train_features_raw)
train_features_lda = lda.transform(train_features_scaled)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = cm.tab10(np.linspace(0, 1, len(np.unique(train_labels))))

# 학습 데이터 시각화
for i, label in enumerate(np.unique(train_labels)):
    idx = train_labels == label
    ax.scatter(train_features_lda[idx, 0], train_features_lda[idx, 1], train_features_lda[idx, 2],
               label=f"Train Class {label}", alpha=0.6, color=colors[i])

# 테스트 데이터 시각화
ax.scatter(test_features_lda[:, 0], test_features_lda[:, 1], test_features_lda[:, 2],
           color='black', marker='x', label='Test samples', s=40)

ax.set_title("LDA Projection (Train + Test)")
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_zlabel("LD3")
ax.legend()
plt.tight_layout()
plt.show()