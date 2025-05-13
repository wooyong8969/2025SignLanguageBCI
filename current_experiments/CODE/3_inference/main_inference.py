from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ---------- 경로 설정 ----------
features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-6)_test_cleaned.npy'
true_label_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-6)_test_labels.csv'
train_features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_train_cleaned.npy'
train_label_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_train_labels.csv'

<<<<<<< HEAD
# 모델 및 변환기 로드
clf = load(r'current_experiments\MODEL\trained_model.joblib')
pipeline = load(r'current_experiments\MODEL\feature_selector.joblib')
lda = load(r'current_experiments\MODEL\lda_reducer.joblib')
le = load(r'current_experiments\MODEL\label_encoder.joblib')
=======
# ---------- 모델 및 변환기 로드 ----------
clf = load('trained_model.joblib')
pipeline = load('feature_selector.joblib')
lda = load('lda_reducer.joblib')
le = load('label_encoder.joblib')
>>>>>>> wooyong8969-clean

# ---------- 테스트 데이터 로드 및 처리 ----------
features = np.load(features_path)
selected = pipeline.transform(features)
reduced = lda.transform(selected)
pred = clf.predict(reduced)
pred_labels = le.inverse_transform(pred)

# ---------- 실제 라벨 로드 및 Break 제외 ----------
df = pd.read_csv(true_label_path, header=None)
true_labels_full = df.iloc[:, 0].astype(str).tolist()
true_labels = [lbl for lbl in true_labels_full if lbl != 'Break']

# ---------- 라벨 수 확인 ----------
assert len(true_labels) == len(pred), f"예측 수({len(pred)})와 라벨 수({len(true_labels)})가 다릅니다!"

# ---------- 평가 ----------
true_encoded = le.transform(true_labels)
accuracy = accuracy_score(true_encoded, pred)
print(f"정확도: {accuracy:.04f}%")

# ---------- 혼동 행렬 ----------
cmt = confusion_matrix(true_encoded, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cmt, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (remove Break)")
plt.tight_layout()
plt.show()

# ---------- 훈련 데이터 로드 ----------
train_features = np.load(train_features_path)
train_label_df = pd.read_csv(train_label_path, header=None)
train_labels_full = train_label_df.iloc[:, 0].astype(str).tolist()
train_labels = [lbl for lbl in train_labels_full if lbl != 'Break']
train_encoded = le.transform(train_labels)

# ---------- 훈련 데이터 변환 ----------
train_selected = pipeline.transform(train_features)
train_reduced = lda.transform(train_selected)

# ---------- 3D 시각화 ----------
unique_classes = np.unique(train_encoded)
n_classes = len(unique_classes)
colors = cm.get_cmap('tab10', n_classes)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# ---------- 훈련 데이터 시각화 (동그라미) ----------
for idx, class_idx in enumerate(unique_classes):
    ix = np.where(train_encoded == class_idx)
    ax.scatter(train_reduced[ix, 0], train_reduced[ix, 1], train_reduced[ix, 2],
               color=colors(idx),
               label=f'Train - {le.inverse_transform([class_idx])[0]}',
               marker='o', alpha=0.3)

# ---------- 테스트 데이터 시각화 (세모) ----------
for idx, class_idx in enumerate(unique_classes):
    ix = np.where(true_encoded == class_idx)
    ax.scatter(reduced[ix, 0], reduced[ix, 1], reduced[ix, 2],
               color=colors(idx),
               label=f'Test - {le.inverse_transform([class_idx])[0]}',
               marker='^', edgecolor='k', linewidth=0.3)

# ---------- 축 설정 ----------
ax.set_xlabel('LDA 1')
ax.set_ylabel('LDA 2')
ax.set_zlabel('LDA 3')
ax.set_title('3D LDA Projection (Train vs Test)')

# ---------- 범례 설정 ----------
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=9)
plt.tight_layout()
plt.show()