import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# 1. 특징 및 라벨 로드
features_path = r'experiment_001(1-4)_train_8d_features.npy'
label_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_train_labels.npy'

features = np.load(features_path)
labels = np.load(label_path)

# 2. 훈련/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. k-NN 분류기 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 4. 예측 및 정확도 평가
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"k-NN 분류 정확도: {acc * 100:.2f}%")
print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 5. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

dump(knn, "knn_model_8d.joblib")