import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 저장된 k-NN 모델 불러오기
knn = load("knn_model_8d.joblib")

# 2. 새로운 EEG 특징 (8차원 임베딩)과 실제 라벨 로드
new_features = np.load(r'experiment_001(1-4)_test_8d_features.npy')
true_labels = np.load(r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_test_labels.npy')

# 3. 예측 수행
pred_labels = knn.predict(new_features)

# 4. 정확도 및 보고서 출력
acc = accuracy_score(true_labels, pred_labels)
print(f"\n새로운 데이터에 대한 정확도: {acc * 100:.2f}%")
print("\n분류 보고서:")
print(classification_report(true_labels, pred_labels))

# 5. 혼동 행렬 시각화
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.title("Confusion Matrix for New Data")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
