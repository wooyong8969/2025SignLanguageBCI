from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ---------- 경로 설정 ----------
features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(6-8)_cleaned.npy'
true_label_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(6-8)_labels.npy'

# ---------- 모델 및 변환기 로드 ----------
clf_svm_rbf = load(r'current_experiments\MODEL\trained_model_pca_svmrbf.joblib')
scaler = load(r'current_experiments\MODEL\feature_selector.joblib')
pca = load(r'current_experiments\MODEL\pca_reducer.joblib')
le = load(r'current_experiments\MODEL\label_encoder.joblib')

# ---------- 테스트 데이터 로드 및 처리 ----------
features = np.load(features_path)
features_scaled = scaler.transform(features)
features_pca = pca.transform(features_scaled)
pred = clf_svm_rbf.predict(features_pca)

# ---------- 실제 라벨 npy 로드 ----------
true_labels = np.load(true_label_path)
true_encoded = true_labels

# ---------- 라벨 수 확인 ----------
assert len(true_labels) == len(pred), f"예측 수({len(pred)})와 라벨 수({len(true_labels)})가 다릅니다!"

# ---------- 평가 ----------
accuracy = accuracy_score(true_encoded, pred)
print(f"검증 정확도: {accuracy:.04f}")

# ---------- 혼동 행렬 ----------
try:
    disp = ConfusionMatrixDisplay.from_estimator(clf_svm_rbf, features_pca, true_encoded, display_labels=le.classes_)
except Exception:
    disp = ConfusionMatrixDisplay.from_estimator(clf_svm_rbf, features_pca, true_encoded)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
