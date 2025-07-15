from joblib import load
import numpy as np
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 저장된 모델 및 변환기 불러오기
clf_logreg = load(r'current_experiments\MODEL\trained_model_lda3_logreg.joblib')
scaler = load(r'current_experiments\MODEL\scaler_for_lda3.joblib')
lda = load(r'current_experiments\MODEL\lda3_reducer.joblib')

# 6-8 데이터 로드
features_68 = np.load(r'current_experiments\DATA\processed\experiment_001\experiment_001(6-8)_cleaned.npy')
labels_68 = np.load(r'current_experiments\DATA\processed\experiment_001\experiment_001(6-8)_labels.npy')

# 변환 적용
features_68_scaled = scaler.transform(features_68)
features_68_lda = lda.transform(features_68_scaled)

# 예측 및 평가
pred_68 = clf_logreg.predict(features_68_lda)
acc_68 = accuracy_score(labels_68, pred_68)

print(f"LDA(3)+LogReg (6-8 데이터) 검증 정확도 = {acc_68:.4f}")

# Confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(clf_logreg, features_68_lda, labels_68)
plt.title("LDA(3)+LogReg Confusion Matrix (Test 6-8)")
plt.tight_layout()
plt.show()
