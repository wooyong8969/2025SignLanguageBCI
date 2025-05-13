from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 경로 설정
features_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_test_cleaned.npy'
true_label_path = r'current_experiments\DATA\processed\experiment_001\experiment_001(1-4)_test_labels.csv'

# 모델 및 변환기 로드
clf = load('trained_model.joblib')
pipeline = load('feature_selector.joblib')
lda = load('lda_reducer.joblib')
le = load('label_encoder.joblib')

# 특징 로드
features = np.load(features_path)

# 변환
selected = pipeline.transform(features)
reduced = lda.transform(selected)
pred = clf.predict(reduced)
pred_labels = le.inverse_transform(pred)

# 실제 라벨 로드 및 Break 제외
df = pd.read_csv(true_label_path, header=None)
true_labels_full = df.iloc[:, 0].astype(str).tolist()
true_labels = [lbl for lbl in true_labels_full if lbl != 'Break']

# 라벨 수와 예측 수 확인
assert len(true_labels) == len(pred), f"예측 수({len(pred)})와 라벨 수({len(true_labels)})가 다릅니다!"

# 인코딩 및 평가
true_encoded = le.transform(true_labels)
accuracy = accuracy_score(true_encoded, pred)
print(f"정확도: {accuracy:.04f}%")

# 혼동 행렬 출력 및 시각화
cm = confusion_matrix(true_encoded, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (remove Break)")
plt.tight_layout()
plt.show()