import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os
from joblib import dump, load
from collections import Counter

le = load(r'current_experiments\MODEL\label_encoder_100.joblib')

features_path = r'current_experiments\DATA\Final\eeg_speech_features.npy'
labels_path = r'current_experiments\DATA\Final\eeg_speech_features_labels.npy'

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(features_path)
    encoded_labels = np.load(labels_path)
else:
    raise FileNotFoundError("feature 또는 label 파일이 존재하지 않습니다.")

print("기존 feature shape:", features.shape)
print("클래스 분포:", Counter(encoded_labels))

train_accs = []
test_accs = []
cms = []

for seed in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=seed
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=10, max_iter=1000)  # max_iter 증가
        ))
    ])

    X_train_sel = pipeline.fit_transform(X_train, y_train)
    X_test_sel = pipeline.transform(X_test)

    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train_sel, y_train)

    train_acc = lda_clf.score(X_train_sel, y_train)
    test_acc = lda_clf.score(X_test_sel, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    y_pred = lda_clf.predict(X_test_sel)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(encoded_labels))
    cms.append(cm)

print(f"[LDA 분류기] 훈련 정확도 평균: {np.mean(train_accs):.4f} (±{np.std(train_accs):.4f})")
print(f"[LDA 분류기] 테스트 정확도 평균: {np.mean(test_accs):.4f} (±{np.std(test_accs):.4f})")

# 평균 혼동 행렬 계산 및 시각화
mean_cm = np.mean(cms, axis=0)
marker_to_word = {
    101: 'hello',
    102: 'help me',
    103: 'sorry',
    104: 'thank u'
}
display_labels = [marker_to_word[cls] for cls in le.inverse_transform(np.unique(encoded_labels))]

disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=display_labels)
disp.plot()
plt.title("<sign> Mean Confusion Matrix (random_state 1~100)")
plt.grid(False)
plt.tight_layout()

plt.show()

import pandas as pd

# 정확도 값 DataFrame으로 변환
acc_df = pd.DataFrame({
    "Train Accuracy": train_accs,
    "Test Accuracy": test_accs
})

# 엑셀로 저장
save_path = r'current_experiments\RESULTS\lda_accuracy_100runs_speech.xlsx'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
acc_df.to_excel(save_path, index=False)

print(f"정확도 결과가 엑셀 파일로 저장되었습니다: {save_path}")
