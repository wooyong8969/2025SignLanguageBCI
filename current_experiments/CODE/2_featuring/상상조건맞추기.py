from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os
from collections import Counter

# ================================
# 1. 세 가지 modality 데이터 로드
# ================================
base_path = r'current_experiments\DATA\Final'

features_sign = np.load(os.path.join(base_path, 'eeg_sign_features.npy'))
features_motor = np.load(os.path.join(base_path, 'eeg_motor_features.npy'))
features_speech = np.load(os.path.join(base_path, 'eeg_speech_features.npy'))

# 각 modality의 라벨 만들기 (단어 구분 X, modality 구분만)
labels_sign = np.array(['sign'] * len(features_sign))
labels_motor = np.array(['motor'] * len(features_motor))
labels_speech = np.array(['speech'] * len(features_speech))

# ================================
# 2. 합치기
# ================================
features = np.vstack([features_sign, features_motor, features_speech])
labels = np.concatenate([labels_sign, labels_motor, labels_speech])

print("전체 feature shape:", features.shape)
print("클래스 분포:", Counter(labels))

# ================================
# 3. 라벨 인코딩
# ================================
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)  # sign=0, motor=1, speech=2 (예시)

# ================================
# 4. 5-Fold Cross-Validation
# ================================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_accs, test_accs = [], []
cms = []

for train_idx, test_idx in kf.split(features, encoded_labels):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = encoded_labels[train_idx], encoded_labels[test_idx]

    # 파이프라인 (스케일링 + 특성 선택)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=10, max_iter=1000)
        ))
    ])

    X_train_sel = pipeline.fit_transform(X_train, y_train)
    X_test_sel = pipeline.transform(X_test)

    # LDA 분류기
    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train_sel, y_train)

    # 성능 저장
    train_accs.append(lda_clf.score(X_train_sel, y_train))
    test_accs.append(lda_clf.score(X_test_sel, y_test))

    y_pred = lda_clf.predict(X_test_sel)
    cms.append(confusion_matrix(y_test, y_pred, labels=np.unique(encoded_labels)))

print(f"[LDA 분류기] 훈련 정확도 (평균 ± 표준편차): {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"[LDA 분류기] 테스트 정확도 (평균 ± 표준편차): {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

# ================================
# 5. Confusion Matrix (평균)
# ================================
mean_cm = np.mean(cms, axis=0)
disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=le.classes_)
disp.plot(cmap="Greys")
plt.title("Confusion Matrix (5-Fold CV, sign vs motor vs speech)")
plt.show()
