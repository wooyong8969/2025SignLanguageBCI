from sklearn.preprocessing import LabelEncoder
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
from collections import Counter

# ================================
# 1. 데이터 로드
# ================================
base_path = r'current_experiments\DATA\Final'

features_sign = np.load(os.path.join(base_path, 'eeg_sign_features.npy'))
labels_sign = np.load(os.path.join(base_path, 'eeg_sign_features_labels.npy'))

features_motor = np.load(os.path.join(base_path, 'eeg_motor_features.npy'))
labels_motor = np.load(os.path.join(base_path, 'eeg_motor_features_labels.npy'))

features_speech = np.load(os.path.join(base_path, 'eeg_speech_features.npy'))
labels_speech = np.load(os.path.join(base_path, 'eeg_speech_features_labels.npy'))

# ================================
# 2. modality+단어 → 12-class 라벨 생성
# ================================
def make_labels(modality, word_labels):
    return np.array([f"{modality}_{w}" for w in word_labels])

labels_sign_full = make_labels("sign", labels_sign)
labels_motor_full = make_labels("motor", labels_motor)
labels_speech_full = make_labels("speech", labels_speech)

# ================================
# 3. 합치기
# ================================
features = np.vstack([features_sign, features_motor, features_speech])
labels = np.concatenate([labels_sign_full, labels_motor_full, labels_speech_full])

print("전체 feature shape:", features.shape)
print("클래스 분포:", Counter(labels))

# ================================
# 4. 라벨 인코딩
# ================================
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
print("클래스 목록:", le.classes_)

# ================================
# 5. 학습/평가
# ================================
train_accs, test_accs = [], []
all_y_test, all_y_pred = [], []

for seed in (101,):
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=seed
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=10, max_iter=1000)
        ))
    ])

    X_train_sel = pipeline.fit_transform(X_train, y_train)
    X_test_sel = pipeline.transform(X_test)

    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train_sel, y_train)

    train_accs.append(lda_clf.score(X_train_sel, y_train))
    test_accs.append(lda_clf.score(X_test_sel, y_test))

    y_pred = lda_clf.predict(X_test_sel)
    all_y_test.append(y_test)
    all_y_pred.append(y_pred)

print(f"[4-class LDA] 훈련 정확도 평균: {np.mean(train_accs):.4f}")
print(f"[4-class LDA] 테스트 정확도 평균: {np.mean(test_accs):.4f}")

all_y_test = np.concatenate(all_y_test)
all_y_pred = np.concatenate(all_y_pred)
cm = confusion_matrix(all_y_test, all_y_pred, labels=np.unique(encoded_labels))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix (4-class: word only, 모든 테스트셋 합침)")
plt.show()