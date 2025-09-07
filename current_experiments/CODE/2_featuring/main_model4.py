import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import os
from joblib import load
from collections import Counter
from itertools import combinations

le = load(r'current_experiments\MODEL\label_encoder_100.joblib')

features_path = r'current_experiments\DATA\Final\eeg_motorㅋ_features.npy'
labels_path = r'current_experiments\DATA\Final\eeg_sign_features_labels.npy'

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(features_path)
    encoded_labels = np.load(labels_path)
else:
    raise FileNotFoundError("feature 또는 label 파일이 존재하지 않습니다.")

print("기존 feature shape:", features.shape)
print("클래스 분포:", Counter(encoded_labels))

marker_to_word = {
    101: 'hello',
    102: 'help me',
    103: 'sorry',
    104: 'thank u'
}

all_classes = [101, 102, 103, 104]
results = {}

for cls1, cls2, cls3 in combinations(all_classes, 3):
    print(f"\n=== {marker_to_word[cls1]} vs {marker_to_word[cls2]} ({cls1} vs {cls2}) ===")
    # 2클래스만 선택
    mask = np.isin(le.inverse_transform(encoded_labels), [cls1, cls2])
    features_2 = features[mask]
    labels_2 = encoded_labels[mask]

    train_accs = []
    test_accs = []
    cms = []

    for seed in range(1, 101):
        X_train, X_test, y_train, y_test = train_test_split(
            features_2, labels_2, test_size=0.2, stratify=labels_2, random_state=seed
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

        train_acc = lda_clf.score(X_train_sel, y_train)
        test_acc = lda_clf.score(X_test_sel, y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        y_pred = lda_clf.predict(X_test_sel)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels_2))
        cms.append(cm)

    mean_train = np.mean(train_accs)
    std_train = np.std(train_accs)
    mean_test = np.mean(test_accs)
    std_test = np.std(test_accs)
    print(f"훈련 정확도 평균: {mean_train:.4f} (±{std_train:.4f})")
    print(f"테스트 정확도 평균: {mean_test:.4f} (±{std_test:.4f})")

    # # 평균 혼동 행렬 시각화
    # mean_cm = np.mean(cms, axis=0)
    # display_labels = [marker_to_word[cls] for cls in le.inverse_transform(np.unique(labels_2))]
    # disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=display_labels)
    # disp.plot()
    # plt.title(f"<motor> Mean Confusion Matrix ({marker_to_word[cls1]} vs {marker_to_word[cls2]})")
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()

    results[(cls1, cls2)] = {
        "train_acc_mean": mean_train,
        "train_acc_std": std_train,
        "test_acc_mean": mean_test,
        "test_acc_std": std_test
    }

print("\n=== 모든 2클래스 쌍 결과 요약 ===")
for (cls1, cls2), res in results.items():
    print(f"{marker_to_word[cls1]} vs {marker_to_word[cls2]}: "
          f"Train {res['train_acc_mean']:.4f}±{res['train_acc_std']:.4f}, "
          f"Test {res['test_acc_mean']:.4f}±{res['test_acc_std']:.4f}")