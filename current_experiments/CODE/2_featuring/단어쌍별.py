import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import combinations

# features, labels 불러오기
features = np.load(r'current_experiments\DATA\Final\eeg_sign_features.npy')
labels = np.load(r'current_experiments\DATA\Final\eeg_sign_features_labels.npy')

# 클래스 이름 불러오기 (label encoder 이용)
from joblib import load
le = load(r'current_experiments\MODEL\label_encoder_100.joblib')
class_names = le.classes_

# 분류기 정의 (스케일링 + SVM)
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear"))
])

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 모든 2-class 조합 수행
results = {}
for c1, c2 in combinations(np.unique(labels), 2):
    # 선택된 두 클래스 데이터만 추출
    mask = np.isin(labels, [c1, c2])
    X, y = features[mask], labels[mask]

    fold_acc = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fold_acc.append(accuracy_score(y_test, y_pred))

    results[f"{class_names[c1]} vs {class_names[c2]}"] = (
        np.mean(fold_acc), np.std(fold_acc)
    )

# 결과 출력
print("=== 5-Fold CV 결과 (Test Accuracy) ===")
for k, (mean_acc, std_acc) in results.items():
    print(f"{k}: {mean_acc:.4f} ± {std_acc:.4f}")
