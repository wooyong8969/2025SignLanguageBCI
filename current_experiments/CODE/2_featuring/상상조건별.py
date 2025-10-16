import numpy as np
import os
from collections import Counter
import pandas as pd
from joblib import load

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ========================================
# 1. 환경 설정
# ========================================
n_repeats = 20
n_splits = 5
base_path = r'current_experiments\DATA\Final'
save_dir = r'current_experiments\RESULTS'
os.makedirs(save_dir, exist_ok=True)

# label encoder 불러오기
le = load(r'current_experiments\MODEL\label_encoder_100.joblib')

# modality 목록
modalities = {
    "motor": ("eeg_motor_features.npy", "eeg_motor_features_labels.npy"),
    "speech": ("eeg_speech_features.npy", "eeg_speech_features_labels.npy"),
    "sign": ("eeg_sign_features.npy", "eeg_sign_features_labels.npy"),
}

# ========================================
# 2. 함수 정의
# ========================================
def run_repeated_cv(features, labels, n_repeats=20, n_splits=5):
    all_train_accs = []
    all_test_accs = []

    for rep in range(n_repeats):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for train_idx, test_idx in kf.split(features, labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # 파이프라인: 스케일링 + 특성 선택(L1 로지스틱)
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

            # 정확도
            train_acc = lda_clf.score(X_train_sel, y_train)
            test_acc = lda_clf.score(X_test_sel, y_test)
            all_train_accs.append(train_acc)
            all_test_accs.append(test_acc)

    return np.array(all_train_accs), np.array(all_test_accs)

# ========================================
# 3. 각 modality 반복 CV 수행
# ========================================
for modality, (feat_file, label_file) in modalities.items():
    feat_path = os.path.join(base_path, feat_file)
    label_path = os.path.join(base_path, label_file)

    if not (os.path.exists(feat_path) and os.path.exists(label_path)):
        print(f"[WARN] {modality} 데이터가 존재하지 않습니다. → 건너뜀")
        continue

    print(f"\n[{modality.upper()}] 데이터 불러오는 중...")
    features = np.load(feat_path)
    encoded_labels = np.load(label_path)
    print("Feature shape:", features.shape)
    print("클래스 분포:", Counter(encoded_labels))

    # 5-Fold CV × 20회 실행
    train_accs, test_accs = run_repeated_cv(features, encoded_labels, n_repeats, n_splits)

    # 성능 요약
    print(f"[{modality.upper()}] Train Acc: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"[{modality.upper()}] Test Acc:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

    # 결과 저장
    acc_df = pd.DataFrame({
        "Train Accuracy": train_accs,
        "Test Accuracy": test_accs
    })
    save_path = os.path.join(save_dir, f"lda_accuracy_5foldcv20_{modality}.xlsx")
    acc_df.to_excel(save_path, index=False)
    print(f"저장 완료 → {save_path}")
