import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# 데이터 로드
features = np.load(r'current_experiments\DATA\open\BCI competition IV\experiment_iv_cleaned.npy')
labels = np.load(r'current_experiments\DATA\open\BCI competition IV\experiment_iv_labels.npy')

# Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

result_list = []

def try_all_models(X_train, X_test, prefix=""):
    res = []
    # 1. LDA(3) + LogReg
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    clf_logreg = LogisticRegression(max_iter=500, random_state=42)
    clf_logreg.fit(X_train_lda, y_train)
    train_acc = clf_logreg.score(X_train_lda, y_train)
    test_acc = clf_logreg.score(X_test_lda, y_test)
    res.append((f"{prefix}LDA(3)+LogReg", train_acc, test_acc, clf_logreg, X_test_lda))

    # 2. LDA(3) + SVM-linear
    clf_svm_linear = SVC(kernel='linear', C=1, random_state=42)
    clf_svm_linear.fit(X_train_lda, y_train)
    train_acc = clf_svm_linear.score(X_train_lda, y_train)
    test_acc = clf_svm_linear.score(X_test_lda, y_test)
    res.append((f"{prefix}LDA(3)+SVM-linear", train_acc, test_acc, clf_svm_linear, X_test_lda))

    # 3. LDA(3) + SVM-rbf
    clf_svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    clf_svm_rbf.fit(X_train_lda, y_train)
    train_acc = clf_svm_rbf.score(X_train_lda, y_train)
    test_acc = clf_svm_rbf.score(X_test_lda, y_test)
    res.append((f"{prefix}LDA(3)+SVM-rbf", train_acc, test_acc, clf_svm_rbf, X_test_lda))

    # 4. PCA(3) + LogReg
    pca = PCA(n_components=3, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    clf_logreg2 = LogisticRegression(max_iter=500, random_state=42)
    clf_logreg2.fit(X_train_pca, y_train)
    train_acc = clf_logreg2.score(X_train_pca, y_train)
    test_acc = clf_logreg2.score(X_test_pca, y_test)
    res.append((f"{prefix}PCA(3)+LogReg", train_acc, test_acc, clf_logreg2, X_test_pca))

    # 5. PCA(3) + SVM-linear
    clf_svm_linear2 = SVC(kernel='linear', C=1, random_state=42)
    clf_svm_linear2.fit(X_train_pca, y_train)
    train_acc = clf_svm_linear2.score(X_train_pca, y_train)
    test_acc = clf_svm_linear2.score(X_test_pca, y_test)
    res.append((f"{prefix}PCA(3)+SVM-linear", train_acc, test_acc, clf_svm_linear2, X_test_pca))

    # 6. PCA(3) + SVM-rbf
    clf_svm_rbf2 = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    clf_svm_rbf2.fit(X_train_pca, y_train)
    train_acc = clf_svm_rbf2.score(X_train_pca, y_train)
    test_acc = clf_svm_rbf2.score(X_test_pca, y_test)
    res.append((f"{prefix}PCA(3)+SVM-rbf", train_acc, test_acc, clf_svm_rbf2, X_test_pca))

    # 7. Random Forest (전체 feature)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf_rf.fit(X_train, y_train)
    train_acc = clf_rf.score(X_train, y_train)
    test_acc = clf_rf.score(X_test, y_test)
    res.append((f"{prefix}RF(all)", train_acc, test_acc, clf_rf, X_test))

    # 8. LogReg (전체 feature)
    clf_logreg_all = LogisticRegression(max_iter=500, random_state=42)
    clf_logreg_all.fit(X_train, y_train)
    train_acc = clf_logreg_all.score(X_train, y_train)
    test_acc = clf_logreg_all.score(X_test, y_test)
    res.append((f"{prefix}LogReg(all)", train_acc, test_acc, clf_logreg_all, X_test))

    # 9. SVM-linear (전체 feature)
    clf_svm_linear_all = SVC(kernel='linear', C=1, random_state=42)
    clf_svm_linear_all.fit(X_train, y_train)
    train_acc = clf_svm_linear_all.score(X_train, y_train)
    test_acc = clf_svm_linear_all.score(X_test, y_test)
    res.append((f"{prefix}SVM-linear(all)", train_acc, test_acc, clf_svm_linear_all, X_test))

    # 10. SVM-rbf (전체 feature)
    clf_svm_rbf_all = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    clf_svm_rbf_all.fit(X_train, y_train)
    train_acc = clf_svm_rbf_all.score(X_train, y_train)
    test_acc = clf_svm_rbf_all.score(X_test, y_test)
    res.append((f"{prefix}SVM-rbf(all)", train_acc, test_acc, clf_svm_rbf_all, X_test))

    return res

# 원본/표준화 데이터 실험
result_list += try_all_models(X_train_scaled, X_test_scaled, prefix="")

# [B] L1 정규화 기반 특징 선택 후 실험
l1_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42))
X_train_l1 = l1_selector.fit_transform(X_train_scaled, y_train)
X_test_l1 = l1_selector.transform(X_test_scaled)
print(f"L1 정규화 후 feature 개수: {X_train_l1.shape[1]}")
result_list += try_all_models(X_train_l1, X_test_l1, prefix="L1+")

# 결과 출력
for name, train_acc, test_acc, _, _ in result_list:
    print(f"{name}: 훈련 정확도 = {train_acc:.4f} / 테스트 정확도 = {test_acc:.4f}")

# 가장 성능 좋은 조합 출력
best_combo = max(result_list, key=lambda x: x[2])
print(f"\n>>> 최적 조합: {best_combo[0]}, 테스트 정확도 = {best_combo[2]:.4f}")

# Confusion matrix 시각화 (최고 정확도 조합 모델)
best_name, _, _, best_model, best_X_test = best_combo
disp = ConfusionMatrixDisplay.from_estimator(best_model, best_X_test, y_test)
plt.title(f"{best_name} Confusion Matrix")
plt.show()
