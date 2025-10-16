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
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 라벨 인코더 불러오기
le = load(r'current_experiments\MODEL\label_encoder_100.joblib')

# feature, label 불러오기
features_path = r'current_experiments\DATA\Final\eeg_motor_features.npy'
labels_path = r'current_experiments\DATA\Final\eeg_motor_features_labels.npy'

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("저장된 feature 파일 불러오는 중...")
    features = np.load(features_path)
    encoded_labels = np.load(labels_path)
else:
    raise FileNotFoundError("feature 또는 label 파일이 존재하지 않습니다.")

print("기존 feature shape:", features.shape)
print("클래스 분포:", Counter(encoded_labels))

# 단일 seed 실험
train_accs = []
test_accs = []
cms = []

for seed in ():
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

    train_acc = lda_clf.score(X_train_sel, y_train)
    test_acc = lda_clf.score(X_test_sel, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    y_pred = lda_clf.predict(X_test_sel)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(encoded_labels))
    cms.append(cm)

print(f"[LDA 분류기] 훈련 정확도 평균: {np.mean(train_accs):.4f} (±{np.std(train_accs):.4f})")
print(f"[LDA 분류기] 테스트 정확도 평균: {np.mean(test_accs):.4f} (±{np.std(test_accs):.4f})")


# --------------------------------------------------
# ✅ LDA 3D 시각화 (전체 데이터 기준)
# --------------------------------------------------

# 전체 데이터에 대해 pipeline 적용 (누설 방지를 위해 독립적으로 fit)
full_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=10, max_iter=1000)
    ))
])
X_selected = full_pipeline.fit_transform(features, encoded_labels)

# LDA 차원 축소 (n_components=3)
lda_vis = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda_vis.fit_transform(X_selected, encoded_labels)

# 시각화용 데이터프레임 생성
df_lda = pd.DataFrame(X_lda, columns=['LD1', 'LD2', 'LD3'])
df_lda['label'] = le.inverse_transform(encoded_labels)

# 3D 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

labels_unique = df_lda['label'].unique()
colors = sns.color_palette('Set2', n_colors=len(labels_unique))

for label, color in zip(labels_unique, colors):
    subset = df_lda[df_lda['label'] == label]
    ax.scatter(subset['LD1'], subset['LD2'], subset['LD3'],
               label=label, color=color, s=60)

ax.set_title("LDA Feature Projection (3D)", fontsize=14)
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.set_zlabel("LD3")
ax.legend(title="Class")
plt.tight_layout()
plt.show()



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
plt.title("<motor> Confusion Matrix (random_state=10)")
plt.grid(False)
plt.tight_layout()
plt.show()

