import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ---------- 1. 데이터 로드 ---------- #
path1 = r'current_experiments\DATA\processed\experiment_001\experiment_001(1)_features.npy'
path2 = r'current_experiments\DATA\processed\experiment_001\experiment_001(2)_features.npy'

features_1 = np.load(path1)
features_2 = np.load(path2)

print("Shape (1):", features_1.shape)
print("Shape (2):", features_2.shape)

# ---------- 2. log1p + 평균 및 분산 비교 ---------- #
features_1_log = np.log1p(np.abs(features_1))
features_2_log = np.log1p(np.abs(features_2))

mean_1, std_1 = features_1_log.mean(axis=0), features_1_log.std(axis=0)
mean_2, std_2 = features_2_log.mean(axis=0), features_2_log.std(axis=0)

plt.figure(figsize=(12, 5))
plt.plot(mean_1, label='Experiment 001(1)')
plt.plot(mean_2, label='Experiment 001(2)', alpha=0.7)
plt.title('Mean of Each Feature (log1p transformed)')
plt.xlabel('Feature Index')
plt.ylabel('Mean Value')
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 3. PCA 시각화 ---------- #
combined = np.concatenate([features_1_log, features_2_log], axis=0)
labels = np.array([0] * len(features_1_log) + [1] * len(features_2_log))

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

pca = PCA(n_components=2)
proj = pca.fit_transform(combined_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], label='(1)', alpha=0.6)
plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], label='(2)', alpha=0.6)
plt.title('PCA Projection of Feature Distributions')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- 4. Domain Classification ---------- #
clf = LogisticRegression()
scores = cross_val_score(clf, combined_scaled, labels, cv=5)
print(f"Domain Classification Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
