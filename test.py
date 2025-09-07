import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test 정확도 데이터
data = {
    "Pairs": [
        "hello vs help me",
        "hello vs sorry",
        "hello vs thank u",
        "help me vs sorry",
        "help me vs thank u",
        "sorry vs thank u",
    ],
    "Sign":  [0.6467, 0.6200, 0.7525, 0.6650, 0.7242, 0.7000],
    "Motor": [0.5833, 0.7108, 0.7817, 0.6725, 0.6058, 0.6725],
    "Speech":[0.6825, 0.6117, 0.6350, 0.6500, 0.5783, 0.5225],
}
df = pd.DataFrame(data)

# 행렬 전치 (Condition x Pairs)
mat = df.set_index("Pairs")[["Sign", "Motor", "Speech"]].T.values

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(mat, cmap="Blues", vmin=0.50, vmax=0.80, aspect="auto")

# 축 라벨 (축 전환 반영)
ax.set_xticks(np.arange(len(df["Pairs"])))
ax.set_xticklabels(df["Pairs"], rotation=30, ha="right")
ax.set_yticks(np.arange(3))
ax.set_yticklabels(["Sign", "Motor", "Speech"])
ax.set_ylabel("Condition")
ax.set_xlabel("Pairs")

# 각 셀에 값 표시 (폰트 크기 크게)
for i in range(mat.shape[0]):      # y축 (Condition)
    for j in range(mat.shape[1]):  # x축 (Pairs)
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                color="black", fontsize=27)

# 컬러바
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Accuracy")

# 제목 제거
ax.set_title("")

plt.tight_layout()
plt.show()
