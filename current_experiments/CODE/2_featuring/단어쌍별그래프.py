import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. 단어쌍 이름
# ===============================
class_pairs = [
    "hello vs help me",
    "hello vs sorry",
    "hello vs thank u",
    "help me vs sorry",
    "help me vs thank u",
    "sorry vs thank u"
]

# ===============================
# 2. 5-Fold CV 결과 입력
#    (수어 → 운동 → 발화 순서)
# ===============================

# 수어 (Sign)
sign_mean = [0.7667, 0.5667, 0.6000, 0.7167, 0.7167, 0.6500]
sign_std  = [0.1700, 0.0972, 0.0624, 0.1130, 0.1546, 0.1616]

# 운동 (Motor)
motor_mean = [0.5833, 0.7167, 0.7000, 0.7167, 0.7167, 0.7833]
motor_std  = [0.0913, 0.0850, 0.1000, 0.0850, 0.1130, 0.1247]

# 발화 (Speech)
speech_mean = [0.6500, 0.5833, 0.5333, 0.6167, 0.5667, 0.5333]
speech_std  = [0.0624, 0.0527, 0.1247, 0.0850, 0.0972, 0.1130]

# ===============================
# 3. 그래프 설정
# ===============================
x = np.arange(len(class_pairs))
width = 0.25

# 흑백/그레이스케일 색상
colors = ["0.85", "0.6", "0.3"]  # 밝은회색, 중간회색, 진회색
labels = ["SI", "MI", "SLI"]
means = [speech_mean, motor_mean, sign_mean]
stds = [speech_std, motor_std, sign_std]


error_params = dict(elinewidth=1.2, capsize=8, capthick=1.2, ecolor="black")

fig, ax = plt.subplots(figsize=(12, 6))

# ===============================
# 4. 막대 그리기 (회색조)
# ===============================
for i in range(3):
    ax.bar(
        x + (i-1)*width, means[i], width,
        yerr=stds[i], error_kw=error_params,
        label=labels[i], color=colors[i], edgecolor="black"
    )

ax.set_title("2-class Word Pair LDA Accuracy (5-Fold CV)", fontsize=14)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_pairs, rotation=30, ha="right", fontsize=11)
ax.legend(fontsize=11, loc=4)
ax.set_ylim(0.0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
