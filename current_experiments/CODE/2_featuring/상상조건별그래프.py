import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 1. 데이터 불러오기
# ======================
file_motor = r"current_experiments\RESULTS\lda_accuracy_5foldcv_motor.xlsx"
file_sign = r"current_experiments\RESULTS\lda_accuracy_5foldcv_sign.xlsx"
file_speech = r"current_experiments\RESULTS\lda_accuracy_5foldcv_speech.xlsx"

data_motor = pd.read_excel(file_motor)
data_sign = pd.read_excel(file_sign)
data_speech = pd.read_excel(file_speech)

# Test Accuracy만 추출
motor_vals = data_motor["Test Accuracy"]
sign_vals = data_sign["Test Accuracy"]
speech_vals = data_speech["Test Accuracy"]

# ======================
# 2. 평균 & 표준편차 계산
# ======================
means = [speech_vals.mean(), motor_vals.mean(), sign_vals.mean()]
stds = [speech_vals.std(), motor_vals.std(), sign_vals.std()]
labels = ["Speech", "Motor", "Sign"]

summary_df = pd.DataFrame({
    "Mean": means,
    "Std": stds
}, index=labels).round(3)

print("=== Accuracy Summary (Mean ± Std) ===")
print(summary_df)

# ======================
# 3. 막대그래프 + 에러바
# ======================
fig, ax = plt.subplots(figsize=(6,5))

# 막대 (평균, 회색으로 채움)
bars = ax.bar(labels, means, color="lightgray", edgecolor="black")

# 에러바 (표준편차, cap 포함)
ax.errorbar(labels, means, yerr=stds, fmt='none',
            ecolor="black", elinewidth=1.5, capsize=10, capthick=1.5)

# y축 범위 0~1 고정
ax.set_ylim(0, 0.6)

ax.set_ylabel("Accuracy")
ax.set_title("4-class LDA Accuracy (5-Fold CV)")

plt.tight_layout()
plt.show()
