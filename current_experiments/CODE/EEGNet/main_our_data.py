
import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from EEGModels import EEGNet
import matplotlib.pyplot as plt

# 1. 파일 경로 설정
mat_path = "current_experiments/DATA/processed/experiment_001/experiment_001(1-8)_cleaned.mat"
label_path = "current_experiments/DATA/processed/experiment_001/experiment_001(1-8)_labels.csv"

# 2. 데이터 불러오기
mat = scipy.io.loadmat(mat_path)
data = mat['EEG_clean']['data'][0,0]  # shape: (n_epochs, n_samples, n_channels)
srate = int(mat['EEG_clean']['srate'][0,0][0,0])

# 3. 데이터 차원 변환 및 정규화
X = np.transpose(data, (0, 2, 1))  # shape: (epochs, channels, samples)
X = X[..., np.newaxis]             # shape: (epochs, channels, samples, 1)

# 4. 라벨 불러오기 및 인코딩
y_str_all = pd.read_csv(label_path, header=None)[0].values

mask = y_str_all != "Break"
X = X[mask]
y_str = y_str_all[mask]

le = LabelEncoder()
y = le.fit_transform(y_str)
y_cat = to_categorical(y)
nb_classes = y_cat.shape[1]

# 5. 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# 6. EEGNet 모델 정의
model = EEGNet(
    nb_classes=nb_classes,
    Chans=X.shape[1],
    Samples=X.shape[2],
    dropoutRate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    dropoutType='Dropout'
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 모델 학습
history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=100,
    validation_data=(X_test, y_test),
    verbose=1
)

# 8. 정확도 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"훈련 정확도: {train_acc * 100:.2f}%")
print(f"테스트 정확도: {test_acc * 100:.2f}%")

# 9. 혼동 행렬 출력
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
