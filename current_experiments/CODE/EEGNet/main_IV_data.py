import mne
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from EEGModels import EEGNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import scipy.io as sio
import numpy as np
import pandas as pd

def preprocess_gdf_EEGNet_smr(gdf_path, resample_rate=128):
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, stim_channel='auto')

    raw.pick(range(22))

    # 필터링
    raw.notch_filter(freqs=50)
    raw.filter(4., 40., fir_design='firwin')

    # 리샘플링
    raw.resample(resample_rate)

    # 데이터 확인
    print(f"raw 값 범위: min={raw.get_data().min():.4f}, max={raw.get_data().max():.4f}")

    # 이벤트 정의 및 추출
    event_id = {'769': 0, '770': 1, '771': 2, '772': 3}
    events, _ = mne.events_from_annotations(raw, event_id=event_id)
    print("총 이벤트 수:", len(events))

    # 에포킹
    epochs_list = []
    labels = []

    for ev in events:
        label = ev[2]
        start = ev[0] + int(0.5 * resample_rate)
        end = ev[0] + int(2.5 * resample_rate)
        if end > raw.n_times:
            continue
        data, _ = raw[:, start:end]

        print(f"Trial data stats: min={data.min():.6f}, max={data.max():.6f}, std={data.std():.6f}")

        epochs_list.append(data)
        labels.append(label)

    print("총 추출된 trials 수:", len(labels))

    X = np.stack(epochs_list, axis=0)      # (trials, channels, samples)
    X = X[..., np.newaxis]                 # (trials, channels, samples, 1)
    X = X * 1e6  # 단위: microvolt로 변환
    print(f"X 최종 확인: min={X.min():.6f}, max={X.max():.6f}, std={X.std():.6f}")
    y = np.array(labels)

    return X, y

# 메인 실행
if __name__ == "__main__":
    gdf_path = r"current_experiments\DATA\open\BCI competition IV\A01T.gdf"
    X, y = preprocess_gdf_EEGNet_smr(gdf_path)

    print(f"\n✅ X shape: {X.shape}")
    print(f"✅ y shape: {y.shape}")
    print(f"X value range: min={X.min():.10f}, max={X.max():.10f}, mean={X.mean():.10f}, std={X.std():.10f}")
    print(f"전체 클래스 분포: {Counter(y)}")

    y_cat = to_categorical(y)
    nb_classes = y_cat.shape[1]
    print(f"클래스 수 (nb_classes): {nb_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.25, stratify=y, random_state=42
    )
    print(f"훈련 클래스 분포: {Counter(np.argmax(y_train, axis=1))}")
    print(f"검증 클래스 분포: {Counter(np.argmax(y_test, axis=1))}")

    model = EEGNet(
        nb_classes=nb_classes,
        Chans=X.shape[1],
        Samples=X.shape[2],
        dropoutRate=0.5,
        kernLength=32,
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

    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=100,
        validation_data=(X_test, y_test),
        verbose=1
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n훈련 정확도: {train_acc * 100:.2f}%")
    print(f"테스트 정확도: {test_acc * 100:.2f}%")

    # Loss 그래프
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    X = X.squeeze(-1)  # shape: (n_trials, n_channels, n_samples)
    X = np.transpose(X, (0, 2, 1))  # → shape: (n_trials, n_samples, n_channels)

    # .mat 저장
    sio.savemat(r"current_experiments\DATA\open\BCI competition IV\experiment_iv_cleaned.mat", {
        'EEG_clean': {
            'data': X.astype(np.float32),
            'srate': np.array([[128]])
        }
    })
    print(".mat 저장 완료")

    # .csv 저장
    label_df = pd.DataFrame(y)
    label_df.to_csv(r"current_experiments\DATA\open\BCI competition IV\experiment_iv_labels.csv", header=False, index=False)
    print(".csv 저장 완료")
