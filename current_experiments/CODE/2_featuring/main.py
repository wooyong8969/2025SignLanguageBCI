from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

mat_path = r'D:\W00Y0NG\PRGM2\2025BCI\current_experiments\DATA\processed\experiment_001_cleaned.mat'
label_csv_path = r'D:\W00Y0NG\PRGM2\2025BCI\current_experiments\DATA\processed\experiment_001_labels.csv'

# ---------- 1.  전처리 된 데이터셋 불러오기 ---------- #
dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()


# ---------- 2. 특징 추출하기 ---------- #
extractor = DWTFeatureExtractor(wavelet='coif1', level=5)

time_features, freq_features = extractor.extract(eeg)

flat_time = extractor.flatten_feature_dict(time_features, extractor.bands)
flat_freq = extractor.flatten_feature_dict(freq_features, extractor.bands)

csp_features = extractor.extract_csp_features(eeg, labels, n_components=4)

features = np.concatenate([flat_time, flat_freq, csp_features], axis=1)

n_epochs = eeg.shape[0]
features = features.reshape(n_epochs, -1)

print("최종 feature shape:", features.shape)
print("최종 라벨 개수:", len(labels))

# ---------- 3. 모델 학습 ---------- #

le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, stratify=y, random_state=42)

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1, gamma='scale'))

clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"훈련 정확도: {train_acc:.4f}")
print(f"학습 정확도:  {test_acc:.4f}")