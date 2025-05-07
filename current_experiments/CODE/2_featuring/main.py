from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np

# 파일 경로 설정
mat_path = r'D:\W00Y0NG\PRGM2\2025BCI\current_experiments\DATA\processed\3seconds_001_cleaned.mat'
label_csv_path = r'D:\W00Y0NG\PRGM2\2025BCI\current_experiments\DATA\processed\3seconds_001_labels.csv'

# 데이터셋 생성 및 불필요한 break 제거
dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()

# 특징 추출기 초기화
extractor = DWTFeatureExtractor(wavelet='coif1', level=5)

# 특징 추출
time_features, freq_features = extractor.extract(eeg)

flat_time = extractor.flatten_feature_dict(time_features, extractor.bands)
flat_freq = extractor.flatten_feature_dict(freq_features, extractor.bands)

# 시간/주파수 특징 결합
features = np.concatenate([flat_time, flat_freq], axis=1)

# 에포크 수로 reshape
n_epochs = eeg.shape[0]
features = features.reshape(n_epochs, -1)

# 결과 확인
print("최종 feature shape:", features.shape)
print("최종 라벨 개수:", len(labels))
