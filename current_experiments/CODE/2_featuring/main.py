from eeg_dataset import EEGDataset
from feature_extractor import DWTFeatureExtractor
import numpy as np

mat_path = r'D:\W00Y0NG\PRGM2\2025BCI\current_experiments\DATA\processed\experiment_001_cleaned.mat'
label_csv_path = r'D:\W00Y0NG\PRGM2\2025BCI\current_experiments\DATA\processed\experiment_001_labels.csv'

dataset = EEGDataset(mat_path, label_csv_path)
dataset.remove_break()

eeg, labels, fs = dataset.get_data()

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
