import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy

class DWTFeatureExtractor:
    def __init__(self, fs=125, wavelet='coif5', level=5):
        self.fs = fs
        self.wavelet = wavelet
        self.level = level
        self.bands = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']

    def extract(self, eeg_data):
        n_epochs, n_channels, n_samples = eeg_data.shape
        time_features = {band: [] for band in self.bands}
        freq_features = {band: [] for band in self.bands}

        for ep in range(n_epochs):
            for ch in range(n_channels):
                signal = eeg_data[ep, ch, :]
                coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.level)
                coeffs_dict = dict(zip(self.bands, coeffs))

                for band in self.bands:
                    x = coeffs_dict[band]
                    time_feat = self._extract_time_features(x)
                    freq_feat = self._extract_freq_features(x)
                    time_features[band].append(time_feat)
                    freq_features[band].append(freq_feat)

        for band in self.bands:
            time_features[band] = np.array(time_features[band])
            freq_features[band] = np.array(freq_features[band])

        return time_features, freq_features
    
    # 시간/주파수 특징 평탄화 함수
    def flatten_feature_dict(feature_dict):
        flat_list = []
        for band in feature_dict:
            flat_list.append(feature_dict[band])
        return np.concatenate(flat_list, axis=1)  # (epochs, features)


    def _extract_time_features(self, x):
        rms = np.sqrt(np.mean(x ** 2))
        var = np.var(x)
        sk = skew(x)
        kurt_ = kurtosis(x)
        aav = np.mean(np.abs(x))
        return [rms, var, sk, kurt_, aav]

    def _extract_freq_features(self, x):
        energy = np.sum(x ** 2)
        prob = np.abs(x) / np.sum(np.abs(x)) if np.sum(np.abs(x)) != 0 else np.ones_like(x) / len(x)
        ent = entropy(prob)
        std = np.std(x)
        return [energy, ent, std]
