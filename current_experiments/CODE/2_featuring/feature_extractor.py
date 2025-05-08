import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP

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

    def flatten_feature_dict(self, feature_dict, bands):
        all_features = []
        for band in bands:
            f = feature_dict[band]
            n_epochs = len(f) // 16
            f = f.reshape(n_epochs, 16, f.shape[-1])
            f = f.reshape(n_epochs, -1)
            all_features.append(f)
        return np.concatenate(all_features, axis=1)


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

    def extract_csp_features(self, eeg_data, labels, n_components=4):
        eeg_data = eeg_data.astype(np.float64)
        le = LabelEncoder()
        y = le.fit_transform(labels)
        classes = np.unique(y)

        csp_features = []
        for c in classes:
            y_binary = (y == c).astype(int)
            csp = CSP(n_components=n_components, log=True)
            X_csp = csp.fit_transform(eeg_data, y_binary)
            csp_features.append(X_csp)

        return np.concatenate(csp_features, axis=1)

