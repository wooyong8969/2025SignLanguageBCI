import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import StandardScaler

class DWTFeatureExtractor:
    def __init__(self, fs=125, wavelet='coif5', level=3):  # level 5 -> 3 (과다 분해 방지)
        self.fs = fs
        self.wavelet = wavelet
        self.level = level
        self.bands = ['cA3', 'cD3', 'cD2', 'cD1']  # 적절한 대역 수로 제한

    def extract(self, eeg_data):
        n_epochs, n_channels, _ = eeg_data.shape
        time_features = []
        freq_features = []

        for ep in range(n_epochs):
            epoch_time = []
            epoch_freq = []
            for ch in range(n_channels):
                signal = eeg_data[ep, ch, :]
                coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.level)
                coeffs_dict = dict(zip(self.bands, coeffs[:len(self.bands)]))

                for band in self.bands:
                    x = coeffs_dict[band]
                    epoch_time.extend(self._extract_time_features(x))
                    epoch_freq.extend(self._extract_freq_features(x))

            time_features.append(epoch_time)
            freq_features.append(epoch_freq)

        return np.array(time_features), np.array(freq_features)

    def _extract_time_features(self, x):
        rms = np.sqrt(np.mean(x ** 2))
        var = np.var(x)
        aav = np.mean(np.abs(x))
        return [rms, var, aav]  # 과도한 고차 통계 제거 (skew, kurtosis 제거)

    def _extract_freq_features(self, x):
        energy = np.sum(x ** 2)
        std = np.std(x)
        prob = np.abs(x) / np.sum(np.abs(x)) if np.sum(np.abs(x)) != 0 else np.ones_like(x) / len(x)
        ent = entropy(prob)
        return [energy, std, ent]

    def extract_csp_features(self, eeg_data, labels, n_components=4):
        eeg_data = eeg_data.astype(np.float64)

        # 스케일 정규화
        n_epochs, n_channels, n_times = eeg_data.shape
        reshaped = eeg_data.reshape(n_epochs * n_channels, n_times)
        scaled = StandardScaler().fit_transform(reshaped)
        eeg_data = scaled.reshape(n_epochs, n_channels, n_times)

        # 안정적인 공분산 추정 옵션 설정
        csp = CSP(n_components=n_components, log=True, cov_est='epoch')

        # fit_transform 수행
        return csp.fit_transform(eeg_data, labels)


    def extract_riemannian_logvar(self, eeg_data):
        covs = Covariances(estimator='oas').transform(eeg_data)
        log_diag = np.log(np.diagonal(covs, axis1=1, axis2=2) + 1e-10)  # log-varience
        return log_diag
