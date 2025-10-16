import numpy as np
from scipy.io import savemat
import os

base_path = r'current_experiments\DATA\Final'

# npy 파일 로드
sign_eeg = np.load(os.path.join(base_path, 'eeg_sign_features.npy'))
sign_labels = np.load(os.path.join(base_path, 'eeg_sign_features_labels.npy'))

motor_eeg = np.load(os.path.join(base_path, 'eeg_motor_features.npy'))
motor_labels = np.load(os.path.join(base_path, 'eeg_motor_features_labels.npy'))

speech_eeg = np.load(os.path.join(base_path, 'eeg_speech_features.npy'))
speech_labels = np.load(os.path.join(base_path, 'eeg_speech_features_labels.npy'))

# MATLAB 형식으로 저장
savemat("eeg_modalities.mat", {
    "sign_eeg": sign_eeg,
    "sign_labels": sign_labels,
    "motor_eeg": motor_eeg,
    "motor_labels": motor_labels,
    "speech_eeg": speech_eeg,
    "speech_labels": speech_labels
})
