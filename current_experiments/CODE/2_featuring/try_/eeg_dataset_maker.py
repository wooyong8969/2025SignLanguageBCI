import numpy as np
import pandas as pd
import h5py
import scipy.io as sio

class EEGDataset:
    def __init__(self, mat_path, label_csv_path):
        self.eeg, self.fs = self._load_mat(mat_path)
        self.labels = self._load_labels(label_csv_path)
        print(f"EEG: {self.eeg.shape}, sf: {self.fs}, Labels: {len(self.labels)})")

<<<<<<<< HEAD:current_experiments/CODE/2_featuring/try_multi_CSP/eeg_dataset_maker.py
    # def _load_mat(self, path):
    #     with h5py.File(path, 'r') as f:
    #         data = f['EEG_clean']['data']
    #         eeg = np.array(data).astype(np.float32)
    #         eeg = np.transpose(eeg, (0, 2, 1))  # epochs, channels, samples
    #         fs = f['EEG_clean']['srate'][0][0]
    #     return eeg, fs
========
    def x_load_mat(self, path):
        with h5py.File(path, 'r') as f:
            data = f['EEG_clean']['data']
            eeg = np.array(data).astype(np.float32)
            eeg = np.transpose(eeg, (0, 2, 1))  # epochs, channels, samples
            fs = f['EEG_clean']['srate'][0][0]
        return eeg, fs
>>>>>>>> a777d4ab37b3bb897ad7493349fd8e91532f2ab0:current_experiments/CODE/2_featuring/try_/eeg_dataset_maker.py

    def _load_mat(self, path):
        mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        eeg_struct = mat['EEG_clean']
        eeg = eeg_struct.data.astype(np.float32)           # shape: (epochs, samples, channels)
        eeg = np.transpose(eeg, (0, 2, 1))                  # → (epochs, channels, samples)
        fs = float(eeg_struct.srate)
        return eeg, fs

    def _load_labels(self, path):
        labels = pd.read_csv(path, header=None).iloc[:, 0].tolist()
        return labels

    def remove_break(self):
        mask = [lbl != 'Break' for lbl in self.labels]
        self.eeg = self.eeg[mask]
        self.labels = [lbl for lbl in self.labels if lbl != 'Break']
        print(f"Break 제거 후 EEG: {self.eeg.shape}, Label 수: {len(self.labels)}")

    def get_data(self):
        return self.eeg, self.labels, self.fs
