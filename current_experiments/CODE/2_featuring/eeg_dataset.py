import numpy as np
import pandas as pd
import h5py

class EEGDataset:
    def __init__(self, mat_path, label_csv_path):
        self.eeg, self.fs = self._load_mat(mat_path)
        self.labels = self._load_labels(label_csv_path)
        print(f"EEG: {self.eeg.shape}, sf: {self.fs}, Labels: {len(self.labels)})")

    def _load_mat(self, path):
        with h5py.File(path, 'r') as f:
            data = f['EEG_clean']['data']
            eeg = np.array(data).astype(np.float32)
            eeg = np.transpose(eeg, (0, 2, 1))  # epochs, channels, samples
            fs = f['EEG_clean']['srate'][0][0]
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
