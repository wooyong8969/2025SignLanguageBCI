import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist
from mne.filter import notch_filter
import mne
from eegadjust import art_comp
import scipy.io as sio


class EEGPreprocessor:
    def __init__(self, csv_path, epoch_path, ch_names, sfreq=125, cut_time=0):
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.cut_time = cut_time
        self.raw_data = pd.read_csv(csv_path, sep='\t', header=None, engine='python').values
        self.epoch_table = pd.read_excel(epoch_path)
        self.epochs_data, self.labels = self._epoch_data()
        self.epochs = self._create_mne_epochs()
        self.brain_areas = self._define_brain_areas()
        self.ch_dist = self._compute_channel_distance_matrix()

    def _epoch_data(self):
        raw = self.raw_data
        fs = self.sfreq
        cut = int(self.cut_time * fs)
        data = raw[cut:, 1:17]
        offset = cut
        epoched = []
        labels = []
        for _, row in self.epoch_table.iterrows():
            start = int(row[0] * fs) - offset
            end = int(row[1] * fs) - offset
            if start < 0 or end > data.shape[0]:
                continue
            segment = data[start:end].T
            epoched.append(segment)
            labels.append(str(row[2]))
        return epoched, labels

    def _create_mne_epochs(self):
        data = np.stack(self.epochs_data, axis=0)
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
        info.rename_channels({name: name.upper() for name in info.ch_names})
        print("Epochs data shape:", data.shape)
        print("Number of selected labels:", len(self.ch_names))
        print("Number of channels in info:", len(info['ch_names']))
        epochs = mne.EpochsArray(data, info)
        events = np.column_stack((np.arange(len(self.labels)), np.zeros(len(self.labels), dtype=int), np.arange(len(self.labels))))
        epochs.events = events
        epochs.event_id = {label: i for i, label in enumerate(set(self.labels))}
        return epochs

    def _define_brain_areas(self):
        ch_index = {ch: i for i, ch in enumerate(self.ch_names)}
        return {
            'eeg': np.array(list(ch_index.values())),
            'frontal': np.array([ch_index[ch] for ch in ['FP1', 'FP2', 'F7', 'F8', 'F3', 'F4']]),
            'posterior': np.array([ch_index[ch] for ch in ['P3', 'P4', 'P7', 'P8', 'O1', 'O2']]),
            'left-eye': np.array([ch_index[ch] for ch in ['FP1', 'F7']]),
            'right-eye': np.array([ch_index[ch] for ch in ['FP2', 'F8']])
        }

    def _compute_channel_distance_matrix(self):
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.rename_channels({name: name.upper() for name in info.ch_names})
        montage.rename_channels({name: name.upper() for name in montage.ch_names})
        info.set_montage(montage, on_missing='warn')
        pos = np.array([info['chs'][i]['loc'][:3] for i in range(len(info.ch_names))])
        return cdist(pos, pos)

    def apply_bandpass(self, l_freq=0.5, h_freq=40.0):
        self.epochs = self.epochs.copy().filter(l_freq, h_freq, fir_design='firwin', filter_length='auto')

    def apply_notch(self, freq=60.0):
        data = self.epochs.get_data(copy=True)
        data = notch_filter(data, Fs=self.sfreq, freqs=[freq], notch_widths=1.0, verbose=True)
        self.epochs._data = data

    def apply_ica(self):
        self.ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto')  # ← 고정값 추천
        self.ica.fit(self.epochs)
        self.ica.get_sources(self.epochs).get_data()

    def apply_adjust(self):
        sources = self.ica.get_sources(self.epochs)
        sources_data = sources.get_data(copy=True)
        sources_data = np.transpose(sources_data, (1, 2, 0))

        mix_mat = self.ica.mixing_matrix_
        n_rows = mix_mat.shape[0]
        for k in self.brain_areas:
            self.brain_areas[k] = self.brain_areas[k][self.brain_areas[k] < n_rows]

        blink, vert, horz, disc = art_comp(sources_data, mix_mat, self.brain_areas, self.ch_dist)
        to_remove = np.where(blink | vert | horz | disc)[0]
        print("Removing components:", to_remove)
        self.ica.exclude = list(to_remove)
        self.epochs = self.ica.apply(self.epochs.copy())

    def apply_zscore(self):
        data = self.epochs.get_data()
        mean = np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True) + 1e-8
        self.epochs._data = (data - mean) / std

    def rereference(self):
        self.epochs = self.epochs.copy().set_eeg_reference('average', projection=False)

    def save(self, save_dir, base_name, save_mat=True):
        with open(f"{save_dir}/{base_name}_labels.csv", 'w') as f:
            for label in self.labels:
                f.write(f"{label}\n")
        if save_mat:
            data = self.epochs.get_data()  # (n_epochs, n_channels, n_times)
            data = np.transpose(data, (0, 2, 1))  # (epochs, samples, channels)
            srate = self.sfreq

            sio.savemat(f"{save_dir}/{base_name}_cleaned.mat", {
                'EEG_clean': {
                    'data': data.astype(np.float32),
                    'srate': np.array([[srate]])
                }
            })
            print(f".mat 파일 저장 완료 → {base_name}_cleaned.mat")


if __name__ == "__main__":
    csv_path = r'current_experiments\DATA\raw\experiment_001\SI_30(3).csv'
    epoch_table_path = r'current_experiments\DATA\video\experiment_001_30_epochs.xlsx'
    save_dir = r'current_experiments\DATA\processed\experiment_001'
    base_name = 'experiment_001(3)'

    selected_labels = ['FP1','FP2','C3','C4','P7','P8','O1','O2',
                       'F7','F8','F3','F4','T7','T8','P3','P4']
    
    pre = EEGPreprocessor(csv_path, epoch_table_path, selected_labels)
    pre.apply_zscore()
    pre.apply_bandpass()
    pre.apply_notch()
    pre.apply_ica()
    pre.apply_adjust()
    pre.rereference()
    pre.save(save_dir, base_name)