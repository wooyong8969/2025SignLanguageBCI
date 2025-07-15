import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist
from mne.filter import notch_filter
import mne
from eegadjust import art_comp
import scipy.io as sio
import matplotlib.pyplot as plt


class EEGPreprocessor:
    def __init__(self, csv_path, epoch_path, ch_names, sfreq=125, cut_time=10):
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

    # def apply_zscore(self):
    #     data = self.epochs.get_data()
    #     mean = np.mean(data, axis=2, keepdims=True)
    #     std = np.std(data, axis=2, keepdims=True) + 1e-8
    #     self.epochs._data = (data - mean) / std

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

def plot_epochs_data(epochs, title, epoch_idx=1, ch_idxs=None):
    """
    epochs: mne.Epochs 또는 np.ndarray (n_epochs, n_channels, n_times)
    epoch_idx: plot할 epoch index
    ch_idxs: plot할 채널 인덱스 리스트(예: [0, 1, 2, 3])
    """
    if hasattr(epochs, "get_data"):
        data = epochs.get_data()[epoch_idx]  # (n_channels, n_times)
        ch_names = epochs.info['ch_names']
    else:
        data = epochs[epoch_idx]
        ch_names = [f'Ch{i+1}' for i in range(data.shape[0])]

    times = np.arange(data.shape[1]) / 125  # sfreq=125Hz 기준, 수정 가능

    if ch_idxs is None:
        ch_idxs = range(min(16, data.shape[0]))  # 기본: 앞 8채널

    plt.figure(figsize=(12, 5))
    offset = 100
    # plt.plot(times, data[5, :] + 0 * offset, label=ch_names[5])
    for i, ch in enumerate(ch_idxs):
        plt.plot(times, data[ch, :] + i * offset, label=ch_names[ch])
    plt.title(f"{title} (epoch {epoch_idx})")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

def plot_long_concat(epochs, ch_idxs=None):
    """
    모든 epoch를 시간 순서대로 이어붙여 전체 신호 시계열 plot (각 채널별).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    # (n_epochs, n_channels, n_times) -> (n_channels, n_epochs * n_times)
    data_concat = data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
    times = np.arange(data_concat.shape[1]) / epochs.info['sfreq']
    if ch_idxs is None:
        ch_idxs = range(min(8, n_channels))
    plt.figure(figsize=(14, 6))
    for ch in ch_idxs:
        plt.plot(times, data_concat[ch, :], label=epochs.info['ch_names'][ch])
    plt.title("All epochs concatenated (channels as rows)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    csv_path = r'current_experiments\DATA\raw\experiment_001\SI_30(3).csv'
    epoch_table_path = r'current_experiments\DATA\video\experiment_001_30_epochs.xlsx'
    save_dir = r'current_experiments\DATA\processed\experiment_001'
    base_name = 'experiment_001(3)'

    selected_labels = ['FP1','FP2','C3','C4','P7','P8','O1','O2',
                       'F7','F8','F3','F4','T7','T8','P3','P4']
    
    pre = EEGPreprocessor(csv_path, epoch_table_path, selected_labels)
    plot_epochs_data(pre.epochs, "raw")
    pre.rereference()
    plot_epochs_data(pre.epochs, "ref") 
    pre.apply_bandpass()
    pre.apply_notch()
    plot_epochs_data(pre.epochs, "filter")
    pre.apply_ica()
    pre.apply_adjust()
    plot_epochs_data(pre.epochs, "artifact")
    # pre.apply_zscore()
    pre.save(save_dir, base_name)