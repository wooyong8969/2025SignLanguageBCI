import scipy.io as sio
import numpy as np
import os

base_dir = r'current_experiments\DATA\processed\experiment_001'
sessions = [6, 7, 8]
mat_files = [os.path.join(base_dir, f'experiment_001({sid})_cleaned.mat') for sid in sessions]
label_files = [os.path.join(base_dir, f'experiment_001({sid})_labels.csv') for sid in sessions]

all_eeg = []
all_labels = []

for mat_path, label_path in zip(mat_files, label_files):
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    eeg_struct = mat['EEG_clean']
    eeg_data = eeg_struct.data.astype(np.float32)  # (epochs, samples, channels)
    all_eeg.append(eeg_data)
    
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    all_labels.extend(labels)

# (n_total_epochs, samples, channels)
concat_eeg = np.concatenate(all_eeg, axis=0)
concat_labels = np.array(all_labels)

print("EEG shape:", concat_eeg.shape)
print("Label shape:", concat_labels.shape)

sio.savemat(os.path.join(base_dir, 'experiment_001(6-8)_cleaned.mat'), {
    'EEG_clean': {
        'data': concat_eeg.astype(np.float32),
        'srate': np.array([[125.0]])
    }
})
with open(os.path.join(base_dir, 'experiment_001(6-8)_labels.csv'), 'w', encoding='utf-8') as f:
    for label in concat_labels:
        f.write(f"{label}\n")
