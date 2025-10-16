import os
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.transforms import Transform

# ----------------------- #
# 1. 데이터 경로 및 설정값
# ----------------------- #

# 사용자 지정
npy_data_path = "current_experiments\DATA\Final\eeg_speech_epochs.npy"
npy_label_path = "current_experiments\DATA\Final\eeg_speech_labels.npy"
label_number = "101"  # 이건 사용자가 설정 (ex. 101, 102 ...)
save_dir = "sLORETA/eeg_speech_epochs_30"
subjects_dir = "C:/Users/wooyo/mne_data/MNE-fsaverage-data"
subject = "fsaverage"
sfreq = 125

# 채널 이름 (총 16채널)
ch_names = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2',
            'F7','F8','F3','F4','T7','T8','P3','P4']

# ---------------------- #
# 2. 데이터 불러오기 및 Epochs 생성
# ---------------------- #

# npy 데이터 로드
eeg_data = np.load(npy_data_path)  # shape: (n_epochs, n_times, n_channels)
labels = np.load(npy_label_path)

# MNE가 요구하는 shape: (n_epochs, n_channels, n_times)
eeg_data = np.transpose(eeg_data, (0, 2, 1))

# Info 객체 생성
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
epochs = mne.EpochsArray(eeg_data, info)
epochs.set_montage(mne.channels.make_standard_montage('standard_1020'))
epochs.set_eeg_reference(projection=True)

# -------------------------- #
# 3. 평균 Evoked 생성 및 저장
# -------------------------- #

prefix = f"eeg_multi_label_{label_number}"
os.makedirs(save_dir, exist_ok=True)

evoked = epochs.average()
evoked.save(os.path.join(save_dir, f"{prefix}-ave.fif"))

# ------------------------------- #
# 4. 공분산 계산 및 저장
# ------------------------------- #

noise_cov = mne.compute_covariance(epochs, method='auto')
mne.write_cov(os.path.join(save_dir, f"{prefix}-cov.fif"), noise_cov)

# ------------------------------------- #
# 5. Forward 모델 생성 및 저장
# ------------------------------------- #

trans = Transform('head', 'mri')
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_solution(mne.make_bem_model(subject=subject, ico=4, subjects_dir=subjects_dir))

fwd = mne.make_forward_solution(
    evoked.info, trans=trans, src=src, bem=bem,
    eeg=True, mindist=5.0, n_jobs=1
)
mne.write_forward_solution(os.path.join(save_dir, f"{prefix}-fwd.fif"), fwd, overwrite=True)

# ------------------------------------- #
# 6. Inverse operator 생성 및 저장
# ------------------------------------- #

inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8)
mne.minimum_norm.write_inverse_operator(
    os.path.join(save_dir, f"{prefix}-inv.fif"), inverse_operator
)

# ------------------------------------- #
# 7. sLORETA 계산 및 .stc 저장
# ------------------------------------- #

stc = apply_inverse(evoked, inverse_operator, lambda2=1. / 9., method='sLORETA')
stc.save(os.path.join(save_dir, f"{prefix}-sLORETA"))  # → 자동으로 -lh.stc / -rh.stc 생성됨

print(f"\n✅ 완료: '{prefix}'에 대한 sLORETA 결과 저장됨.")
