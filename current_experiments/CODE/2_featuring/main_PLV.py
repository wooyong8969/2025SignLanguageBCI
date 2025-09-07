# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mne
from scipy.signal import butter, filtfilt, hilbert
from matplotlib.colors import LinearSegmentedColormap

white_to_blue = LinearSegmentedColormap.from_list('white_to_blue', ["#d2e3ff", '#005fce'])

# =========================
# 설정
# =========================
TARGET_CHS = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2',
              'F7','F8','F3','F4','T7','T8','P3','P4']

FIF_PATH  = r"PLV\eeg_speech_preprocessed_baseline_cleaned.fif"
LOCS_PATH = r"PLV\Standard-10-20-Cap81.locs"

BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta' : (13, 30),
    'Gamma': (30, 40),
}

THRESH    = 0.6
TOPK_PCT  = None

# =========================
# 유틸
# =========================
def bandpass(sig, sfreq, low, high, order=4):
    b, a = butter(order, [low/(sfreq/2), high/(sfreq/2)], btype='band')
    return filtfilt(b, a, sig)

def load_eeg_flex(path):
    try:
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        events = mne.find_events(raw, verbose=False)
        if len(events) == 0:
            X = raw.get_data()
            sf = raw.info['sfreq']
            chs = raw.ch_names
            labels = np.array([0])
            X = X.T.reshape(1, X.shape[0], X.shape[1])
        else:
            event_id = {str(e): e for e in np.unique(events[:, 2])}
            epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=2.0,
                                baseline=None, preload=True, verbose=False)
            X = epochs.get_data()
            sf = epochs.info['sfreq']
            chs = epochs.ch_names
            labels = epochs.events[:, 2]
        return X, sf, chs, labels
    except ValueError:
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        return epochs.get_data(), epochs.info['sfreq'], epochs.ch_names, epochs.events[:, 2]

def compute_plv_matrix(data, sfreq, band=(8,13)):
    T, C, N = data.shape
    low, high = band
    phase = np.empty((T, C, N), float)
    for t in range(T):
        for c in range(C):
            filt = bandpass(data[t, c], sfreq, low, high)
            phase[t, c] = np.angle(hilbert(filt))
    plv = np.ones((C, C), float)
    for i in range(C):
        for j in range(i+1, C):
            dphi = np.angle(np.exp(1j * (phase[:, i, :] - phase[:, j, :])))
            plv[i, j] = plv[j, i] = float(np.abs(np.mean(np.exp(1j * dphi))))
    return plv

def for_plv_name(name: str) -> str:
    return {'Fp1': 'FP1', 'Fp2': 'FP2'}.get(name, name)

def montage_xy_for(info_names, montage):
    ch_pos = montage.get_positions()["ch_pos"]
    XY = []
    for nm in info_names:
        key = nm if nm in ch_pos else next((k for k in ch_pos if k.lower()==nm.lower()), None)
        if key is None: XY.append([0.0, 0.0])
        else:
            x, y, _ = ch_pos[key]; XY.append([x, y])
    XY = np.asarray(XY, float)
    XY -= XY.mean(axis=0, keepdims=True)
    return XY

def match_pos_to_ax(ax, pos_xy, margin=0.96):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    r_ax = min(xlim[1]-xlim[0], ylim[1]-ylim[0]) / 2.0 / 1.25
    r_pos = np.sqrt((pos_xy**2).sum(axis=1)).max()
    if r_pos == 0:
        return pos_xy
    return pos_xy * (r_ax / r_pos) * margin

# =========================
# 데이터 및 채널 매핑
# =========================
X, sfreq, all_chs, labels = load_eeg_flex(FIF_PATH)
montage = mne.channels.read_custom_montage(LOCS_PATH)

chs_for_info = [ch for ch in TARGET_CHS if any(ch.lower()==m.lower() for m in montage.ch_names)]
info = mne.create_info(chs_for_info, sfreq=sfreq, ch_types='eeg')
info.set_montage(montage)
C_plot = len(info.ch_names)
if C_plot < 2:
    raise RuntimeError("시각화 가능한 채널이 2개 미만입니다.")

pick_plv = []
ch_names_plv = []
for ch in info.ch_names:
    ch_plv = for_plv_name(ch)
    idx = None
    if ch_plv in all_chs:
        idx = all_chs.index(ch_plv)
    else:
        for k, nm in enumerate(all_chs):
            if nm.lower() == ch_plv.lower():
                idx = k; break
    if idx is not None:
        pick_plv.append(idx)
        ch_names_plv.append(all_chs[idx])
    else:
        print(f"[경고] 데이터에 채널 없음(PLV 제외): {ch} (치환:{ch_plv})")

if len(pick_plv) < 2:
    raise RuntimeError("PLV 계산에 사용할 채널이 2개 미만입니다.")

plv_index = {nm.lower(): i for i, nm in enumerate(ch_names_plv)}
def idx_plv(name_plot: str):
    i = plv_index.get(name_plot.lower())
    if i is not None: return i
    return plv_index.get(for_plv_name(name_plot).lower(), None)

# =========================
# 모드 선택 및 시각화
# =========================
mode = input("시각화 모드 선택 (1: 라벨별, 2: 전체 평균): ").strip()
while mode not in ['1', '2']:
    mode = input("잘못된 입력입니다. 1 또는 2를 입력해주세요: ").strip()

band_items = list(BANDS.items())

if mode == '1':
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        trial_mask = (labels == lbl)
        X_sel = X[:, pick_plv, :][trial_mask]
        if X_sel.shape[0] == 0:
            print(f"[참고] 라벨 {lbl} 에 해당하는 trial 없음. 건너뜀.")
            continue

        fig, axes = plt.subplots(1, len(band_items), figsize=(5*len(band_items), 6.2), squeeze=False)
        axes = axes[0]

        for col, (band_name, band_rng) in enumerate(band_items):
            ax = axes[col]
            data_topo = np.zeros(C_plot)
            mne.viz.plot_topomap(data_topo, info, axes=ax, outlines='head',
                                 contours=0, sensors=False, cmap='Greys', show=False)
            for im in ax.get_images(): im.set_alpha(0.35)

            pos = montage_xy_for(info.ch_names, montage)
            pos = match_pos_to_ax(ax, pos, margin=0.96)

            plv_mat_plvspace = compute_plv_matrix(X_sel, sfreq, band=band_rng)

            plv_mat = np.zeros((C_plot, C_plot), float); np.fill_diagonal(plv_mat, 1.0)
            for i, ni in enumerate(info.ch_names):
                ii = idx_plv(ni)
                if ii is None: continue
                for j, nj in enumerate(info.ch_names):
                    jj = idx_plv(nj)
                    if jj is None: continue
                    plv_mat[i, j] = plv_mat_plvspace[ii, jj]

            thr = THRESH
            if TOPK_PCT is not None:
                vals = plv_mat[np.triu_indices_from(plv_mat, k=1)]
                cut = np.quantile(vals, 1 - TOPK_PCT)
                thr = max(THRESH, cut)

            cmap = white_to_blue
            denom = (1.0 - thr + 1e-12)
            for i in range(C_plot):
                for j in range(i+1, C_plot):
                    w = plv_mat[i, j]
                    if w >= thr:
                        x1, y1 = pos[i]; x2, y2 = pos[j]
                        ax.plot([x1, x2], [y1, y2],
                                color=cmap(w),
                                lw=1 + 4.0*(w - thr)/denom,
                                alpha=0.25 + 0.75*(w - thr)/denom,
                                zorder=2)

            A = np.triu(plv_mat, 1); mask = (A >= thr)
            strength = (mask*A) + (mask*A).T
            score = strength.sum(axis=1)
            sizes = (50 + (180-50)*(score/score.max())) if score.max()>0 else np.full(C_plot, 50)

            # ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c='white', edgecolors='k', zorder=3)
            for (x, y), name in zip(pos, info.ch_names):
                ax.text(x, y, name, ha='center', va='center', fontsize=9, zorder=4)

            ax.set_aspect('equal'); ax.axis('off')
            # ax.set_title(f"Label {lbl} — {band_name} ({band_rng[0]}–{band_rng[1]} Hz)\nthr≥{THRESH}"
            #              + (f", top {int(TOPK_PCT*100)}%" if TOPK_PCT else ""))

            norm = mpl.colors.Normalize(vmin=THRESH, vmax=1.0)
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("PLV")

        save_path = f"plv_topomap_speech_label{lbl}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"저장 완료: {save_path}")

elif mode == '2':
    X_sel = X[:, pick_plv, :]

    fig, axes = plt.subplots(1, len(band_items), figsize=(5*len(band_items), 6.2), squeeze=False)
    axes = axes[0]

    for col, (band_name, band_rng) in enumerate(band_items):
        ax = axes[col]
        data_topo = np.zeros(C_plot)
        mne.viz.plot_topomap(data_topo, info, axes=ax, outlines='head',
                             contours=0, sensors=False, cmap='Greys', show=False)
        for im in ax.get_images(): im.set_alpha(0.35)

        pos = montage_xy_for(info.ch_names, montage)
        pos = match_pos_to_ax(ax, pos, margin=0.96)

        plv_mat_plvspace = compute_plv_matrix(X_sel, sfreq, band=band_rng)

        plv_mat = np.zeros((C_plot, C_plot), float); np.fill_diagonal(plv_mat, 1.0)
        for i, ni in enumerate(info.ch_names):
            ii = idx_plv(ni)
            if ii is None: continue
            for j, nj in enumerate(info.ch_names):
                jj = idx_plv(nj)
                if jj is None: continue
                plv_mat[i, j] = plv_mat_plvspace[ii, jj]

        thr = THRESH
        if TOPK_PCT is not None:
            vals = plv_mat[np.triu_indices_from(plv_mat, k=1)]
            cut = np.quantile(vals, 1 - TOPK_PCT)
            thr = max(THRESH, cut)

        cmap = white_to_blue
        denom = (1.0 - thr + 1e-12)
        for i in range(C_plot):
            for j in range(i+1, C_plot):
                w = plv_mat[i, j]
                if w >= thr:
                    x1, y1 = pos[i]; x2, y2 = pos[j]
                    ax.plot([x1, x2], [y1, y2],
                            color=cmap(w),
                            lw=2 + 4.0*(w - thr)/denom,
                            alpha=0.25 + 0.75*(w - thr)/denom,
                            zorder=2)

        A = np.triu(plv_mat, 1); mask = (A >= thr)
        strength = (mask*A) + (mask*A).T
        score = strength.sum(axis=1)
        sizes = (50 + (180-50)*(score/score.max())) if score.max()>0 else np.full(C_plot, 50)

        # ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c='white', edgecolors='k', zorder=3)
        for (x, y), name in zip(pos, info.ch_names):
            ax.text(x, y, name, ha='center', va='center', fontsize=9, zorder=4)

        ax.set_aspect('equal'); ax.axis('off')
        # ax.set_title(f"all mean — {band_name} ({band_rng[0]}–{band_rng[1]} Hz)\nthr≥{THRESH}"
        #              + (f", top {int(TOPK_PCT*100)}%" if TOPK_PCT else ""))

        norm = mpl.colors.Normalize(vmin=THRESH, vmax=1.0)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("PLV")

    save_path = f"plv_topomap_speech_ALL.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"저장 완료: {save_path}")
