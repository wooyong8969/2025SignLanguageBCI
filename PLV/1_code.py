import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import matplotlib as mpl

# =========================
# 설정
# =========================
TARGET_CHS = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2',
              'F7','F8','F3','F4','T7','T8','P3','P4']

FIF_PATH   = r"PLV\eeg_sign_preprocessed_baseline_cleaned.fif"
LOCS_PATH  = r"PLV\Standard-10-20-Cap81.locs"

BAND       = (8, 13)      # Alpha
THRESHOLD  = 0.70         # 간선 임계값
TARGET_R   = 0.48         # 두피 원(0.5) 안쪽 배치 스케일

# =========================
# 유틸 함수
# =========================
def bandpass(sig, sfreq, low, high, order=4):
    b, a = butter(order, [low/(sfreq/2), high/(sfreq/2)], btype='band')
    return filtfilt(b, a, sig)

def load_eeg_flex(path):
    """FIF 또는 Epochs에서 (trials, ch, time), sfreq, ch_names, labels 반환."""
    try:
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        events = mne.find_events(raw, verbose=False)
        if len(events) == 0:
            X = raw.get_data()                  # (ch, time)
            sf = raw.info['sfreq']
            chs = raw.ch_names
            labels = np.array([0])
            X = X.T.reshape(1, X.shape[0], X.shape[1])  # (1,ch,time)
        else:
            event_id = {str(e): e for e in np.unique(events[:, 2])}
            epochs = mne.Epochs(raw, events, event_id,
                                tmin=-0.5, tmax=2.0, baseline=None,
                                preload=True, verbose=False)
            X = epochs.get_data()               # (trials,ch,time)
            sf = epochs.info['sfreq']
            chs = epochs.ch_names
            labels = epochs.events[:, 2]
        return X, sf, chs, labels
    except ValueError:
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        return epochs.get_data(), epochs.info['sfreq'], epochs.ch_names, epochs.events[:, 2]

def montage_to_pos2d(ch_names, montage, target_radius=0.48):
    """montage 3D -> 2D(top-view), 중심정렬 + 최대반경=target_radius로 스케일."""
    ch_pos = montage.get_positions()["ch_pos"]  # {name:(x,y,z)}
    XY = []
    for name in ch_names:
        key = name if name in ch_pos else next((k for k in ch_pos if k.lower()==name.lower()), None)
        if key is None:
            XY.append([0.0, 0.0])
        else:
            x, y, _ = ch_pos[key]; XY.append([x, y])
    XY = np.asarray(XY, float)
    XY -= XY.mean(axis=0, keepdims=True)
    r = np.sqrt((XY**2).sum(axis=1)); rmax = float(r.max()) if r.size else 1.0
    if rmax > 0: XY *= (target_radius / rmax)
    return XY  # (n,2)

def compute_plv_matrix(data, sfreq, band=(8,13)):
    """data: (T,C,N) -> (C,C) PLV."""
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

# =========================
# 데이터/몽타주 로드
# =========================
X, sfreq, all_chs, labels = load_eeg_flex(FIF_PATH)
montage = mne.channels.read_custom_montage(LOCS_PATH)

# =========================
# 시각화 채널(좌표/라벨)은 montage 기준 원래 이름 유지
# =========================
present_for_plot = []
for ch in TARGET_CHS:
    if any(ch.lower()==m.lower() for m in montage.ch_names):
        present_for_plot.append(ch)
missing_plot = [ch for ch in TARGET_CHS if ch not in present_for_plot]
if missing_plot:
    print(f"[참고] 몽타주에 없어 좌표에서 제외된 채널: {missing_plot}")

# 시각화용 채널 이름 / 좌표
ch_names_plot = present_for_plot
pos = montage_to_pos2d(ch_names_plot, montage, target_radius=TARGET_R)  # (C_plot, 2)
C_plot = len(ch_names_plot)
if C_plot < 2:
    raise RuntimeError("시각화 가능한 채널이 2개 미만입니다.")

# =========================
# PLV 계산 채널 선택 (여기서만 Fp1/Fp2 → FP1/FP2로 치환)
# =========================
def for_plv_name(name: str) -> str:
    return {'Fp1':'FP1', 'Fp2':'FP2'}.get(name, name)

# 데이터 채널 인덱스(PLV용): 치환 후 정확히 매칭
pick_plv = []
ch_names_plv = []  # PLV 계산에 실제 사용한 채널명(데이터 기준)
for ch in ch_names_plot:
    ch_plv = for_plv_name(ch)  # Fp1/Fp2만 FP1/FP2로
    # exact match 우선, 없으면 대소문자 무시 매칭
    idx = None
    if ch_plv in all_chs:
        idx = all_chs.index(ch_plv)
    else:
        for k, nm in enumerate(all_chs):
            if nm.lower() == ch_plv.lower():
                idx = k; break
    if idx is not None:
        pick_plv.append(idx)
        ch_names_plv.append(all_chs[idx])  # 실제 데이터 이름 기록
    else:
        print(f"[경고] 데이터에 채널 없음(PLV 제외): {ch} (치환:{ch_plv})")

# 실제 PLV에 사용할 채널만 추림
if len(pick_plv) < 2:
    raise RuntimeError("PLV 계산에 사용할 채널이 2개 미만입니다.")
X_sel = X[:, pick_plv, :]                  # (T, C_plv, N)
C_plv = len(pick_plv)

# =========================
# PLV 계산 (실제 값)
# =========================
plv_mat_plvspace = compute_plv_matrix(X_sel, sfreq, band=BAND)  # (C_plv,C_plv)

# =========================
# PLV 행렬을 시각화 채널 순서로 재배치
#  - ch_names_plot (좌표/라벨) ↔ ch_names_plv (데이터 실제명) 맵핑
#  - 기본적으로 같은 채널(대소문자 무시/FP 치환) 찾아 삽입, 없으면 0
# =========================
# 데이터 실제명(lower) 세트
plv_name_map = {nm.lower(): i for i, nm in enumerate(ch_names_plv)}

def lookup_plv_index(plot_name: str):
    # plot_name 자체(lower)
    key = plot_name.lower()
    if key in plv_name_map:
        return plv_name_map[key]
    # Fp1/Fp2 → FP1/FP2 치환 후 재검색
    key2 = for_plv_name(plot_name).lower()
    return plv_name_map.get(key2, None)

# 최종 시각화용 PLV 행렬 (C_plot,C_plot)
plv_mat = np.zeros((C_plot, C_plot), float)
np.fill_diagonal(plv_mat, 1.0)
for i, ch_i in enumerate(ch_names_plot):
    ii = lookup_plv_index(ch_i)
    if ii is None:  # 데이터에 없음
        continue
    for j, ch_j in enumerate(ch_names_plot):
        jj = lookup_plv_index(ch_j)
        if jj is None:
            continue
        plv_mat[i, j] = plv_mat_plvspace[ii, jj]

# =========================
# Topomap 배경 값 (원하면 다른 feature로 교체)
# =========================
data_topo = np.random.rand(C_plot)

# =========================
# 시각화 (pos를 plot_topomap에 직접 전달해 좌표 일치)
# =========================
fig, ax = plt.subplots(figsize=(7, 7))
topomap_ax = fig.add_axes([0.25, 0.25, 0.5, 0.5])
mne.viz.plot_topomap(data_topo, pos, axes=ax, outlines='head',
                     contours=0, sensors=False, cmap='Blues', show=False)
for im in ax.get_images():
    im.set_alpha(0.35)

cmap = plt.cm.viridis
denom = (1.0 - THRESHOLD + 1e-12)

# 간선
for i in range(C_plot):
    for j in range(i+1, C_plot):
        w = plv_mat[i, j]
        if w >= THRESHOLD:
            x1, y1 = pos[i]; x2, y2 = pos[j]
            ax.plot([x1, x2], [y1, y2],
                    color=cmap(w),
                    lw=0.5 + 4.0*(w-THRESHOLD)/denom,
                    alpha=0.25 + 0.75*(w-THRESHOLD)/denom,
                    zorder=2)

# 노드 크기: 임계 이상 간선 가중치 합
A = np.triu(plv_mat, 1); mask = (A >= THRESHOLD)
strength = (mask*A) + (mask*A).T
score = strength.sum(axis=1)
sizes = (50 + (180-50)*(score/score.max())) if score.max()>0 else np.full(C_plot, 50)

ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c='white', edgecolors='k', zorder=3)
for (x, y), name in zip(pos, ch_names_plot):
    ax.text(x, y, name, ha='center', va='center', fontsize=9, zorder=4)

norm = mpl.colors.Normalize(vmin=THRESHOLD, vmax=1.0)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.06); cbar.set_label("PLV")

ax.set_aspect('equal'); ax.axis('off')
plt.title(f"PLV Network (band {BAND[0]}–{BAND[1]} Hz, thr≥{THRESHOLD})")
plt.tight_layout(); plt.show()
