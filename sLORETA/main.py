import mne
import os
from mne import read_source_estimate
from mne import read_labels_from_annot

# 저장 위치
save_dir = "sLORETA\\eeg_sign_epochs_30"
subjects_dir = os.path.expanduser("C:/Users/wooyo/mne_data/MNE-fsaverage-data")

# 좌반구 Broca & Wernicke, 운동 피질 이름 (Desikan-Killiany atlas 기준)
BROCA_LABELS = ['parsopercularis-lh', 'parstriangularis-lh']
WERNICKE_LABELS = ['superiortemporal-lh', 'supramarginal-lh']
MOTOR_LABELS = ['precentral-lh', 'precentral-rh']  # 운동 피질은 양쪽 다 표시

def load_and_plot(prefix):
    try:
        global label
        stc = read_source_estimate(
            os.path.join(save_dir, f"{prefix}-sLORETA"),
            subject='fsaverage'
        )
        print(f"[불러오기 성공] {prefix}-sLORETA")

        brain = stc.plot(
            subject='fsaverage',
            subjects_dir=subjects_dir,
            initial_time=0.1,
            hemi='both',
            surface='inflated',
            time_viewer=True
        )

        # 좌우반구 label 읽기
        labels_lh = read_labels_from_annot('fsaverage', parc='aparc', hemi='lh', subjects_dir=subjects_dir)
        labels_rh = read_labels_from_annot('fsaverage', parc='aparc', hemi='rh', subjects_dir=subjects_dir)
        all_labels = labels_lh + labels_rh

        # 색상 표시
        for lbl in all_labels:
            if lbl.name in BROCA_LABELS:
                brain.add_label(lbl, borders=True, color='red', alpha=1.0)
            elif lbl.name in WERNICKE_LABELS:
                brain.add_label(lbl, borders=True, color='blue', alpha=1.0)
            elif lbl.name in MOTOR_LABELS:
                brain.add_label(lbl, borders=True, color='green', alpha=1.0)

        brain.show()

    except Exception as e:
        print(f"[불러오기 실패] {prefix}: {e}")


if __name__ == "__main__":
    while True:
        label = input("불러올 label 번호를 입력하세요(q 종료): ").strip()
        if label == 'q':
            break
        prefix = f"eeg_multi_label_{label}"
        load_and_plot(prefix)

    print("\n[완료] label별, trial별 sLORETA 결과 불러오기가 종료되었습니다.")
