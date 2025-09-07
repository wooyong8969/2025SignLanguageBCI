import os
import random
import time
import cv2
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# ===== 설정 =====
stim_type = 'multi'  # 'text', 'sign', 'multi' 중 선택
session_num = 3     # 세션 번호 지정
image_dir = os.path.join('current_experiments', 'CODE', '0_making_video', 'StimuliImage')

output_file = os.path.join(
    'current_experiments',
    'DATA',
    f"eeg_{stim_type}_session{session_num}.csv"
)

stimuli = ['hello', 'helpme', 'sorry', 'thanku']
marker_map = {'hello': 1, 'helpme': 2, 'sorry': 3, 'thanku': 4}

# 시간 설정 (ms)
rest_ms = 4000       # 휴식
stim_ms = 4000       # 자극 제시
fixation_ms = 1000   # 고정 십자
imagine_ms = 3000    # 상상

# ===== BrainFlow 세션 설정 =====
params = BrainFlowInputParams()
params.serial_port = 'COM7'
board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
board.prepare_session()
board.start_stream(45000, f'file://{output_file}:w')
print("▶ EEG 기록 시작")

# ===== 화면 설정 =====
screen_size = (1920, 1080)
cv2.namedWindow("Stimulus", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Stimulus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ===== 자극 순서 생성 =====
trial_order = stimuli * 10
random.shuffle(trial_order)

for word in trial_order:
    stim_img_path = os.path.join(image_dir, f'{stim_type}_{word}.PNG')
    fixation_path = os.path.join(image_dir, 'fixation.png')
    rest_path = os.path.join(image_dir, 'rest.png')

    # 1. 휴식 이미지
    rest_img = cv2.imread(rest_path)
    rest_img = cv2.resize(rest_img, screen_size)
    cv2.imshow("Stimulus", rest_img)
    cv2.waitKey(rest_ms)

    # 2. 자극 제시 + 자극 마커 삽입
    stim_img = cv2.imread(stim_img_path)
    stim_img = cv2.resize(stim_img, screen_size)
    board.insert_marker(marker_map[word])  # 자극 마커 (1~4)
    cv2.imshow("Stimulus", stim_img)
    cv2.waitKey(stim_ms)

    # 3. 고정 십자
    fixation_img = cv2.imread(fixation_path)
    fixation_img = cv2.resize(fixation_img, screen_size)
    cv2.imshow("Stimulus", fixation_img)
    cv2.waitKey(fixation_ms)

    # 4. 상상 구간 + 상상 마커 삽입
    white = np.ones((screen_size[1], screen_size[0], 3), dtype=np.uint8) *255
    board.insert_marker(marker_map[word] + 100)  # 상상 마커 (101~104)
    cv2.imshow("Stimulus", white)
    cv2.waitKey(imagine_ms)

# ===== 종료 =====
board.stop_stream()
board.release_session()
cv2.destroyAllWindows()
print("■ EEG 기록 완료 및 저장됨")
