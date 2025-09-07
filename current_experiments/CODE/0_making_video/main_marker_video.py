import os
import random
import cv2
import numpy as np

# ===== 설정 =====
stim_type = 'multi'  # 'text', 'sign', 'multi'
session_num = 3
image_dir = os.path.join('current_experiments', 'CODE', '0_making_video', 'StimuliImage')

video_output = os.path.join(
    'current_experiments',
    'DATA',
    f"stimulus_session{session_num}.mp4"
)

stimuli = ['hello', 'helpme', 'sorry', 'thanku']

# 시간 설정 (ms)
rest_ms = 4000
stim_ms = 4000
fixation_ms = 1000
imagine_ms = 3000

# ===== 화면/영상 설정 =====
screen_size = (1920, 1080)
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_output, fourcc, fps, screen_size)

cv2.namedWindow("Stimulus", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Stimulus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ===== 랜덤으로 단어 5개 뽑기 =====
trial_order = [random.choice(stimuli) for _ in range(5)]

for i, word in enumerate(trial_order, 1):
    print(f"--- Trial {i}: {word} ---")

    stim_img_path = os.path.join(image_dir, f'{stim_type}_{word}.PNG')
    fixation_path = os.path.join(image_dir, 'fixation.png')
    rest_path = os.path.join(image_dir, 'rest.png')

    # 1. 휴식
    rest_img = cv2.imread(rest_path)
    rest_img = cv2.resize(rest_img, screen_size)
    for _ in range(int(rest_ms/1000*fps)):
        cv2.imshow("Stimulus", rest_img)
        video_writer.write(rest_img)
        cv2.waitKey(int(1000/fps))

    # 2. 자극 제시
    stim_img = cv2.imread(stim_img_path)
    stim_img = cv2.resize(stim_img, screen_size)
    for _ in range(int(stim_ms/1000*fps)):
        cv2.imshow("Stimulus", stim_img)
        video_writer.write(stim_img)
        cv2.waitKey(int(1000/fps))

    # 3. 고정 십자
    fixation_img = cv2.imread(fixation_path)
    fixation_img = cv2.resize(fixation_img, screen_size)
    for _ in range(int(fixation_ms/1000*fps)):
        cv2.imshow("Stimulus", fixation_img)
        video_writer.write(fixation_img)
        cv2.waitKey(int(1000/fps))

    # 4. 상상 구간 (흰색 화면)
    white = np.ones((screen_size[1], screen_size[0], 3), dtype=np.uint8) * 255
    for _ in range(int(imagine_ms/1000*fps)):
        cv2.imshow("Stimulus", white)
        video_writer.write(white)
        cv2.waitKey(int(1000/fps))

# ===== 종료 =====
video_writer.release()
cv2.destroyAllWindows()
print("■ 영상 저장 완료:", video_output)
