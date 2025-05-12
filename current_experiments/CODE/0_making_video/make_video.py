import cv2
import numpy as np
import random
import os
import pandas as pd
# pip install openpyxl

# 파라미터 설정
screen_size = (1920, 1080)
fps = 10
text_duration = 3  # 이미지 표시 시간 (초)
black_min = 3  # 검정 화면 최소 시간 (초)
black_max = 3  # 검정 화면 최대 시간 (초)
output_file = r"current_experiments\DATA\video\experiment_001_10_video.mp4"
excel_output = r"current_experiments\DATA\video\experiment_001_10_epochs.xlsx"

# 이미지 파일 경로
image_folder = r"D:\W00Y0NG\PRGM2\2025BCI\current_experiments\CODE\0_making_video\Image"
image_files = {
    "Hello": os.path.join(image_folder, "hello.png"),
    "Thank you": os.path.join(image_folder, "thanku.png"),
    "Sorry": os.path.join(image_folder, "sorry.png"),
    "Help me": os.path.join(image_folder, "helpme.png")
}

# 각 이미지 30번씩 등장, 순서 랜덤-
stimuli_list = list(image_files.items()) * 10
random.shuffle(stimuli_list)

# VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, screen_size)

# 실험 데이터 기록
epoch_data = []
total_time = 0

# 프레임 생성 함수
def create_image_frame(label, image_path, duration):
    global total_time
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (screen_size[0], screen_size[1]))
    
    start_time = total_time
    for _ in range(duration * fps):
        out.write(img_resized)
    total_time += duration
    
    epoch_data.append([start_time, total_time, label])

# 검정 화면 생성 함수
def create_black_frame(duration):
    global total_time
    start_time = total_time
    for _ in range(duration * fps):
        frame = np.zeros((screen_size[1], screen_size[0], 3), dtype=np.uint8)

        cv2.line(frame, (960 - 50, 540), (960 + 50, 540), (0, 0, 255), 5)
        cv2.line(frame, (960, 540 - 50), (960, 540 + 50), (0, 0, 255), 5)
        
        out.write(frame)
    total_time += duration
    epoch_data.append([start_time, total_time, "Break"])

# 실험 영상 생성
for label, image_path in stimuli_list:
    create_image_frame(label, image_path, text_duration)
    create_black_frame(random.randint(black_min, black_max))

out.release()
cv2.destroyAllWindows()

columns = ["start", "end", "class"]
df = pd.DataFrame(epoch_data, columns=columns)
df.to_excel(excel_output, index=False)

print(f"실험 영상 생성: {output_file}")
print(f"에포킹 정보 저장: {excel_output}")