from joblib import load

# LabelEncoder 로드
le = load('label_encoder.joblib')

# 클래스 목록 확인
print("LabelEncoder가 학습한 클래스 목록:")
print(le.classes_)

# 총 클래스 수
print(f"총 클래스 수: {len(le.classes_)}")

# 인코딩 예시 (문자 → 숫자)
sample_labels = ['Hello', 'Help me', 'Sorry', 'Thank you']
print("Encoding 예시:", le.transform(sample_labels))

# 디코딩 예시 (숫자 → 문자)
sample_indices = [0, 1, 2, 3]
print("Decoding 예시:", le.inverse_transform(sample_indices))
