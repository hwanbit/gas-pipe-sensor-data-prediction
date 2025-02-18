import numpy as np
import matplotlib.pyplot as plt
import pickle

# pkl 파일 목록
history_files = [
    'training_history_3sec.pkl',
    'training_history_6sec.pkl',
    'training_history_10sec.pkl',
    'training_history_30sec.pkl',
    'training_history_60sec.pkl',
    'training_history_120sec.pkl'
]

# 모델 이름 리스트 (그래프에 사용할 라벨)
model_names = ['3s', '6s', '10s', '30s', '60s', '120s']

# 정확도와 표준편차를 저장할 리스트
accuracies = []
accuracy_std = []

# 각 pkl 파일을 순차적으로 처리
for file in history_files:
    # pkl 파일 로드
    with open(file, 'rb') as f:
        history = pickle.load(f)

    # 훈련 정확도와 검증 정확도
    train_accuracy = history['accuracy']  # 훈련 정확도
    val_accuracy = history['val_accuracy']  # 검증 정확도

    # 평균 정확도 계산
    train_accuracy_mean = np.mean(train_accuracy) * 100  # 백분율로 변환
    val_accuracy_mean = np.mean(val_accuracy) * 100  # 백분율로 변환

    # 표준편차 계산
    train_accuracy_std = np.std(train_accuracy) * 100
    val_accuracy_std = np.std(val_accuracy) * 100

    # 모델별 정확도 리스트에 추가
    accuracies.append(val_accuracy_mean)  # 검증 정확도
    accuracy_std.append(val_accuracy_std)  # 검증 정확도 표준편차

# 그래프 설정
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='gray', alpha=0.7)  # 막대그래프

# 에러 바 추가 (표준편차)
plt.errorbar(model_names, accuracies, yerr=accuracy_std, fmt='none', color='red', capsize=5, elinewidth=2)

# 그래프 제목과 레이블 설정
plt.title('Prediction Model Accuracy', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy [%]', fontsize=14)

# 정확도 값 출력 (각 바 위에)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)

# 그래프 출력
plt.tight_layout()
plt.show()