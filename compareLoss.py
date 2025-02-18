import matplotlib.pyplot as plt
import pickle

epochs = 100

# 학습 기록 파일 리스트
history_files = [
    'training_history_3sec.pkl',
    'training_history_6sec.pkl',
    'training_history_10sec.pkl',
    'training_history_30sec.pkl',
    'training_history_60sec.pkl',
    'training_history_120sec.pkl'
]

# 모델 이름 리스트
model_names = ['3s', '6s', '10s', '30s', '60s', '120s']

# 학습 기록을 저장할 리스트
histories = []

# 학습 기록 파일 불러오기
for file in history_files:
    with open(file, 'rb') as f:
        histories.append(pickle.load(f))

# 그래프 시각화: 2x3 그리드의 서브플롯
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Prediction Models Loss and MAE Comparison', fontsize=16)

# 서브플롯 좌표 설정
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

# 각 모델별로 Loss와 MAE 그래프 그리기
for idx, (model_name, pos) in enumerate(zip(model_names, positions)):
    row, col = pos
    ax = axes[row][col]  # 서브플롯 위치 설정
    ax.plot(histories[idx]['loss'], label='Loss', color='tab:blue')
    ax.plot(histories[idx]['mae'], label='MAE', color='tab:orange')
    ax.set_title(f'{model_name} Prediction Model')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.legend()
    # ax.grid() 호출 삭제 또는 False 설정
    ax.grid(False)

# 여백 조정
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()