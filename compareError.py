import numpy as np
import matplotlib.pyplot as plt
import pickle

# 저장된 파일에서 데이터 불러오기
data_1 = np.load('./predictions3sec_and_labels.npz')
data_2 = np.load('./predictions120sec_and_labels.npz')

# 불러온 데이터 (첫 번째 차원만 선택, 온도 1 기준)
predictions_3sec = data_1['predictions_3sec'][:, 0]  # 첫 번째 출력값 (온도 1)
predictions_60sec = data_2['predictions_120sec'][:, 0]
y_test_1 = data_1['y_test'][:, 0]  # 실제값의 첫 번째 출력값
y_test_2 = data_2['y_test'][:, 0]

# 오차 계산 (온도 1 기준)
errors_3sec = predictions_3sec - y_test_1
errors_60sec = predictions_60sec - y_test_2

# 그래프 비교
plt.figure(figsize=(10, 6))

# 모델 60sec 오차
plt.plot(errors_60sec, label="120sec", linestyle="--", color="blue")

# 모델 3sec 오차
plt.plot(errors_3sec, label="3sec", color="red")

# 그래프 설정
plt.title("Compare Error with Models (Temperature_1)", fontsize=16)
plt.xlabel("Samples", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# 그래프 출력
plt.show()