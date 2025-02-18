import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# 훈련 데이터 로드
file_path = './Data/'
all_files = glob.glob(os.path.join(file_path, "_Merged_Data_data_set_00*.csv"))

pd.set_option('display.max_columns', None) # 모든 열 출력

# 파일들을 로드하고 결합
dataframes = []
for file in all_files:
    # 각 파일 읽기 (컬럼 이름 포함 or 제외 여부 설정 가능)
    df = pd.read_csv(file, names=['second', 'temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel', 'gas_leak'])
    dataframes.append(df)

# 전체 데이터 파일
merged_data = pd.concat(dataframes, ignore_index=True)

# 데이터 확인
print(f"총 데이터 개수: {merged_data.shape[0]} 행")
print(merged_data.head())

# print(df.info()) # 결측치 확인 (없음, 모든 데이터 float 형태)

# null 값이 있는지 확인 (없음)
# print(df.isnull().any(axis=1))
# print(df.isnull().any(axis=0))

# 데이터 정보 확인
print(merged_data.describe())
mode_df = merged_data.mode().iloc[0].dropna()
print('===========mode===========')
print(mode_df)

# 조건 필터링으로 100도 이상인 데이터 개수 확인
temp1_above_100 = merged_data[merged_data['temperature_1'] > 100]
temp2_above_100 = merged_data[merged_data['temperature_2'] > 100]
print(f"temperature_1에서 100도 이상인 데이터 개수 : {len(temp1_above_100)}")
print(f"temperature_2에서 100도 이상인 데이터 개수 : {len(temp2_above_100)}")

# 온도가 100 이하인 데이터만 남기기
merged_data = merged_data[(merged_data['temperature_1'] <= 100) & (merged_data['temperature_2'] <= 100)]
print(f"100도를 초과하는 데이터를 제거한 후의 행 개수 : {len(merged_data)}")

# 훈련 데이터 스케일링
merged_data['temperature_1'] = merged_data['temperature_1'] / 100
merged_data['pressure_1'] = merged_data['pressure_1'] / 3.25
merged_data['temperature_2'] = merged_data['temperature_2'] / 100
merged_data['pressure_2'] = merged_data['pressure_2'] / 3.25
merged_data['accel'] = merged_data['accel'] * 10
merged_data['gas_leak'] = merged_data['gas_leak'] / 1007

merged_data.drop(columns = ['second'])

# numpy array로 변환
train_data = merged_data.to_numpy()

# 슬라이딩 윈도우 함수 정의
def sliding_window(data, window_size, target_step):
    """
    NumPy 배열을 슬라이딩 윈도우로 변환
    Args:
        dataframe (pd.DataFrame): 시계열 데이터가 있는 DataFrame
        window_size (int): 입력 시퀀스의 길이 (과거 데이터를 몇 초 사용할지)
        target_step (int): 미래 예측할 데이터 (몇 초 이후 데이터를 예측할지)

    Returns:
        X (np.array): 입력 데이터 (샘플, 시간, 특성)
        y (np.array): 출력 데이터 (샘플, 특성)
    """
    X = []
    y = []

    for i in range(len(data) - window_size - target_step + 1):
        X.append(data[i:i + window_size, 1:])
        y.append(data[i + window_size + target_step - 1, 1:])
    return np.array(X), np.array(y)

# 슬라이딩 윈도우 파라미터 설정
window_size = 5  # 5초 동안의 데이터 사용
target_step = 3  # 3초 뒤의 데이터 예측

# 슬라이딩 윈도우로 X, y 생성
X_train, y_train = sliding_window(train_data, window_size, target_step)

X_train = np.reshape(X_train, newshape=(-1, 5, 6, 1))
y_train = np.reshape(y_train, newshape=(-1, 6))

# CNN 모델 정의
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(5, 6, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 모델 학습
history_3sec = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    shuffle=True,  # 학습 과정에서 섞기
    verbose=1
)

# 모델 요약 출력
model.summary()

# Loss와 MAE 그래프 표시
plt.figure(figsize=(12, 5))
plt.plot(history_3sec.history['loss'], label='Loss')
plt.plot(history_3sec.history['mae'], label='MAE')
plt.title('Train Data')
plt.legend()
plt.tight_layout()
plt.show()

# 테스트 데이터 로드
test_data = pd.read_csv('./Data/_Merged_Data_data_set_01500.csv',
                 names=['second', 'temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel', 'gas_leak'])

# 온도가 100 이하인 데이터만 남기기
test_data = test_data[(test_data['temperature_1'] <= 100) & (test_data['temperature_2'] <= 100)]
print(f"100도를 초과하는 데이터를 제거한 후의 행 개수 : {len(test_data)}")

test_data['temperature_1'] = test_data['temperature_1'] / 100
test_data['pressure_1'] = test_data['pressure_1'] / 3.25
test_data['temperature_2'] = test_data['temperature_2'] / 100
test_data['pressure_2'] = test_data['pressure_2'] / 3.25
test_data['accel'] = test_data['accel'] * 10
test_data['gas_leak'] = test_data['gas_leak'] / 1007

test_data.drop(columns = ['second'])

test_data = test_data.to_numpy()

# 테스트 데이터를 슬라이딩 윈도우로 변환
X_test, y_test = sliding_window(test_data, window_size, target_step)

X_test = np.reshape(X_test, (-1, 5, 6, 1))
y_test = np.reshape(y_test, (-1, 6))

# 예측 및 결과 시각화
# 테스트 데이터에 대한 모델의 예측 결과를 시각화한다.
predictions = model.predict(X_test)

# 그래프 출력
plt.figure(figsize=(12, 6))

# Plot 1: 모델 예측 결과
plt.subplot(1, 2, 1)
time_steps_predictions = np.arange(predictions.shape[0])
for i, label in enumerate(['temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel', 'gas_leak']):
    plt.plot(time_steps_predictions, predictions[:, i], label=label)
plt.title("3sec predictions")
plt.xlabel("Seconds")
plt.ylabel("Ratio [0-1]")
plt.legend()

# Plot 2: 원본 데이터
plt.subplot(1, 2, 2)
time_steps = np.arange(test_data.shape[0])  # numpy 배열에서 인덱스를 생성
for i, label in enumerate(['temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel', 'gas_leak']):
    plt.plot(time_steps, test_data[:, i+1], label=label)
plt.title("Original")
plt.xlabel("Seconds")
plt.ylabel("Ratio [0-1]")
plt.legend()

plt.tight_layout()
plt.show()

# 오차 계산
errors = test_data[:predictions.shape[0], 1:] - predictions  # 원본 데이터에서 예측 값 차이 계산

# 평균, 최소, 최대 오차 계산
mean_error = np.mean(np.abs(errors), axis=0)
min_error = np.min(np.abs(errors), axis=0)
max_error = np.max(np.abs(errors), axis=0)

# 결과 출력
columns = ['temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel', 'gas_leak']
error_table = pd.DataFrame({
    'Feature': columns,
    'Mean Error': mean_error,
    'Min Error': min_error,
    'Max Error': max_error
})

print(error_table)