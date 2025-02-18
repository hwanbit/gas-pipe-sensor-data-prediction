import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

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

# CSV 파일들이 있는 디렉토리 경로
directory = './Data/'

# 디렉토리 내 모든 CSV 파일 목록 가져오기
all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 랜덤으로 k개의 CSV 파일 선택
selected_files = random.sample(all_files, 100)

# 선택된 파일 목록 출력
print(f'{len(selected_files)}개의 랜덤 csv 파일 : ', selected_files)

# 최종 데이터프레임을 저장할 빈 리스트 초기화
df_list = []

# 슬라이딩 윈도우 파라미터 설정
window_size = 120  # 초 동안의 데이터 사용
target_step = 120  # 초 뒤의 데이터 예측

# 파일들을 순회하면서 처리
for file in selected_files:
    # 각 파일 읽기
    current_df = pd.read_csv(os.path.join(directory, file),
                             names=['second', 'temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel','gas_leak'])

    # 슬라이딩 윈도우 함수 적용
    X, y = sliding_window(current_df.values, window_size, target_step)

    # 필요하다면 여기서 X와 y를 추가 처리하거나 저장할 수 있음
    # 예를 들어, 현재 파일의 데이터를 리스트에 추가
    df_list.append(current_df)

# 모든 데이터프레임 결합
df = pd.concat(df_list, ignore_index=True)

# 데이터프레임 출력
pd.set_option('display.max_columns', None)  # 모든 열 출력
print(df)

print(df.info()) # 결측치 확인 (없음, 모든 데이터 float 형태)

# null 값이 있는지 확인 (없음)
print(df.isnull().any(axis=1))
print(df.isnull().any(axis=0))

# 데이터 정보 확인
print(df.describe())
mode_df = df.mode().iloc[0].dropna()
print('===========mode===========')
print(mode_df)

# 조건 필터링으로 100도 이상인 데이터 개수 확인
temp1_above_100 = df[df['temperature_1'] > 100]
temp2_above_100 = df[df['temperature_2'] > 100]
print(f"temperature_1에서 100도 이상인 데이터 개수 : {len(temp1_above_100)}")
print(f"temperature_2에서 100도 이상인 데이터 개수 : {len(temp2_above_100)}")

# 온도가 100 이하인 데이터만 남기기
df = df[(df['temperature_1'] <= 100) & (df['temperature_2'] <= 100)]
print(f"100도를 초과하는 데이터를 제거한 후의 행 개수 : {len(df)}")

# 훈련 데이터 스케일링
df['temperature_1'] = df['temperature_1'] / 100
df['pressure_1'] = df['pressure_1'] / 3.25
df['temperature_2'] = df['temperature_2'] / 100
df['pressure_2'] = df['pressure_2'] / 3.25
df['accel'] = df['accel'] * 10
df['gas_leak'] = df['gas_leak'] / 1007

df.drop(columns = ['second'])

# numpy array로 변환
train_data = df.to_numpy()

# 슬라이딩 윈도우로 X, y 생성
X_train, y_train = sliding_window(train_data, window_size, target_step)

X_train = np.reshape(X_train, newshape=(-1, window_size, 6, 1))
y_train = np.reshape(y_train, newshape=(-1, 6))

# CNN 모델 정의
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(window_size, 6, 1)),
    Flatten(),
    Dense(256, activation='sigmoid'),
    Dense(6)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])

# 모델 체크포인트 콜백 설정
# val_loss가 가장 작을 때의 모델 저장
model_checkpoint = ModelCheckpoint(
    filepath='best_model_120sec.h5',  # 모델 저장 경로
    monitor='val_loss',        # 모니터링할 지표
    mode='min',                # val_loss는 최소값을 찾아야 함
    save_best_only=True,       # 가장 좋은 모델만 저장
    verbose=1                  # 저장 시 메시지 출력
)

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',        # 모니터링할 지표
    patience=10,               # 10번의 에포크 동안 개선되지 않으면 중단
    restore_best_weights=True  # 가장 좋은 가중치로 복원
)

# 모델 학습시 콜백 추가
history_120sec = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=40,
    shuffle=True,
    verbose=1,
    callbacks=[model_checkpoint, early_stopping]
)

# 저장된 모델의 최적 epoch 확인
if os.path.exists('best_model_120sec.h5'):
    print(f"\n최종 저장된 모델의 epoch: {model_checkpoint.best}")

print(history_120sec)

# 모델 요약 출력
model.summary()

# 모델의 성능 확인을 위해 점선과 실선으로 히스토그램으로 나타냄
history_dict = history_120sec.history  # loss와 dict 형태로 저장되어 있음
print(history_dict.keys())

with open('training_history_120sec.pkl', 'wb') as f:
    pickle.dump(history_dict, f)

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss)+1)

# r은 빨간색 실선
plt.plot(epochs, loss, 'r', label='Training loss')
# b는 파란색 실선
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Loss와 MAE 그래프 표시
plt.figure(figsize=(12, 5))
plt.plot(history_120sec.history['loss'], label='Loss')
plt.plot(history_120sec.history['mae'], label='MAE')
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

X_test = np.reshape(X_test, (-1, window_size, 6, 1))
y_test = np.reshape(y_test, (-1, 6))

# 예측 및 결과 시각화
# 테스트 데이터에 대한 모델의 예측 결과를 시각화함
predictions = model.predict(X_test)
# predictions와 y_test 저장
np.savez('predictions120sec_and_labels.npz',
         predictions_120sec=predictions,
         y_test=y_test)

# 그래프 출력
plt.figure(figsize=(12, 6))

# Plot 1: 모델 예측 결과
plt.subplot(1, 2, 1)
time_steps_predictions = np.arange(predictions.shape[0])
for i, label in enumerate(['temperature_1', 'pressure_1', 'temperature_2', 'pressure_2', 'accel', 'gas_leak']):
    plt.plot(time_steps_predictions, predictions[:, i], label=label)
plt.title("120sec predictions")
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