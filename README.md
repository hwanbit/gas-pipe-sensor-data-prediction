# Time Series Sensor Data Prediction Model
This project implements a CNN-based time series prediction model for gas pipe's sensor data analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Data Structure](#data-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Error Analysis](#error-analysis)
- [Files Generated](#files-generated)

## Overview
The model is designed to predict multiple sensor values (temperature, pressure, acceleration, and gas leak measurements) at different time intervals into the future (3, 6, 10, 30, 60, and 120 seconds) based on historical data. This multi-temporal prediction approach allows for both short-term and long-term forecasting of sensor behaviors.

## Features
- Multi-variable time series prediction using CNN
- Sliding window data preparation
- Data preprocessing including outlier removal and scaling
- Automated model checkpointing
- Early stopping to prevent overfitting
- Comprehensive error analysis and visualization

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/hwanbit/gas-pipe-sensor-data-prediction.git
   ```
2. Install dependencies.

## Usage
1. Prepare your sensor data CSV files in the `./Data/` directory
2. Run the main script:
```python
python project_3sec.py
```
3. Load saved model for predictions:
```python
from tensorflow.keras.models import load_model
model = load_model('best_model_3sec.h5')
```
4. View training history:
```python
import pickle
with open('training_history_3sec.pkl', 'rb') as file:
    loaded_history = pickle.load(file)
```

## Dependencies
- Python 3.8
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Project Structure
```
.
│
├── Data/                  # Sensor data CSV files
│   └── *.csv              # Sensor measurement data files
│
├── compareError.py        # Generate error comparison graph
├── compareLoss.py         # Generate loss comparison graph
├── predictionAcc.py       # Generate Prediction model Accuracy graph
├── project_3sec.py        # 3 second prediction model
├── project_6sec.py        # 6 second prediction model
├── project_10sec.py       # 10 second prediction model
├── project_30sec.py       # 30 second prediction model
├── project_60sec.py       # 60 second prediction model
├── project_120sec.py      # 120 second prediction model
└── projectMerged.py       # Prediction model with all data
```

## Data Structure
The model works with CSV files containing the following sensor measurements:
- temperature_1: First temperature sensor(°C)
- pressure_1: First pressure sensor
- temperature_2: Second temperature sensor(°C)
- pressure_2: Second pressure sensor
- accel: Acceleration measurements
- gas_leak: Gas leak sensor readings

## Data Preprocessing
- Removal of temperature readings above 100°C
- Feature scaling:
  - Temperatures normalized by 100
  - Pressures normalized by 3.25
  - Acceleration multiplied by 10
  - Gas leak normalized by 1007
- Sliding window approach (e.g., 3-second window, 3-second prediction)

## Model Architecture
```python
Model: Sequential
- Conv2D(32, (3,3), activation='relu', input_shape=(3, 6, 1))
- Flatten()
- Dense(256, activation='sigmoid')
- Dense(6)
```

## Model Training
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE), Accuracy
- Validation split: 20%
- Early stopping with 10 epochs patience
- Model checkpointing based on validation loss

## Error Analysis
The model provides error statistics for each sensor type:
- Mean absolute error
- Minimum error
- Maximum error

## Files Generated
- `best_model_3sec.h5`: Best performing model weights
- `training_history_3sec.pkl`: Training history
- `predictions3sec_and_labels.npz`: Test predictions and actual values

---

## CNN 모델링을 통한 예측 모델 생성
가스 배관의 온도와 압력, 진동, 가스 누출 데이터가 들어있는 1,000개의 csv 파일을 이용하여 각 변수의 미래 값을 예측하는 모델을 만드는 것이 목표입니다.<br/>
미래의 3초 뒤, 6초 뒤, 10초 뒤, 30초 뒤, 60초 뒤, 120초 뒤의 예측 값을 구하고 그래프로 표현하여 원본 그래프와의 차이를 확인하는 것으로 모델을 평가할 수 있을 것입니다.

### 1. 슬라이딩 윈도우 함수 정의
슬라이딩 윈도우 기법: 시간 종속성을 학습하는 데 효과적인 방법이고, CNN 모델을 활용하기 위해선 입력 데이터의 크기가 고정되어야 합니다.
```python
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
```

### 2. CSV 파일을 랜덤으로 읽어오기
학습데이터 1,000개 파일 목록을 읽은 뒤에 랜덤으로 파일을 선택하고 원하는 개수만큼 가져옵니다.
```python
directory = './Data/'
all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
selected_files = random.sample(all_files, 100)
print(f'{len(selected_files)}개의 랜덤 csv 파일 : ', selected_files)
```

### 3. 각 파일에 슬라이딩 윈도우 적용
슬라이딩 윈도우가 적용된 파일을 모두 합쳐 Dataframe 형태로 만듭니다.
```python
# 최종 데이터프레임을 저장할 빈 리스트 초기화
df_list = []

# 슬라이딩 윈도우 파라미터 설정
window_size = 3  # 초 동안의 데이터 사용
target_step = 3  # 초 뒤의 데이터 예측

# 파일들을 순회하면서 처리
for file in selected_files:
    # 각 파일 읽기
    current_df = pd.read_csv(os.path.join(directory, file),
                             names=['second', 'temperature_1', 'pressure_1',
                                    'temperature_2', 'pressure_2', 'accel','gas_leak'])

    # 슬라이딩 윈도우 함수 적용
    X, y = sliding_window(current_df.values, window_size, target_step)

    # 필요하다면 여기서 X와 y를 추가 처리하거나 저장할 수 있음
    # 예를 들어, 현재 파일의 데이터를 리스트에 추가
    df_list.append(current_df)

# 모든 데이터프레임 결합
df = pd.concat(df_list, ignore_index=True)
```

### 4. 데이터 분석
데이터를 콘솔창에 출력하거나 각 열의 데이터를 그래프로 시각화하여 나타냅니다.
```python
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
```
```python
# 데이터를 시각화하여 확인 (한 창에 6개의 그래프)
plt.figure(figsize=(16, 10))  # 전체 그래프 크기 설정

# 각 컬럼에 대해 서브플롯 생성
for i, column in enumerate(df.columns[1:], start=1):
    plt.subplot(2, 3, i)  # 2행 3열 중 i번째 서브플롯
    plt.plot(df.index, df[column], label=column)
    plt.title(f'{column} Trends', fontsize=14)
    plt.xlabel('Seconds', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

plt.tight_layout()  # 서브플롯 간격 자동 조정
plt.show()

# 각 열에 대해 박스 플롯 생성 (이상치 확인)
for column in df.columns[1:]:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[column].dropna(), vert=False, patch_artist=True)
    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)
    plt.show()
```
<img src="https://github.com/user-attachments/assets/ca60903f-39ec-4310-b343-32293388d95c" height="400"/>
<img src="https://github.com/user-attachments/assets/e844eec8-569d-4703-939b-71bdeb70fc33" height="500"/>

### 5. 온도 데이터 전처리
온도 데이터가 0도부터 110도까지 존재하는데, 정규화 전에 110도 이상 데이터를 제외할지 여부를 결정합니다.
```python
temp1_above_100 = df[df['temperature_1'] > 100]
temp2_above_100 = df[df['temperature_2'] > 100]
print(f"temperature_1에서 100도 이상인 데이터 개수 : {len(temp1_above_100)}")
print(f"temperature_2에서 100도 이상인 데이터 개수 : {len(temp2_above_100)}")

df = df[(df['temperature_1'] <= 100) & (df['temperature_2'] <= 100)]
print(f"100도를 초과하는 데이터를 제거한 후의 행 개수 : {len(df)}")
```

### 6. 훈련 데이터 스케일링
각 열의 최댓값으로 나누어 데이터 정규화를 진행합니다.
```python
df['temperature_1'] = df['temperature_1'] / 100
df['pressure_1'] = df['pressure_1'] / 3.25
df['temperature_2'] = df['temperature_2'] / 100
df['pressure_2'] = df['pressure_2'] / 3.25
df['accel'] = df['accel'] * 10
df['gas_leak'] = df['gas_leak'] / 1007
```

### 7. Numpy array로 변환
second 열을 삭제하고 Dataframe을 Numpy array로 변환합니다.<br/>
모델 학습 시 Numpy array로 진행하는 것이 연산을 빠르게 해주기 때문입니다.
```python
df.drop(columns = ['second'])
train_data = df.to_numpy()
```

### 8. 모델 학습 방법
X 데이터는 과거 window_size초 동안의 데이터입니다. y 데이터는 미래 target_step초 이후의 데이터, 즉 라벨 데이터입니다.<br/>
CNN 학습 모델에 데이터를 넣기 위해 reshape 해줍니다. CNN은 4차원 입력 형식을 받아들이므로 4차원으로 변환해주어야 합니다.
```python
X_train, y_train = sliding_window(train_data, window_size, target_step)

X_train = np.reshape(X_train, newshape=(-1, window_size, 6, 1))
y_train = np.reshape(y_train, newshape=(-1, 6))
```

### 9. 모델 설계
모델은 아래와 같이 설계하였습니다.
* 입력층: window_size의 6개 열을 1채널로 입력받아 CNN이 처리하도록 합니다. 2D Convolution 연산을 수행하며 32개의 필터를 사용합니다. 커널 크기는 3x3이며, activation function으로는 ReLU 함수를 사용합니다.
* Flatten 층: 다차원 데이터를 1차원 벡터로 평탄화합니다.
* Dense 층: 256개의 노드를 가지며 activation function으로는 sigmoid 함수를 사용합니다.
* 출력층: 최종적으로 6개의 값을 출력합니다. 회귀 문제에서는 출력층에 활성화 함수를 사용하지 않는 경우가 많아 그대로 출력되게 하였습니다.
```python
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(window_size, 6, 1)),
    Flatten(),
    Dense(256, activation='sigmoid'),
    Dense(6)
])
```

### 10. 모델 컴파일
모델 컴파일을 optimizer로는 adam을 설정하고, loss는 mse, metrics는 mae를 설정합니다.
* adam: 시계열 데이터같은 연속적 특성을 학습할 때 수렴이 빠르고 안정적입니다.
* mse: 오차를 제곱하여 평균을 낸 값입니다. 시계열 예측이나 회귀 문제에서 가장 일반적으로 사용되는 손실 함수입니다.
* mae: 오차의 절대값을 평균 낸 값입니다. 모델 성능을 평가할 때 실제값과 예측값의 평균 차이를 직관적으로 확인할 수 있습니다.
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])
```

### 11. 콜백
체크포인트, 콜백, 조기 종료를 설정합니다.
```python
# val_loss가 가장 작을 때의 모델 저장
model_checkpoint = ModelCheckpoint(
    filepath='best_model_3sec.h5',  # 모델 저장 경로
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
history_3sec = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=40,
    shuffle=True,
    verbose=1,
    callbacks=[model_checkpoint, early_stopping]
)
```

---

## 그래프를 생성하여 모델 평가
* project.py 파일에서 학습된 모델로 테스트 데이터에 대한 예측을 수행하고 예측 결과 그래프와 원본 데이터 그래프를 생성합니다.<br/>
<img src="https://github.com/user-attachments/assets/bbd29b0a-9057-4471-8b6e-0c3c8bd14f4d" height="400"/><br/>

* compareLoss.py 파일에서 모델 평가를 진행한 뒤 loss와 MAE 그래프를 생성합니다.
<img src="https://github.com/user-attachments/assets/19614d71-e4af-4eee-be46-4fd867a31aa2" height="400"/><br/>

* compareError.py 파일에서 시간별 예측 모델들의 오차율을 비교하는 그래프를 생성합니다.
<img src="https://github.com/user-attachments/assets/e41e46a4-f884-42a7-9af6-3b2ca444a41f" height="500"/><br/>

* predictionAcc.py 파일에서 모델별 예측 정확도 및 표준편차 그래프를 생성합니다.
<img src="https://github.com/user-attachments/assets/e65b14ff-59cd-4a08-8cbe-4a7548f91ba6" height="400"/><br/>

---

## 결론
이번 모델링에서 깊이 고찰하며 해결해야 했던 문제들은 다음과 같습니다.
1. 여러 개의 데이터 파일을 불러올 시, 파일 당 윈도우 사이즈를 겹치지 않게 적용해야하는 일
2. 불러왔을 때 그대로 Dataframe을 그대로 사용하는 것이 아닌, numpy array로 변환해야 하는 일
3. CNN 모델에 데이터를 넣기 위해 reshape 해야하는 일
4. 모델 별 적정 window size를 찾는 일
5. 데이터 양에 맞는 batch 사이즈를 찾는 일
6. 적정 epoch를 찾기 위해 callback을 도입하는 일
7. 원하는 그래프를 띄우기 위해 알맞은 데이터를 수집하는 일
<br/>

따라서 다음과 같은 결과를 학습할 수 있었습니다.
1. 시계열 데이터 예측의 시간 범위 고려
2. 파라미터 값에 따른 모델의 성능 차이
3. 데이터 전처리의 중요성
4. 모델 학습의 안정성
5. 모델 평가의 다각화
