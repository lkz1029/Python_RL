import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from collections import deque
import random

# 환경 설정
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.95  # 할인율
epsilon = 1.0  # 탐색 vs 활용
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
train_start = 1000
memory_size = 2000

# Replay Buffer
memory = deque(maxlen=memory_size)

# Q-Network
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=state_size, activation="relu"))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(action_size, activation="linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model

model = build_model()
target_model = build_model()

# Target 모델 업데이트
def update_target_model():
    target_model.set_weights(model.get_weights())

# Epsilon-greedy 정책
def epsilon_greedy_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# 학습 함수
def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(state)
        if done:
            target[0][action] = reward
        else:
            t = target_model.predict(next_state)
            target[0][action] = reward + gamma * np.amax(t[0])
        
        model.fit(state, target, epochs=1, verbose=0)

# 학습 진행
episodes = 500
for e in range(episodes):
    state, _ = env.reset()  # 첫 번째 값만 가져오기

    # 상태의 형식과 내용을 확인하여 디버깅
    print(f"Raw initial state: {state}, type: {type(state)}")  # 원본 상태 출력

    # 상태를 NumPy 배열로 변환
    try:
        state = np.array(state, dtype=np.float32)  # NumPy 배열로 변환
        state = np.reshape(state, [1, state_size])  # [1, state_size] 형태로 변환
        print(f"Processed initial state: {state}, type: {type(state)}, shape: {state.shape}")  # 변환 후 상태 출력

    except Exception as e:
        print(f"Error processing initial state: {e}")  # 오류 발생 시 메시지 출력
        continue  # 다음 에피소드로 넘어감

    done = False
    time = 0

    while not done:
        action = epsilon_greedy_action(state, epsilon)

        # env.step()의 반환값 확인
        step_result = env.step(action)  # 반환값을 리스트에 저장
        print(f"Step result: {step_result}")  # 반환 값 출력
        
        # 반환값 수에 따라 처리
        if len(step_result) == 5:
            next_state, reward, done, truncated, info = step_result
        elif len(step_result) == 4:
            next_state, reward, done, info = step_result
            truncated = False  # 기본값 설정
        else:
            print(f"Unexpected return value: {step_result}")  # 예외 사항 출력
            break  # 루프 종료

        # next_state도 동일하게 처리
        print(f"Raw next state: {next_state}, type: {type(next_state)}")  # 원본 next 상태 출력

        try:
            next_state = np.array(next_state, dtype=np.float32)  # NumPy 배열로 변환
            next_state = np.reshape(next_state, [1, state_size])  # [1, state_size] 형태로 변환
            print(f"Processed next state: {next_state}, type: {type(next_state)}, shape: {next_state.shape}")  # 변환 후 상태 출력

        except Exception as e:
            print(f"Error processing next state: {e}")  # 오류 발생 시 메시지 출력
            done = True  # 다음 상태 처리에 문제가 있을 경우 에피소드 종료

        # 경험을 저장
        memory.append((state, action, reward, next_state, done))
        state = next_state
        time += 1

        # 학습
        if len(memory) > train_start:
            replay()

        # Target 네트워크 업데이트
        if done:
            update_target_model()
            print(f"Episode: {e + 1}/{episodes}, Score: {time}, Epsilon: {epsilon:.2f}")

    # Epsilon 감소
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()

