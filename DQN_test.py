import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque
from matplotlib import pyplot as plt
import os
from gym.wrappers import RecordVideo

# 하이퍼파라미터 설정
LEARNING_RATE = 0.001  # 학습률
GAMMA = 0.99  # 할인 인자
ENV_NAME = 'CartPole-v1'  # 사용할 환경 이름
EPSILON = 0.1  # Epsilon-greedy 정책에서의 탐색 확률
MAX_STEPS = 1000  # 주어진 시간 (최대 스텝 수)
MEMORY_SIZE = 10000  # Replay buffer 크기
BATCH_SIZE = 64  # 미니 배치 크기
TARGET_UPDATE = 100  # 타깃 네트워크 업데이트 주기

# Q 함수 신경망 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fcq = nn.Linear(256, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q = self.fcq(x)
        return q

# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(state), torch.tensor(action), torch.tensor(reward), torch.stack(next_state), torch.tensor(done)

    def __len__(self):
        return len(self.buffer)

# DQN 에이전트 클래스
class DQNAgent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.policy_net = DQN(self.input_size, self.output_size)
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.loss_fn = nn.MSELoss()
        self.steps_done = 0

    def select_action(self, state):
        # Epsilon-greedy 정책
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # 무작위 행동
        else:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()  # Q 값이 최대인 행동

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # 주 네트워크에서 Q 값 계산
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 타깃 네트워크에서 다음 상태의 최대 Q 값 계산
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_state_action_values = rewards + (1 - dones) * self.gamma * next_state_values

        # 손실 계산
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # 신경망 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        print('------------Start Training--------------')
        reward_list = []  # 보상 기록 리스트
        max_reward = 0  # 최대 보상 초기화

        for epoch in range(100000):
            done = False
            state = self.env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 상태 변환
            reward_sum = 0
            step = 0

            while not done and step < MAX_STEPS:
                action = self.select_action(state)  # 행동 선택
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                done = terminated or truncated

                # Replay Buffer에 저장
                self.memory.push(state, action, reward, next_state, done)

                # 상태 업데이트
                state = next_state
                reward_sum += reward
                step += 1

                # 모델 최적화
                self.optimize_model()

            # 타깃 네트워크 업데이트
            if epoch % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Epsilon 감소
            self.epsilon = max(0.01, self.epsilon * 0.995)

            # 결과 출력 및 보상 기록
            print(f"Epoch: {epoch}, reward: {reward_sum}, steps: {step}, epsilon: {self.epsilon:.3f}")
            reward_list.append(reward_sum)

            if reward_sum >= max_reward:
                max_reward = reward_sum
                torch.save(self.policy_net.state_dict(), "DQN_model.pt")
                print("Model Saved")

        self.env.close()

        # 학습 과정 시각화
        plt.plot(reward_list)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.savefig('DQN_training_rewards.png')

# 테스트 함수
def test_video(model):
    print('------------Start Test Video--------------')
    model.eval()

    env = RecordVideo(gym.make(ENV_NAME, render_mode="rgb_array"), video_folder="./videos", episode_trigger=lambda x: x % 1 == 0)
    done = False
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    reward_sum = 0
    step = 0

    while not done and step < MAX_STEPS:
        with torch.no_grad():
            action = torch.argmax(model(state)).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        done = terminated or truncated
        reward_sum += reward
        state = next_state
        step += 1

    print(f"Total reward: {reward_sum}, steps: {step}")
    env.close()

# 메인 함수
if __name__ == "__main__":
    agent = DQNAgent()
    # Uncomment the next line to train the model
    # agent.train()

    test_model = DQN(agent.input_size, agent.output_size)
    if os.path.exists("DQN_model.pt"):
        test_model.load_state_dict(torch.load("DQN_model.pt"))
    else:
        print("No pre-trained model found. Training from scratch.")

    test_video(test_model)
