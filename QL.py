import gym
import numpy as np
import torch
import random
import torch.nn as nn
import copy
from matplotlib import pyplot as plt
import os
from gym.wrappers import RecordVideo

# 하이퍼파라미터 설정
LEARNING_RATE = 0.0005  # 학습률
GAMMA = 0.95  # 할인 인자
ENV_NAME = 'CartPole-v1'  # 사용할 환경 이름
EPSILON = 0.1  # Epsilon-greedy 정책에서의 탐색 확률

# Q 함수 신경망 정의
class Agent(nn.Module):
    def __init__(self) -> None:
        super(Agent, self).__init__()
        self.env = gym.make(ENV_NAME)  # 환경 생성
        self.input_size = self.env.observation_space.shape[0]  # 상태 공간 차원
        self.output_size = self.env.action_space.n  # 행동 공간 차원
        # 신경망 레이어 정의
        self.fc1 = nn.Linear(self.input_size, 128)  # 첫 번째 은닉층
        self.fc2 = nn.Linear(128, 128)  # 두 번째 은닉층
        self.fcq = nn.Linear(128, self.output_size)  # Q 값 출력 레이어
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))  # 첫 번째 은닉층의 ReLU 활성화
        x = torch.relu(self.fc2(x))  # 두 번째 은닉층의 ReLU 활성화
        q = self.fcq(x)  # Q 값 출력
        return q

# Q-learning 에이전트 클래스
class QLearning():
    def __init__(self):
        self.agent = Agent()  # Q 함수 네트워크 인스턴스 생성
        self.env = gym.make(ENV_NAME)  # 환경 생성
        self.epsilon = EPSILON  # Epsilon-greedy 초기화
        self.gamma = GAMMA  # 할인 인자 초기화
        self.loss_fn = torch.nn.MSELoss()  # 손실 함수 (MSE)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=LEARNING_RATE)  # 옵티마이저 (Adam)

    def train(self):
        print('------------Start Training--------------')
        reward_list = []  # 보상 기록 리스트
        max_reward = 0  # 최대 보상 초기화
        for epoch in range(10000):  # 10,000 에포크 동안 학습
            done = False
            state_ = self.env.reset()[0]  # 환경 초기화 및 첫 상태 가져오기
            reward_sum = 0  # 보상 합계 초기화
            step = 0  # 스텝 카운트 초기화
            while not done:  # 에피소드가 끝날 때까지 반복
                state = torch.from_numpy(state_).float()  # 상태를 텐서로 변환
                q_val = self.agent(state)  # 현재 상태에 대한 Q 값 계산
                
                # Epsilon-greedy 정책으로 행동 선택
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()  # 랜덤 행동 선택
                else:
                    action = torch.argmax(q_val).item()  # Q 값이 최대인 행동 선택
                    
                next_state_, reward, terminated, truncated, _ = self.env.step(action)  # 선택한 행동 수행
                next_state = torch.from_numpy(next_state_).float()  # 다음 상태를 텐서로 변환
                step += 1  # 스텝 카운트 증가
                done = terminated or truncated  # 에피소드 종료 여부 확인
                
                reward_sum += reward  # 보상 합계에 현재 보상 추가
                newQ = self.agent(next_state).detach()  # 다음 상태에서의 Q 값 계산 (그래디언트 제외)
                maxQ = torch.max(newQ)  # 최대 Q 값 계산
                
                X = q_val[action].unsqueeze(0)  # 현재 상태의 Q 값 가져오기

                # 최종 Q 값 계산
                Y = reward if done else reward + self.gamma * maxQ
                Y = torch.Tensor([Y]).detach()  # 최종 Q 값 텐서로 변환
                loss = self.loss_fn(X, Y)  # 손실 계산
                
                # Q 함수 네트워크 업데이트
                self.optimizer.zero_grad()  # 기울기 초기화
                loss.backward()  # 역전파
                self.optimizer.step()  # 가중치 업데이트
                
                state_ = next_state_  # 상태 업데이트

            # 에포크 결과 출력
            print("Epoch: {}, reward: {}, step: {}".format(epoch, reward_sum, step))
            reward_list.append(reward_sum)  # 보상 기록 추가
            
            # 최상의 모델 저장
            if reward_sum >= max_reward:
                max_reward = reward_sum
                torch.save(copy.deepcopy(self.agent.state_dict()), "Q-learning.pt")  # 모델 저장
                print("Model Saved")
            
        self.env.close()  # 환경 종료
        
        # 학습 과정 시각화
        n_epochs = len(reward_list)
        plt.plot(np.arange(n_epochs), reward_list)  # 보상 시각화
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.savefig('Q-learning_reward.png')  # 시각화 결과 저장

# 테스트 함수
def test(model):
    print('------------Start Test--------------')
    model.eval()  # 평가 모드로 전환
    env = gym.make(ENV_NAME)  # 환경 생성
    for epoch in range(10):  # 10 에피소드 동안 테스트
        done = False
        state_ = env.reset()[0]  # 환경 초기화 및 첫 상태 가져오기
        reward_sum = 0  # 보상 합계 초기화
        step = 0  # 스텝 카운트 초기화
        while not done:  # 에피소드가 끝날 때까지 반복
            state = torch.from_numpy(state_).float()  # 상태를 텐서로 변환
            q_val = model(state)  # Q 값 계산
            action = torch.argmax(q_val).item()  # 행동 선택
            
            next_state_, reward, terminated, truncated, _ = env.step(action)  # 선택한 행동 수행
            reward_sum += reward  # 보상 합계에 현재 보상 추가
            state_ = next_state_  # 상태 업데이트
            done = terminated or truncated  # 에피소드 종료 여부 확인
            
            step += 1  # 스텝 카운트 증가
        print("Epoch: {}, reward: {}, step: {}".format(epoch, reward_sum, step))  # 에피소드 결과 출력

# 비디오 테스트 함수
def test_video(model):
    print('------------Start Test Video--------------')
    model.eval()  # 평가 모드로 전환
    env = RecordVideo(gym.make(ENV_NAME), "./")  # 비디오 기록을 위한 환경 생성
    for epoch in range(1):  # 1 에피소드 동안 비디오 테스트
        done = False
        state_ = env.reset()[0]  # 환경 초기화 및 첫 상태 가져오기
        reward_sum = 0  # 보상 합계 초기화
        step = 0  # 스텝 카운트 초기화
        while not done:  # 에피소드가 끝날 때까지 반복
            state = torch.from_numpy(state_).float()  # 상태를 텐서로 변환
            q_val = model(state)  # Q 값 계산
            action = torch.argmax(q_val).item()  # 행동 선택
            
            next_state_, reward, terminated, truncated, _ = env.step(action)  # 선택한 행동 수행
            reward_sum += reward  # 보상 합계에 현재 보상 추가
            state_ = next_state_  # 상태 업데이트
            done = terminated or truncated  # 에피소드 종료 여부 확인
            
            step += 1  # 스텝 카운트 증가
        print("Epoch: {}, reward: {}, step: {}".format(epoch, reward_sum, step))  # 에피소드 결과 출력

# 메인 함수
if __name__ == "__main__":
    model = QLearning()  # Q-learning 모델 인스턴스 생성
    # Uncomment the next line to train the model
    # model.train()  # 모델 학습 (주석 해제하면 학습 시작)

    test_model = Agent()  # 테스트 모델 인스턴스 생성
    if os.path.exists("Q-learning.pt"):  # 저장된 모델 파일 확인
        test_model.load_state_dict(torch.load("Q-learning.pt"))  # 모델 로드
    else:
        print("No pre-trained model found. Training from scratch.")  # 모델이 없으면 메시지 출력

    test_video(test_model)  # 비디오 테스트 실행
