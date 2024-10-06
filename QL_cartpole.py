# coding:utf-8
# [0] 라이브러리 임포트
import gym  # 카트폴(CartPole) 실행 환경
from gym import wrappers  # gym의 이미지 저장
import numpy as np
import time


# [1] Q 함수를 이산화하여 정의하는 함수 ------------
# 관측한 상태를 이산값으로 디지털 변환하는 함수
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 각 값을 이산값으로 변환
def digitize_state(observation):
    # observation의 길이를 확인합니다.
    if len(observation) != 4:
        raise ValueError(f"Expected 4 values in observation, but got {len(observation)}: {observation}")

    cart_pos = observation[0]
    cart_v = observation[1]
    pole_angle = observation[2]
    pole_v = observation[3]
    
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])


# [2] 행동 a(t)를 구하는 함수 -------------------------------------
def get_action(next_state, episode):
    # 점차 최적 행동만을 취하는 ε-탐욕법
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action


# [3] Q 테이블을 업데이트하는 함수 -------------------------------------
def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q = max(q_table[next_state][0], q_table[next_state][1])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_Max_Q)

    return q_table

# [4] 메인 함수 시작 파라미터 설정--------------------------------------------------------
env = gym.make('CartPole-v1')  # CartPole 환경 생성
max_number_of_steps = 200  # 1회 시도의 스텝 수
num_consecutive_iterations = 100  # 학습 완료 평가에 사용되는 평균 시도 회수
num_episodes = 2000  # 총 시도 회수
goal_average_reward = 195  # 이 보상을 초과하면 학습 종료 (중심으로의 제어 없음)
# 상태를 6분할(4변수)하여 Q 함수(테이블)를 생성
num_dizitized = 6  # 분할 수
q_table = np.random.uniform(
    low=-1, high=1, size=(num_dizitized**4, env.action_space.n))

total_reward_vec = np.zeros(num_consecutive_iterations)  # 각 시도의 보상을 저장
final_x = np.zeros((num_episodes, 1))  # 학습 후 각 시도의 t=200에서의 x 위치를 저장
islearned = 0  # 학습 완료 플래그
isrender = 0  # 렌더링 플래그


# [5] 메인 루틴--------------------------------------------------
for episode in range(num_episodes):  # 시도 수 만큼 반복
    # 환경 초기화
    observation = env.reset()
    
    # 관측값 출력
    print(f"Initial observation: {observation}")  # 관측값 출력

    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):  # 1회 시도의 루프
        if islearned == 1:  # 학습 완료 시 카트폴을 렌더링
            env.render()
            time.sleep(0.1)
            print(observation[0])  # 카트의 x 위치를 출력

        # 행동 a_t의 실행으로 s_{t+1}, r_{t} 등을 계산
        observation, reward, done, info = env.step(action)

        # 관측값 출력
        print(f"Observation after action: {observation}")  # 관측값 출력

        # 보상을 설정하고 부여
        if done:
            if t < 195:
                reward = -200  # 넘어지면 패널티
            else:
                reward = 1  # 서있는 상태에서 종료 시 패널티 없음
        else:
            reward = 1  # 각 스텝에서 서있으면 보상 추가

        episode_reward += reward  # 보상 추가

        # 이산 상태 s_{t+1}를 구하고 Q 함수를 업데이트
        next_state = digitize_state(observation)  # t+1에서의 관측 상태를 이산값으로 변환
        q_table = update_Qtable(q_table, state, action, reward, next_state)

        # 다음 행동 a_{t+1}을 구함
        action = get_action(next_state, episode)  # a_{t+1}

        state = next_state

        # 종료 시의 처리
        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:],
                                          episode_reward))  # 보상 기록
            if islearned == 1:  # 학습 완료 시 최종 x 좌표 저장
                final_x[episode, 0] = observation[0]
            break

    if (total_reward_vec.mean() >=
            goal_average_reward):  # 직전 100 에피소드가 규정 보상 이상이면 성공
        print('Episode %d train agent successfully!' % episode)
        islearned = 1
        # np.savetxt('learned_Q_table.csv', q_table, delimiter=",") # Q 테이블 저장 시
        if isrender == 0:
            # env = wrappers.Monitor(env, './movie/cartpole-experiment-1') # 동영상 저장 시
            isrender = 1
    # 10 에피소드로 어떤 행동을 보일지 보고 싶다면 아래 주석 해제
    # if episode > 10:
    #     if isrender == 0:
    #         env = wrappers.Monitor(env, './movie/cartpole-experiment-1') # 동영상 저장 시
    #         isrender = 1
    #     islearned = 1;

if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")
