import numpy as np
import matplotlib.pyplot as plt

class KArmedBandit:
    def __init__(self, k=10, mean=0, std=1):

        self.k = k
        self.q_true = np.random.normal(mean, std, k)  # 真实奖励均值
        self.optimal_action = np.argmax(self.q_true)  # 最优动作索引
        self.reset()
    
    def reset(self):
        self.q_estimates = np.zeros(self.k)  # 估计值
        self.action_counts = np.zeros(self.k)  # 每个动作的选择次数
    
    def step(self, action):

        reward = self.q_true[action]  # 直接返回固定奖励值（无噪声）
        self.action_counts[action] += 1
        return reward, action == self.optimal_action

    def optimal_action_percentage(self, action_history):
        return np.mean(np.array(action_history) == self.optimal_action)

