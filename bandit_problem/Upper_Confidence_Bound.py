import numpy as np
import matplotlib.pyplot as plt
from bandit_environment import KArmedBandit  # 确保你的 `KArmedBandit` 类已定义

def ucb(bandit, c=2, steps=1000):
    """
    运行 UCB 策略
    bandit: KArmedBandit 环境
    c: 探索因子（控制探索的程度）
    steps: 运行的总步数
    返回：
    - rewards: 每一步的奖励
    - optimal_action_count: 每一步选择最优臂的次数
    """
    bandit.reset()  # 重置环境
    rewards = []
    optimal_action_count = []
    q_estimates = np.zeros(bandit.k)  # 估计值
    action_counts = np.zeros(bandit.k)  # 每个臂的选择次数

    # 先让每个臂都至少被选一次（避免除 0）
    for action in range(bandit.k):
        reward, is_optimal = bandit.step(action)
        rewards.append(reward)
        optimal_action_count.append(is_optimal)
        q_estimates[action] = reward
        action_counts[action] = 1

    for t in range(bandit.k, steps):
        ucb_values = q_estimates + c * np.sqrt(np.log(t) / action_counts)  # 计算 UCB 值
        action = np.argmax(ucb_values)  # 选择 UCB 最高的臂
        reward, is_optimal = bandit.step(action)

        # 记录结果
        rewards.append(reward)
        optimal_action_count.append(is_optimal)

        # 更新估计值
        action_counts[action] += 1
        q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]

    return rewards, optimal_action_count

def run_ucb_experiment(k=10, steps=1000, runs=200, c_values=[1, 2, 5]):
    """
    运行 Bandit 实验，测试不同的 UCB 探索因子 c
    k: 拉杆数量
    steps: 运行步数
    runs: 实验重复次数（统计平均结果）
    c_values: 不同的 UCB 探索因子
    """
    all_rewards = {c: np.zeros(steps) for c in c_values}
    all_optimal_actions = {c: np.zeros(steps) for c in c_values}

    for _ in range(runs):
        bandit = KArmedBandit(k=k)  # 每次实验创建新的 Bandit

        for c in c_values:
            rewards, optimal_actions = ucb(bandit, c=c, steps=steps)
            all_rewards[c] += np.array(rewards)
            all_optimal_actions[c] += np.array(optimal_actions)
    
    # 计算平均值
    for c in c_values:
        all_rewards[c] /= runs
        all_optimal_actions[c] /= runs

    # 创建子图
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制奖励曲线
    for c in c_values:
        axs[0].plot(all_rewards[c], label=f"c = {c}")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].set_title("UCB: Reward Convergence for Different c Values")
    axs[0].legend()

    # 绘制最优臂选择概率曲线
    for c in c_values:
        axs[1].plot(all_optimal_actions[c], label=f"c = {c}")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Probability of Selecting Optimal Arm")
    axs[1].set_title("UCB: Optimal Arm Convergence for Different c Values")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# 运行 UCB 实验
run_ucb_experiment()
