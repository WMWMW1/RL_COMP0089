import numpy as np
import matplotlib.pyplot as plt
from bandit_environment import KArmedBandit

def epsilon_greedy(bandit, epsilon=0.1, steps=1000):
    """
    Run the ε-greedy strategy
    bandit: KArmedBandit environment
    epsilon: exploration probability
    steps: number of steps to run
    Returns:
    - rewards: rewards at each step
    - optimal_action_count: number of times the optimal arm is chosen at each step
    """
    bandit.reset()  # Reset the environment
    rewards = []
    optimal_action_count = []
    
    for step in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.choice(bandit.k)  # Random exploration
        else:
            action = np.argmax(bandit.q_estimates)  # Choose the arm with the highest current estimate
        
        reward, is_optimal = bandit.step(action)
        rewards.append(reward)
        optimal_action_count.append(is_optimal)

        # Update estimates (incremental update formula)
        bandit.q_estimates[action] += (reward - bandit.q_estimates[action]) / (bandit.action_counts[action])

    return rewards, optimal_action_count

def run_experiment(k=10, steps=1000, runs=200):
    """
    Run the Bandit experiment, testing different ε values
    k: number of arms
    steps: number of steps to run
    runs: number of experiment repetitions (to average results)
    """
    epsilons = [0, 0.1, 0.01]  # ε values
    all_rewards = {eps: np.zeros(steps) for eps in epsilons}
    all_optimal_actions = {eps: np.zeros(steps) for eps in epsilons}

    for _ in range(runs):
        bandit = KArmedBandit(k=k)  # Create a new Bandit for each experiment

        for eps in epsilons:
            rewards, optimal_actions = epsilon_greedy(bandit, epsilon=eps, steps=steps)
            all_rewards[eps] += np.array(rewards)
            all_optimal_actions[eps] += np.array(optimal_actions)
    
    # Calculate averages
    for eps in epsilons:
        all_rewards[eps] /= runs
        all_optimal_actions[eps] /= runs
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot reward curves
    for eps in epsilons:
        axs[0].plot(all_rewards[eps], label=f"ε = {eps}")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Average Reward")
    axs[0].set_title("ε-Greedy: Reward Convergence for Different ε Values")
    axs[0].legend()

    # Plot optimal action selection probability curves
    for eps in epsilons:
        axs[1].plot(all_optimal_actions[eps], label=f"ε = {eps}")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Probability of Selecting Optimal Arm")
    axs[1].set_title("ε-Greedy: Optimal Arm Convergence for Different ε Values")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Run the experiment
run_experiment()
