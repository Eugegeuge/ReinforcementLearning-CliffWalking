import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.agent import SarsaAgent

def train_agent(env, alpha, epsilon, n_episodes=500):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = SarsaAgent(n_states, n_actions, alpha=alpha, epsilon=epsilon)
    
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 500:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
    return rewards

def main():
    env = gym.make('CliffWalking-v1', is_slippery=True)
    
    alphas = [0.1, 0.5, 0.9]
    epsilons = [0.1, 0.3, 0.5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Study Alpha (fixed epsilon=0.1)
    print("Studying Alpha...")
    for alpha in alphas:
        rewards = train_agent(env, alpha, epsilon=0.1)
        # Smooth
        smooth = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax1.plot(smooth, label=f"alpha={alpha}")
    
    ax1.set_title("Efecto de Alpha (Epsilon=0.1)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa Suavizada")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Study Epsilon (fixed alpha=0.5)
    print("Studying Epsilon...")
    for eps in epsilons:
        rewards = train_agent(env, alpha=0.5, epsilon=eps)
        smooth = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax2.plot(smooth, label=f"epsilon={eps}")
        
    ax2.set_title("Efecto de Epsilon (Alpha=0.5)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Recompensa Suavizada")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("parameter_study.png")
    print("Saved 'parameter_study.png'")

if __name__ == "__main__":
    main()
