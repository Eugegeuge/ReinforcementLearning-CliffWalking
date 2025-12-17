#!/usr/bin/env python3
"""
Comparación completa: SARSA, Q-Learning y Monte Carlo
Con y sin Epsilon Decay - 10K episodios
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random

N_EPISODES = 10000
PRINT_INTERVAL = 2000
WINDOW = 100
MAX_STEPS = 500


class SlipperyCliffWalking(gym.ActionWrapper):
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


class RLAgent:
    """Clase base con opción de epsilon decay."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, 
                 epsilon=0.1, use_decay=False, min_epsilon=0.01, n_episodes=10000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.use_decay = use_decay
        self.decay_rate = 5 / n_episodes if use_decay else 0
        self.episode_count = 0
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def decay_epsilon(self):
        if self.use_decay:
            self.episode_count += 1
            self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.episode_count)
    
    def update(self, state, action, reward, next_state, next_action=None):
        raise NotImplementedError
    
    def on_episode_end(self):
        self.decay_epsilon()


class SarsaAgent(RLAgent):
    def update(self, state, action, reward, next_state, next_action):
        next_q = self.q_table[next_state, next_action] if next_action is not None else 0
        td_target = reward + self.gamma * next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])


class QLearningAgent(RLAgent):
    def update(self, state, action, reward, next_state, next_action=None):
        max_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])


class MonteCarloAgent(RLAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_history = []
        
    def update(self, state, action, reward, next_state, next_action=None):
        self.episode_history.append((state, action, reward))
        
    def on_episode_end(self):
        if self.episode_history:
            G = 0
            for t in range(len(self.episode_history) - 1, -1, -1):
                state, action, reward = self.episode_history[t]
                G = reward + self.gamma * G
                self.q_table[state, action] += self.alpha * (G - self.q_table[state, action])
        self.episode_history = []
        self.decay_epsilon()


def run_experiment(env, agent, agent_name):
    rewards = []
    print(f"  Entrenando {agent_name}...", end=" ", flush=True)
    start = time.time()
    
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        action = agent.choose_action(state)
        
        while not done and steps < MAX_STEPS:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        rewards.append(total_reward)
        agent.on_episode_end()
    
    elapsed = time.time() - start
    avg_last = np.mean(rewards[-100:])
    print(f"Avg(últ.100): {avg_last:>8.2f} | {elapsed:.1f}s")
    return rewards


def main():
    print("\n" + "="*75)
    print("  COMPARACIÓN: SARSA vs Q-Learning vs Monte Carlo")
    print("  Con y Sin Epsilon Decay | 10K episodios")
    print("="*75)
    
    n_states = 48
    n_actions = 4
    
    # Configuraciones
    configs = [
        ("Sin Decay (ε=0.1)", False, 0.1),
        ("Con Decay (1.0→0.01)", True, 1.0),
    ]
    
    agent_types = [
        ("SARSA", SarsaAgent),
        ("Q-Learning", QLearningAgent),
        ("Monte Carlo", MonteCarloAgent),
    ]
    
    all_results = {}
    
    for config_name, use_decay, init_eps in configs:
        print(f"\n--- {config_name} ---")
        for agent_name, AgentClass in agent_types:
            env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), 0.1)
            agent = AgentClass(
                n_states, n_actions, alpha=0.1, gamma=0.99,
                epsilon=init_eps, use_decay=use_decay, 
                min_epsilon=0.01, n_episodes=N_EPISODES
            )
            key = f"{agent_name} {config_name}"
            all_results[key] = run_experiment(env, agent, agent_name)
            env.close()
    
    # Tabla de resultados
    print("\n" + "="*75)
    print("  RESULTADOS FINALES")
    print("="*75)
    print(f"\n{'Modelo':<35} | {'Avg(Total)':>10} | {'Avg(últ.100)':>12} | {'Éxito':>8}")
    print("-"*75)
    
    for key, rewards in all_results.items():
        avg_total = np.mean(rewards)
        avg_last = np.mean(rewards[-100:])
        success = sum(1 for r in rewards if r > -50) / len(rewards) * 100
        print(f"{key:<35} | {avg_total:>10.2f} | {avg_last:>12.2f} | {success:>7.1f}%")
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'SARSA': 'blue', 'Q-Learning': 'orange', 'Monte Carlo': 'green'}
    
    for i, (config_name, _, _) in enumerate(configs):
        ax = axes[0, 0] if i == 0 else axes[1, 0]
        for agent_name, _ in agent_types:
            key = f"{agent_name} {config_name}"
            smooth = np.convolve(all_results[key], np.ones(WINDOW)/WINDOW, mode='valid')
            ax.plot(smooth, label=agent_name, color=colors[agent_name], alpha=0.8)
        ax.set_title(f'Recompensa {config_name}', fontsize=11)
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Recompensa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Comparación por algoritmo
    for j, (agent_name, _) in enumerate(agent_types):
        ax = axes[0, j] if j < 3 else axes[1, j-3]
        if j >= 1:
            ax = axes[0, j]
        for config_name, _, _ in configs:
            key = f"{agent_name} {config_name}"
            smooth = np.convolve(all_results[key], np.ones(WINDOW)/WINDOW, mode='valid')
            label = "Con Decay" if "Con" in config_name else "Sin Decay"
            ax.plot(smooth, label=label, alpha=0.8)
        ax.set_title(f'{agent_name}: Decay vs No Decay', fontsize=11)
        ax.set_xlabel('Episodio')
        ax.set_ylabel('Recompensa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Histogramas
    for j, (agent_name, _) in enumerate(agent_types):
        ax = axes[1, j]
        for config_name, _, _ in configs:
            key = f"{agent_name} {config_name}"
            label = "Con Decay" if "Con" in config_name else "Sin Decay"
            ax.hist(all_results[key][-1000:], bins=30, alpha=0.5, label=label)
        ax.set_title(f'{agent_name}: Distribución (últ.1000)', fontsize=11)
        ax.set_xlabel('Recompensa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparación: Con vs Sin Epsilon Decay (10K episodios)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('full_comparison_decay.png', dpi=150)
    plt.close()
    
    print(f"\n  Guardado: full_comparison_decay.png")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
