#!/usr/bin/env python3
"""
Comparación Monte Carlo: Epsilon Alto, Medio, Bajo y con Decay
10K episodios
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random

N_EPISODES = 10000
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


class MonteCarloAgent:
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
        self.episode_history = []
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
        
    def update(self, state, action, reward):
        self.episode_history.append((state, action, reward))
        
    def on_episode_end(self):
        if self.episode_history:
            G = 0
            for t in range(len(self.episode_history) - 1, -1, -1):
                state, action, reward = self.episode_history[t]
                G = reward + self.gamma * G
                self.q_table[state, action] += self.alpha * (G - self.q_table[state, action])
        self.episode_history = []
        
        if self.use_decay:
            self.episode_count += 1
            self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.episode_count)


def run_experiment(env, agent, name):
    rewards = []
    epsilon_history = []
    
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        rewards.append(total_reward)
        epsilon_history.append(agent.epsilon)
        agent.on_episode_end()
    
    return {'rewards': rewards, 'epsilon': epsilon_history}


def main():
    print("\n" + "="*70)
    print("  MONTE CARLO: Comparación de Epsilon")
    print("  10K episodios | CliffWalking + Slippery(0.1)")
    print("="*70)
    
    configs = [
        ("ε=0.5 (Alto)", 0.5, False),
        ("ε=0.3 (Medio)", 0.3, False),
        ("ε=0.1 (Bajo)", 0.1, False),
        ("ε=0.01 (Muy Bajo)", 0.01, False),
        ("Decay 1.0→0.01", 1.0, True),
    ]
    
    results = {}
    
    for name, eps, use_decay in configs:
        print(f"\n  Entrenando {name}...", end=" ", flush=True)
        env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), 0.1)
        agent = MonteCarloAgent(48, 4, epsilon=eps, use_decay=use_decay, n_episodes=N_EPISODES)
        results[name] = run_experiment(env, agent, name)
        avg = np.mean(results[name]['rewards'][-100:])
        print(f"Avg(últ.100): {avg:.2f}")
        env.close()
    
    # Tabla de resultados
    print("\n" + "="*70)
    print("  RESULTADOS")
    print("="*70)
    print(f"\n{'Config':<20} | {'Avg Total':>10} | {'Avg últ.100':>12} | {'Avg últ.1000':>12} | {'Éxito':>8}")
    print("-"*70)
    
    for name, data in results.items():
        r = data['rewards']
        avg_total = np.mean(r)
        avg_100 = np.mean(r[-100:])
        avg_1000 = np.mean(r[-1000:])
        success = sum(1 for x in r if x > -50) / len(r) * 100
        print(f"{name:<20} | {avg_total:>10.2f} | {avg_100:>12.2f} | {avg_1000:>12.2f} | {success:>7.1f}%")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['red', 'orange', 'blue', 'purple', 'green']
    
    # 1. Recompensa suavizada
    ax1 = axes[0, 0]
    for (name, _), color in zip(results.items(), colors):
        smooth = np.convolve(results[name]['rewards'], np.ones(WINDOW)/WINDOW, mode='valid')
        ax1.plot(smooth, label=name, color=color, alpha=0.8)
    ax1.set_title('Recompensa Media (ventana=100)', fontsize=12)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolución de Epsilon
    ax2 = axes[0, 1]
    for (name, _), color in zip(results.items(), colors):
        ax2.plot(results[name]['epsilon'], label=name, color=color, alpha=0.8)
    ax2.set_title('Evolución de Epsilon', fontsize=12)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Epsilon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Histograma recompensas (últimos 1000)
    ax3 = axes[1, 0]
    for (name, _), color in zip(results.items(), colors):
        ax3.hist(results[name]['rewards'][-1000:], bins=40, alpha=0.4, label=name, color=color)
    ax3.set_title('Distribución Recompensas (últ. 1000 ep.)', fontsize=12)
    ax3.set_xlabel('Recompensa')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Barras comparativas
    ax4 = axes[1, 1]
    names = list(results.keys())
    avg_100_vals = [np.mean(results[n]['rewards'][-100:]) for n in names]
    avg_1000_vals = [np.mean(results[n]['rewards'][-1000:]) for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax4.bar(x - width/2, avg_100_vals, width, label='Avg últ.100', color='steelblue')
    ax4.bar(x + width/2, avg_1000_vals, width, label='Avg últ.1000', color='lightcoral')
    ax4.set_ylabel('Recompensa')
    ax4.set_title('Comparación Final', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Alto\n(0.5)', 'Medio\n(0.3)', 'Bajo\n(0.1)', 'Muy Bajo\n(0.01)', 'Decay\n(1→0.01)'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Monte Carlo: Efecto del Epsilon', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('montecarlo_epsilon_comparison.png', dpi=150)
    plt.close()
    
    print(f"\n  Guardado: montecarlo_epsilon_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
