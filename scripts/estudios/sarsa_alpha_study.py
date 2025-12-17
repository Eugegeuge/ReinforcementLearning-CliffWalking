#!/usr/bin/env python3
"""
SARSA: Comparación de Alpha (Learning Rate)
10K episodios
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


class SarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
        
    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])


def run_experiment(env, agent):
    rewards = []
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        rewards.append(total_reward)
    return rewards


def main():
    print("\n" + "="*70)
    print("  SARSA: Comparación de Alpha (Learning Rate)")
    print("  10K episodios | ε=0.1, γ=0.99")
    print("="*70)
    
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]
    results = {}
    
    for alpha in alphas:
        name = f"α={alpha}"
        print(f"\n  Entrenando {name}...", end=" ", flush=True)
        env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), 0.1)
        agent = SarsaAgent(48, 4, alpha=alpha, gamma=0.99, epsilon=0.1)
        results[name] = run_experiment(env, agent)
        avg = np.mean(results[name][-100:])
        print(f"Avg(últ.100): {avg:.2f}")
        env.close()
    
    # Tabla de resultados
    print("\n" + "="*70)
    print("  RESULTADOS")
    print("="*70)
    print(f"\n{'Alpha':<10} | {'Avg Total':>10} | {'Avg últ.100':>12} | {'Avg últ.1000':>12} | {'Éxito':>8}")
    print("-"*70)
    
    for name, r in results.items():
        avg_total = np.mean(r)
        avg_100 = np.mean(r[-100:])
        avg_1000 = np.mean(r[-1000:])
        success = sum(1 for x in r if x > -50) / len(r) * 100
        print(f"{name:<10} | {avg_total:>10.2f} | {avg_100:>12.2f} | {avg_1000:>12.2f} | {success:>7.1f}%")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    
    # 1. Recompensa suavizada
    ax1 = axes[0, 0]
    for (name, r), color in zip(results.items(), colors):
        smooth = np.convolve(r, np.ones(WINDOW)/WINDOW, mode='valid')
        ax1.plot(smooth, label=name, color=color, alpha=0.8)
    ax1.set_title('Recompensa Media (ventana=100)', fontsize=12)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoom primeros 2000 episodios
    ax2 = axes[0, 1]
    for (name, r), color in zip(results.items(), colors):
        smooth = np.convolve(r[:2000], np.ones(WINDOW)/WINDOW, mode='valid')
        ax2.plot(smooth, label=name, color=color, alpha=0.8)
    ax2.set_title('Convergencia Inicial (primeros 2000 ep.)', fontsize=12)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Recompensa')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Barras comparativas
    ax3 = axes[1, 0]
    names = list(results.keys())
    avg_100_vals = [np.mean(results[n][-100:]) for n in names]
    avg_1000_vals = [np.mean(results[n][-1000:]) for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, avg_100_vals, width, label='Avg últ.100', color='steelblue')
    ax3.bar(x + width/2, avg_1000_vals, width, label='Avg últ.1000', color='lightcoral')
    ax3.set_ylabel('Recompensa')
    ax3.set_title('Comparación Final por Alpha', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{a}' for a in alphas])
    ax3.set_xlabel('Alpha')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Desviación estándar
    ax4 = axes[1, 1]
    stds = [np.std(results[n][-1000:]) for n in names]
    ax4.bar(names, stds, color=colors)
    ax4.set_title('Estabilidad (Desv. Est. últ.1000)', fontsize=12)
    ax4.set_xlabel('Alpha')
    ax4.set_ylabel('Desviación Estándar')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('SARSA: Efecto del Alpha (Learning Rate)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/sarsa/sarsa_alpha_comparison.png', dpi=150)
    plt.close()
    
    print(f"\n  Guardado: graphs/sarsa/sarsa_alpha_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
