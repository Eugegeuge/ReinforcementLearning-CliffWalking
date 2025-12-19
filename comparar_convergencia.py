#!/usr/bin/env python3
"""
Comparación lado a lado de la convergencia de los 3 algoritmos.
Genera gráficas de curvas de aprendizaje.
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

MAX_EPISODES = 5000
SLIP_PROB = 0.1
WINDOW = 100


class SlipperyCliffWalking(gym.ActionWrapper):
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


def train_and_track(agent_type, episodes=MAX_EPISODES):
    """Entrena y guarda recompensas por episodio."""
    q_table = np.zeros((48, 4))
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), SLIP_PROB)
    alpha, gamma, epsilon = 0.1 if agent_type != 'Monte Carlo' else 0.01, 0.99, 0.1
    
    rewards = []
    history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        if agent_type == 'SARSA':
            action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[state])
            while not done and steps < 500:
                next_state, reward, term, trunc, _ = env.step(action)
                next_action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[next_state])
                td = reward + gamma * q_table[next_state, next_action] - q_table[state, action]
                q_table[state, action] += alpha * td
                state, action = next_state, next_action
                total_reward += reward
                done = term or trunc
                steps += 1
                
        elif agent_type == 'Q-Learning':
            while not done and steps < 500:
                action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[state])
                next_state, reward, term, trunc, _ = env.step(action)
                td = reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
                q_table[state, action] += alpha * td
                state = next_state
                total_reward += reward
                done = term or trunc
                steps += 1
                
        else:  # Monte Carlo
            while not done and steps < 500:
                action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[state])
                next_state, reward, term, trunc, _ = env.step(action)
                history.append((state, action, reward))
                state = next_state
                total_reward += reward
                done = term or trunc
                steps += 1
            
            # Update Monte Carlo
            G = 0
            for s, a, r in reversed(history):
                G = r + gamma * G
                q_table[s, a] += alpha * (G - q_table[s, a])
            history = []
        
        rewards.append(total_reward)
    
    env.close()
    return rewards


def main():
    print("\n" + "="*60)
    print("  COMPARACIÓN DE CONVERGENCIA")
    print("="*60)
    
    results = {}
    colors = {'SARSA': 'blue', 'Q-Learning': 'green', 'Monte Carlo': 'red'}
    
    for agent_type in ['SARSA', 'Q-Learning', 'Monte Carlo']:
        print(f"\n  Entrenando {agent_type}...", end=' ', flush=True)
        results[agent_type] = train_and_track(agent_type)
        print("✓")
    
    # Gráfico de convergencia
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Curvas de aprendizaje suavizadas
    ax1 = axes[0, 0]
    for name, rewards in results.items():
        smooth = np.convolve(rewards, np.ones(WINDOW)/WINDOW, mode='valid')
        ax1.plot(smooth, label=name, color=colors[name], alpha=0.8)
    ax1.set_title('Curvas de Aprendizaje (suavizado)', fontsize=12)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa Media')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-20, color='black', linestyle='--', alpha=0.5)
    
    # 2. Primeros 1000 episodios (velocidad de convergencia)
    ax2 = axes[0, 1]
    for name, rewards in results.items():
        smooth = np.convolve(rewards[:1000], np.ones(50)/50, mode='valid')
        ax2.plot(smooth, label=name, color=colors[name], alpha=0.8)
    ax2.set_title('Velocidad de Convergencia (primeros 1000 ep.)', fontsize=12)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Recompensa Media')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución final
    ax3 = axes[1, 0]
    data = [results[name][-500:] for name in results.keys()]
    bp = ax3.boxplot(data, labels=list(results.keys()), patch_artist=True)
    for patch, name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax3.set_title('Distribución Final (últimos 500 ep.)', fontsize=12)
    ax3.set_ylabel('Recompensa')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Tabla resumen
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['Métrica', 'SARSA', 'Q-Learning', 'Monte Carlo']]
    
    for name in ['Avg Total', 'Avg últ.100', 'Std últ.500', 'Éxito (%)']:
        row = [name]
        for agent in ['SARSA', 'Q-Learning', 'Monte Carlo']:
            r = results[agent]
            if name == 'Avg Total':
                row.append(f'{np.mean(r):.1f}')
            elif name == 'Avg últ.100':
                row.append(f'{np.mean(r[-100:]):.1f}')
            elif name == 'Std últ.500':
                row.append(f'{np.std(r[-500:]):.1f}')
            else:
                success = sum(1 for x in r if x > -50) / len(r) * 100
                row.append(f'{success:.1f}%')
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2)
    
    # Colorear encabezado
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Resumen Comparativo', fontsize=12, pad=20)
    
    plt.suptitle('Comparación de Algoritmos RL en CliffWalking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/comparacion_todos/convergencia_comparacion.png', dpi=150)
    plt.close()
    
    print(f"\n  Guardado: graphs/comparacion_todos/convergencia_comparacion.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
