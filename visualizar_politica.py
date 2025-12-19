#!/usr/bin/env python3
"""
Visualiza la política aprendida con flechas en el grid.
Compara SARSA vs Q-Learning lado a lado.
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

TRAIN_EPISODES = 5000
SLIP_PROB = 0.1


class SlipperyCliffWalking(gym.ActionWrapper):
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


def train_agent(agent_type, episodes=TRAIN_EPISODES):
    q_table = np.zeros((48, 4))
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), SLIP_PROB)
    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        if agent_type == 'SARSA':
            action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[state])
            while not done and steps < 500:
                next_state, reward, term, trunc, _ = env.step(action)
                next_action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[next_state])
                td = reward + gamma * q_table[next_state, next_action] - q_table[state, action]
                q_table[state, action] += alpha * td
                state, action = next_state, next_action
                done = term or trunc
                steps += 1
        else:
            while not done and steps < 500:
                action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[state])
                next_state, reward, term, trunc, _ = env.step(action)
                td = reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
                q_table[state, action] += alpha * td
                state = next_state
                done = term or trunc
                steps += 1
    
    env.close()
    return q_table


def visualize_policy(q_tables, filename):
    """Visualiza políticas lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Flechas para cada acción
    arrows = {0: (0, 0.3), 1: (0.3, 0), 2: (0, -0.3), 3: (-0.3, 0)}  # Arriba, Derecha, Abajo, Izquierda
    
    for ax, (name, q_table) in zip(axes, q_tables.items()):
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.5, 3.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # Dibujar grid
        for i in range(5):
            ax.axhline(y=i-0.5, color='black', linewidth=1)
        for j in range(13):
            ax.axvline(x=j-0.5, color='black', linewidth=1)
        
        # Colorear celdas especiales
        ax.fill_between([0.5, 10.5], [2.5, 2.5], [3.5, 3.5], color='red', alpha=0.3)  # Cliff
        ax.fill_between([-0.5, 0.5], [2.5, 2.5], [3.5, 3.5], color='green', alpha=0.3)  # Start
        ax.fill_between([10.5, 11.5], [2.5, 2.5], [3.5, 3.5], color='blue', alpha=0.3)  # Goal
        
        # Dibujar flechas y etiquetas
        for row in range(4):
            for col in range(12):
                state = row * 12 + col
                
                if row == 3 and 0 < col < 11:  # Cliff
                    ax.text(col, row, '💀', ha='center', va='center', fontsize=14)
                elif row == 3 and col == 0:  # Start
                    ax.text(col, row, 'S', ha='center', va='center', fontsize=16, fontweight='bold', color='green')
                elif row == 3 and col == 11:  # Goal
                    ax.text(col, row, 'G', ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
                else:
                    best_action = np.argmax(q_table[state])
                    dx, dy = arrows[best_action]
                    ax.arrow(col - dx/2, row - dy/2, dx, dy, head_width=0.15, head_length=0.1, 
                            fc='darkgreen', ec='darkgreen', linewidth=2)
        
        ax.set_title(f'Política: {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')
        ax.set_xticks(range(12))
        ax.set_yticks(range(4))
    
    plt.suptitle('Comparación de Políticas Aprendidas\n(Flechas indican la mejor acción en cada estado)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Guardado: {filename}")


def main():
    print("\n" + "="*60)
    print("  VISUALIZACIÓN DE POLÍTICAS")
    print("="*60)
    
    q_tables = {}
    for agent_type in ['SARSA', 'Q-Learning']:
        print(f"\n  Entrenando {agent_type}...", end=' ', flush=True)
        q_tables[agent_type] = train_agent(agent_type)
        print("✓")
    
    visualize_policy(q_tables, 'graphs/comparacion_todos/politicas_comparacion.png')
    
    print("\n  ✅ Visualización completada!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
