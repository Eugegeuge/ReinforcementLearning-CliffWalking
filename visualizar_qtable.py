#!/usr/bin/env python3
"""
Visualiza la Q-Table de cada algoritmo como heatmap.
Muestra qué tan valiosa considera el agente cada acción en cada estado.
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


def create_agent(agent_type):
    q_table = np.zeros((48, 4))
    return {'type': agent_type, 'q_table': q_table, 'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1}


def choose_action(agent, state):
    if np.random.rand() < agent['epsilon']:
        return np.random.randint(4)
    return np.argmax(agent['q_table'][state])


def train(agent, episodes=TRAIN_EPISODES):
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), SLIP_PROB)
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        if agent['type'] == 'SARSA':
            action = choose_action(agent, state)
            while not done and steps < 500:
                next_state, reward, term, trunc, _ = env.step(action)
                next_action = choose_action(agent, next_state)
                td = reward + agent['gamma'] * agent['q_table'][next_state, next_action] - agent['q_table'][state, action]
                agent['q_table'][state, action] += agent['alpha'] * td
                state, action = next_state, next_action
                done = term or trunc
                steps += 1
        elif agent['type'] == 'Q-Learning':
            while not done and steps < 500:
                action = choose_action(agent, state)
                next_state, reward, term, trunc, _ = env.step(action)
                td = reward + agent['gamma'] * np.max(agent['q_table'][next_state]) - agent['q_table'][state, action]
                agent['q_table'][state, action] += agent['alpha'] * td
                state = next_state
                done = term or trunc
                steps += 1
        else:  # Monte Carlo
            history = []
            while not done and steps < 500:
                action = choose_action(agent, state)
                next_state, reward, term, trunc, _ = env.step(action)
                history.append((state, action, reward))
                state = next_state
                done = term or trunc
                steps += 1
            # Actualizar al final del episodio
            G = 0
            for s, a, r in reversed(history):
                G = r + agent['gamma'] * G
                agent['q_table'][s, a] += agent['alpha'] * (G - agent['q_table'][s, a])
    
    env.close()
    return agent


def visualize_qtable(agent, filename):
    """Crea visualización de la Q-table como heatmap."""
    q_table = agent['q_table'].reshape(4, 12, 4)
    action_names = ['↑ Arriba', '→ Derecha', '↓ Abajo', '← Izquierda']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (ax, action_name) in enumerate(zip(axes.flat, action_names)):
        data = q_table[:, :, idx]
        
        # Enmascarar el acantilado
        masked = np.ma.array(data)
        for c in range(1, 11):
            masked[3, c] = np.ma.masked
        
        im = ax.imshow(masked, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=0)
        ax.set_title(f'{action_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')
        
        # Añadir valores
        for i in range(4):
            for j in range(12):
                if i == 3 and 0 < j < 11:
                    ax.text(j, i, '💀', ha='center', va='center', fontsize=10)
                elif i == 3 and j == 11:
                    ax.text(j, i, '🏁', ha='center', va='center', fontsize=10)
                else:
                    val = data[i, j]
                    color = 'white' if val < -50 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)
        
        ax.set_xticks(range(12))
        ax.set_yticks(range(4))
    
    plt.suptitle(f'Q-Table: {agent["type"]}\n(Valor Q por acción en cada estado)', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes, shrink=0.6, label='Valor Q')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Guardado: {filename}")


def main():
    print("\n" + "="*60)
    print("  VISUALIZACIÓN DE Q-TABLES")
    print("="*60)
    
    for agent_type in ['SARSA', 'Q-Learning', 'Monte Carlo']:
        print(f"\n  Entrenando {agent_type}...", end=' ', flush=True)
        agent = create_agent(agent_type)
        episodes = 10000 if agent_type == 'Monte Carlo' else 5000
        train(agent, episodes)
        print("✓")
        
        folder = agent_type.lower().replace("-", "").replace(" ", "")
        filename = f'graphs/{folder}/qtable_visualization.png'
        
        # Crear carpeta si no existe
        import os
        os.makedirs(f'graphs/{folder}', exist_ok=True)
        
        visualize_qtable(agent, filename)
    
    print("\n  ✅ Visualizaciones completadas!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
