#!/usr/bin/env python3
"""
Q-LEARNING ÓPTIMO para CliffWalking
Parámetros optimizados basados en estudios de hiperparámetros
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
import json

# PARÁMETROS ÓPTIMOS
N_EPISODES = 10000
ALPHA = 0.1       # Learning rate óptimo
GAMMA = 0.99      # Factor descuento alto
EPSILON = 0.01    # Exploración baja
MAX_STEPS = 500
WINDOW = 100


class SlipperyCliffWalking(gym.ActionWrapper):
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


class QLearningOptimo:
    """Q-Learning con parámetros óptimos."""
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
        
    def update(self, state, action, reward, next_state):
        max_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])


def print_policy(agent):
    actions = ['^', '>', 'v', '<']
    print("\n  POLÍTICA APRENDIDA:")
    print("  " + "-"*25)
    for row in range(4):
        line = f"  {row}: "
        for col in range(12):
            s = row * 12 + col
            if row == 3 and 0 < col < 11:
                line += "C "
            elif s == 47:
                line += "G "
            else:
                line += actions[np.argmax(agent.q_table[s])] + " "
        print(line)
    print("  " + "-"*25)


def main():
    print("\n" + "="*70)
    print("  Q-LEARNING ÓPTIMO - ENTRENAMIENTO")
    print("="*70)
    print(f"""
  Parámetros Óptimos:
  ├─ Alpha (α):    {ALPHA} (balance velocidad/estabilidad)
  ├─ Gamma (γ):    {GAMMA} (alto valor para recompensas futuras)
  ├─ Epsilon (ε):  {EPSILON} (mínima exploración)
  ├─ Episodios:    {N_EPISODES}
  └─ Max Steps:    {MAX_STEPS}
    """)
    
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), 0.1)
    agent = QLearningOptimo(48, 4)
    
    rewards = []
    steps_list = []
    
    print("  Entrenando...\n")
    start = time.time()
    
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        rewards.append(total_reward)
        steps_list.append(steps)
        
        if (episode + 1) % 2000 == 0:
            avg = np.mean(rewards[-WINDOW:])
            print(f"  Ep {episode+1:>5}/{N_EPISODES} | Avg(últ.{WINDOW}): {avg:>8.2f}")
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("  RESULTADOS FINALES")
    print("="*70)
    
    stats = {
        'avg_total': float(np.mean(rewards)),
        'avg_100': float(np.mean(rewards[-100:])),
        'avg_1000': float(np.mean(rewards[-1000:])),
        'max': float(np.max(rewards)),
        'min': float(np.min(rewards)),
        'std': float(np.std(rewards[-1000:])),
        'success_rate': float(sum(1 for r in rewards if r > -50) / len(rewards) * 100),
        'time': float(elapsed)
    }
    
    print(f"""
  Recompensa Media Total:    {stats['avg_total']:>10.2f}
  Recompensa Media últ.100:  {stats['avg_100']:>10.2f}
  Recompensa Media últ.1000: {stats['avg_1000']:>10.2f}
  Recompensa Máxima:         {stats['max']:>10.2f}
  Recompensa Mínima:         {stats['min']:>10.2f}
  Desviación Estándar:       {stats['std']:>10.2f}
  Tasa de Éxito:             {stats['success_rate']:>10.1f}%
  Tiempo de Entrenamiento:   {stats['time']:>10.2f}s
    """)
    
    print_policy(agent)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    smooth = np.convolve(rewards, np.ones(WINDOW)/WINDOW, mode='valid')
    ax1.plot(smooth, color='green', alpha=0.8)
    ax1.axhline(y=-13, color='red', linestyle='--', label='Óptimo teórico (~-13)')
    ax1.set_title('Curva de Aprendizaje', fontsize=12)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(rewards[-1000:], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=stats['avg_1000'], color='red', linestyle='--', label=f'Media: {stats["avg_1000"]:.1f}')
    ax2.set_title('Distribución Recompensas (últ. 1000)', fontsize=12)
    ax2.set_xlabel('Recompensa')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    smooth_steps = np.convolve(steps_list, np.ones(WINDOW)/WINDOW, mode='valid')
    ax3.plot(smooth_steps, color='orange', alpha=0.8)
    ax3.axhline(y=13, color='red', linestyle='--', label='Camino óptimo (13 pasos)')
    ax3.set_title('Pasos por Episodio', fontsize=12)
    ax3.set_xlabel('Episodio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ['Parámetro', 'Valor', 'Justificación'],
        ['Alpha (α)', str(ALPHA), 'Balance velocidad/estabilidad'],
        ['Gamma (γ)', str(GAMMA), 'Alto = Planifica'],
        ['Epsilon (ε)', str(EPSILON), 'Bajo = Explota'],
        ['Episodios', str(N_EPISODES), 'Convergencia rápida'],
        ['', '', ''],
        ['Avg últ.100', f'{stats["avg_100"]:.2f}', ''],
        ['Tasa Éxito', f'{stats["success_rate"]:.1f}%', ''],
    ]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Resumen', fontsize=12, pad=20)
    
    plt.suptitle('Q-Learning Óptimo - Resultados', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs/qlearning/qlearning_optimo_resultados.png', dpi=150)
    plt.close()
    
    with open('graphs/qlearning/qlearning_optimo_stats.json', 'w') as f:
        json.dump({'params': {'alpha': ALPHA, 'gamma': GAMMA, 'epsilon': EPSILON, 'episodes': N_EPISODES}, 'stats': stats}, f, indent=2)
    
    print(f"\n  Guardado: graphs/qlearning/qlearning_optimo_resultados.png")
    print(f"  Guardado: graphs/qlearning/qlearning_optimo_stats.json")
    print("="*70 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()
