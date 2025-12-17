#!/usr/bin/env python3
"""
Comparación: Monte Carlo con Epsilon Decay vs Sin Epsilon Decay
10K episodios para demostrar la importancia del decaimiento de epsilon
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random

N_EPISODES = 10000
PRINT_INTERVAL = 1000
WINDOW = 100
MAX_STEPS = 500


class SlipperyCliffWalking(gym.ActionWrapper):
    """Wrapper que hace el entorno CliffWalking 'resbaladizo'."""
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


class MonteCarloAgent:
    """Monte Carlo con opción de epsilon decay."""
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
        
    def update(self, state, action, reward, next_state, next_action=None):
        self.episode_history.append((state, action, reward))
        
    def on_episode_end(self):
        if len(self.episode_history) > 0:
            G = 0
            for t in range(len(self.episode_history) - 1, -1, -1):
                state, action, reward = self.episode_history[t]
                G = reward + self.gamma * G
                self.q_table[state, action] += self.alpha * (G - self.q_table[state, action])
        
        # Epsilon decay solo si está activado
        if self.use_decay:
            self.episode_count += 1
            self.epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.episode_count)
                
        self.episode_history = []


def run_experiment(env, agent, agent_name, n_episodes=N_EPISODES):
    """Ejecuta el experimento."""
    rewards = []
    epsilon_history = []
    
    print(f"\n{'='*60}")
    print(f"  {agent_name}")
    print(f"  Epsilon inicial: {agent.epsilon:.2f}, Decay: {agent.use_decay}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, None)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        rewards.append(total_reward)
        epsilon_history.append(agent.epsilon)
        agent.on_episode_end()
        
        if (episode + 1) % PRINT_INTERVAL == 0:
            avg = np.mean(rewards[-WINDOW:])
            print(f"  Ep {episode+1:>5} | Avg: {avg:>8.2f} | Eps: {agent.epsilon:.4f}")
    
    elapsed = time.time() - start_time
    print(f"  Completado en {elapsed:.2f}s")
    
    return {'rewards': rewards, 'epsilon': epsilon_history, 'time': elapsed}


def print_policy(agent, name):
    """Imprime la política aprendida."""
    actions = ['^', '>', 'v', '<']
    print(f"\nPolítica {name}:")
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


def main():
    print("\n" + "="*70)
    print("  COMPARACIÓN: MONTE CARLO CON vs SIN EPSILON DECAY")
    print("  Episodios: 10,000 | Entorno: CliffWalking + Slippery(0.1)")
    print("="*70)
    
    # Crear entorno
    base_env = gym.make('CliffWalking-v1')
    env = SlipperyCliffWalking(base_env, slip_probability=0.1)
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Agente SIN epsilon decay (epsilon fijo = 0.1)
    agent_fixed = MonteCarloAgent(
        n_states, n_actions, 
        alpha=0.1, gamma=0.99, epsilon=0.1,
        use_decay=False
    )
    
    # Agente CON epsilon decay (1.0 -> 0.01)
    agent_decay = MonteCarloAgent(
        n_states, n_actions, 
        alpha=0.1, gamma=0.99, epsilon=1.0,
        use_decay=True, min_epsilon=0.01, n_episodes=N_EPISODES
    )
    
    # Ejecutar experimentos
    results_fixed = run_experiment(env, agent_fixed, "MC Sin Decay (ε=0.1 fijo)")
    
    # Resetear entorno para segundo agente
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), slip_probability=0.1)
    results_decay = run_experiment(env, agent_decay, "MC Con Decay (ε: 1.0 → 0.01)")
    
    # Mostrar políticas
    print_policy(agent_fixed, "SIN Epsilon Decay")
    print_policy(agent_decay, "CON Epsilon Decay")
    
    # Estadísticas finales
    print("\n" + "="*70)
    print("  ESTADÍSTICAS FINALES")
    print("="*70)
    
    for name, results in [("Sin Decay (ε=0.1)", results_fixed), 
                          ("Con Decay (1.0→0.01)", results_decay)]:
        r = results['rewards']
        print(f"\n  {name}:")
        print(f"    Recompensa Media Total:        {np.mean(r):>10.2f}")
        print(f"    Recompensa Media (últ. 100):   {np.mean(r[-100:]):>10.2f}")
        print(f"    Recompensa Media (últ. 1000):  {np.mean(r[-1000:]):>10.2f}")
        print(f"    Recompensa Máxima:             {np.max(r):>10.2f}")
        print(f"    Recompensa Mínima:             {np.min(r):>10.2f}")
        print(f"    Tasa Éxito (reward > -50):     {sum(1 for x in r if x > -50)/len(r)*100:>10.1f}%")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Recompensas suavizadas
    ax1 = axes[0, 0]
    smooth_fixed = np.convolve(results_fixed['rewards'], np.ones(WINDOW)/WINDOW, mode='valid')
    smooth_decay = np.convolve(results_decay['rewards'], np.ones(WINDOW)/WINDOW, mode='valid')
    ax1.plot(smooth_fixed, label='Sin Decay (ε=0.1)', color='red', alpha=0.8)
    ax1.plot(smooth_decay, label='Con Decay (1.0→0.01)', color='green', alpha=0.8)
    ax1.set_title('Recompensa Media (ventana=100)', fontsize=12)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolución de Epsilon
    ax2 = axes[0, 1]
    ax2.plot(results_fixed['epsilon'], label='Sin Decay', color='red', alpha=0.8)
    ax2.plot(results_decay['epsilon'], label='Con Decay', color='green', alpha=0.8)
    ax2.set_title('Evolución de Epsilon', fontsize=12)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Epsilon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Histograma de recompensas (últimos 1000)
    ax3 = axes[1, 0]
    ax3.hist(results_fixed['rewards'][-1000:], bins=50, alpha=0.5, label='Sin Decay', color='red')
    ax3.hist(results_decay['rewards'][-1000:], bins=50, alpha=0.5, label='Con Decay', color='green')
    ax3.set_title('Distribución Recompensas (últimos 1000 ep.)', fontsize=12)
    ax3.set_xlabel('Recompensa')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Recompensa acumulada
    ax4 = axes[1, 1]
    ax4.plot(np.cumsum(results_fixed['rewards']), label='Sin Decay', color='red', alpha=0.8)
    ax4.plot(np.cumsum(results_decay['rewards']), label='Con Decay', color='green', alpha=0.8)
    ax4.set_title('Recompensa Acumulada', fontsize=12)
    ax4.set_xlabel('Episodio')
    ax4.set_ylabel('Recompensa Acumulada')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Monte Carlo: Comparación Epsilon Decay vs Fijo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('epsilon_decay_comparison.png', dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("  Guardado: epsilon_decay_comparison.png")
    print("="*70 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()
