#!/usr/bin/env python3
"""
Script completo de entrenamiento con 15K episodios.
Imprime toda la información posible incluyendo evolución de recompensas.
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.agent import SarsaAgent, QLearningAgent, MonteCarloAgent
from src.utils import plot_metrics, print_policy
import time
import random

N_EPISODES = 15000
PRINT_INTERVAL = 500  # Imprimir progreso cada X episodios
WINDOW = 100  # Ventana para suavizado


class SlipperyCliffWalking(gym.ActionWrapper):
    """Wrapper que hace el entorno CliffWalking 'resbaladizo' (igual que el notebook)."""
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        # Con probabilidad slip_probability, ignoramos la acción elegida y tomamos una aleatoria
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


def run_experiment_verbose(env, agent, agent_name, n_episodes=N_EPISODES, max_steps=500):
    """Ejecuta el experimento con output detallado."""
    rewards = []
    steps_list = []
    
    print(f"\n{'='*60}")
    print(f"  ENTRENANDO: {agent_name}")
    print(f"  Episodios: {n_episodes}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
        rewards.append(total_reward)
        steps_list.append(steps)
        agent.on_episode_end()
        
        # Imprimir progreso
        if (episode + 1) % PRINT_INTERVAL == 0 or episode == 0:
            avg_reward = np.mean(rewards[-WINDOW:]) if len(rewards) >= WINDOW else np.mean(rewards)
            avg_steps = np.mean(steps_list[-WINDOW:]) if len(steps_list) >= WINDOW else np.mean(steps_list)
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            remaining = (n_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
            
            # Mostrar epsilon actual para MC
            eps_info = ""
            if hasattr(agent, 'epsilon') and agent_name == 'Monte Carlo':
                eps_info = f" | Eps: {agent.epsilon:.3f}"
            
            print(f"  Ep {episode+1:>6}/{n_episodes} | "
                  f"Reward: {total_reward:>7.1f} | "
                  f"Avg(últimos {WINDOW}): {avg_reward:>7.2f} | "
                  f"Steps: {steps:>4}{eps_info} | "
                  f"ETA: {remaining/60:>5.1f}min")
    
    elapsed = time.time() - start_time
    
    print(f"\n  COMPLETADO en {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"  Velocidad: {n_episodes/elapsed:.1f} episodios/seg")
    
    return {'rewards': rewards, 'steps': steps_list, 'time': elapsed}


def print_reward_evolution(all_metrics, intervals=[1000, 5000, 10000, 15000]):
    """Imprime la evolución de recompensas en intervalos clave."""
    print(f"\n{'='*80}")
    print("  EVOLUCIÓN DE RECOMPENSA MEDIA (ventana de 100 episodios)")
    print(f"{'='*80}")
    
    header = f"{'Episodio':<12}"
    for name in all_metrics.keys():
        header += f" | {name:>15}"
    print(header)
    print("-" * len(header))
    
    for ep in intervals:
        if ep > N_EPISODES:
            continue
        row = f"{ep:<12}"
        for name, data in all_metrics.items():
            rewards = data['rewards']
            if ep <= len(rewards):
                # Promedio de los últimos 100 episodios hasta ese punto
                start_idx = max(0, ep - WINDOW)
                avg = np.mean(rewards[start_idx:ep])
                row += f" | {avg:>15.2f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)


def print_final_statistics(all_metrics):
    """Imprime estadísticas finales detalladas."""
    print(f"\n{'='*80}")
    print("  ESTADÍSTICAS FINALES")
    print(f"{'='*80}")
    
    for name, data in all_metrics.items():
        rewards = data['rewards']
        steps = data['steps']
        
        print(f"\n  {name}:")
        print(f"    Recompensa Media (total):          {np.mean(rewards):>10.2f}")
        print(f"    Recompensa Media (últimos 100):    {np.mean(rewards[-100:]):>10.2f}")
        print(f"    Recompensa Media (últimos 1000):   {np.mean(rewards[-1000:]):>10.2f}")
        print(f"    Recompensa Máxima:                 {np.max(rewards):>10.2f}")
        print(f"    Recompensa Mínima:                 {np.min(rewards):>10.2f}")
        print(f"    Desviación Estándar:               {np.std(rewards):>10.2f}")
        print(f"    Pasos Medios (últimos 100):        {np.mean(steps[-100:]):>10.1f}")
        print(f"    Tiempo de Entrenamiento:           {data['time']:>10.2f}s")
        
        # Tasa de éxito (recompensa > -50 indica que llegó sin caer mucho)
        success_count = sum(1 for r in rewards if r > -50)
        success_rate = (success_count / len(rewards)) * 100
        print(f"    Tasa de Éxito (reward > -50):      {success_rate:>10.1f}%")
        
        # Episodio de convergencia (primera vez que promedio > -20)
        convergence_ep = -1
        for i in range(WINDOW, len(rewards)):
            if np.mean(rewards[i-WINDOW:i]) > -20:
                convergence_ep = i
                break
        print(f"    Episodio de Convergencia:          {convergence_ep:>10}")


def plot_detailed_metrics(all_metrics, filename="detailed_metrics.png"):
    """Genera gráficos detallados."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'SARSA': 'blue', 'Q-Learning': 'green', 'Monte Carlo': 'red'}
    
    # 1. Recompensa suavizada
    ax1 = axes[0, 0]
    for name, data in all_metrics.items():
        smooth = np.convolve(data['rewards'], np.ones(WINDOW)/WINDOW, mode='valid')
        ax1.plot(smooth, label=name, color=colors.get(name, None), alpha=0.8)
    ax1.set_title(f"Recompensa Media (ventana={WINDOW})", fontsize=14)
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pasos por episodio suavizados
    ax2 = axes[0, 1]
    for name, data in all_metrics.items():
        smooth = np.convolve(data['steps'], np.ones(WINDOW)/WINDOW, mode='valid')
        ax2.plot(smooth, label=name, color=colors.get(name, None), alpha=0.8)
    ax2.set_title(f"Pasos por Episodio (ventana={WINDOW})", fontsize=14)
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Pasos")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Recompensa acumulada
    ax3 = axes[1, 0]
    for name, data in all_metrics.items():
        cumsum = np.cumsum(data['rewards'])
        ax3.plot(cumsum, label=name, color=colors.get(name, None), alpha=0.8)
    ax3.set_title("Recompensa Acumulada", fontsize=14)
    ax3.set_xlabel("Episodio")
    ax3.set_ylabel("Recompensa Acumulada")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Histograma de recompensas finales
    ax4 = axes[1, 1]
    for name, data in all_metrics.items():
        ax4.hist(data['rewards'][-1000:], bins=50, alpha=0.5, label=name, color=colors.get(name, None))
    ax4.set_title("Distribución de Recompensas (últimos 1000 episodios)", fontsize=14)
    ax4.set_xlabel("Recompensa")
    ax4.set_ylabel("Frecuencia")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  Guardado: {filename}")


def main():
    print("\n" + "="*80)
    print("  ENTRENAMIENTO COMPLETO DE CLIFF WALKING - 15000 EPISODIOS")
    print("="*80)
    
    try:
        # Crear entorno con SlipperyCliffWalking wrapper (como el notebook)
        base_env = gym.make('CliffWalking-v1')
        env = SlipperyCliffWalking(base_env, slip_probability=0.1)
        print("\n  Entorno: CliffWalking-v1 + SlipperyCliffWalking (slip_prob=0.1)")
            
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        print(f"  Estados: {n_states}, Acciones: {n_actions}")
        
        # Definir agentes
        agents = {
            'SARSA': SarsaAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1),
            'Q-Learning': QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1),
            'Monte Carlo': MonteCarloAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, 
                                             min_epsilon=0.01, n_episodes=N_EPISODES)
        }
        
        print(f"\n  Parámetros: alpha=0.1, gamma=0.99")
        print(f"  Monte Carlo usa epsilon decay: 1.0 -> 0.01 (como el notebook)")
        
        all_metrics = {}
        
        # Entrenar cada agente
        for name, agent in agents.items():
            metrics = run_experiment_verbose(env, agent, name)
            all_metrics[name] = metrics
            
            # Mostrar política aprendida
            print_policy(agent)
        
        # Imprimir evolución de recompensas
        print_reward_evolution(all_metrics)
        
        # Estadísticas finales
        print_final_statistics(all_metrics)
        
        # Generar gráficos
        print(f"\n{'='*60}")
        print("  GENERANDO GRÁFICOS...")
        print(f"{'='*60}")
        
        plot_metrics(all_metrics)
        print("  Guardado: metrics_comparison.png")
        
        plot_detailed_metrics(all_metrics)
        
        print(f"\n{'='*80}")
        print("  ¡ENTRENAMIENTO COMPLETADO!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
