import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.agent import SarsaAgent
from main import run_experiment
import json

def run_parameter_study():
    # Configuración
    alphas = [0.1, 0.5, 0.9]
    gammas = [0.9, 0.99]
    n_episodes = 500
    
    # Crear entorno
    try:
        env = gym.make('CliffWalking-v1', is_slippery=True)
    except:
        env = gym.make('CliffWalking-v1')
        
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    results = {}
    
    print(f"Iniciando estudio de parámetros ({len(alphas)*len(gammas)} combinaciones)...")
    
    # Colores para graficar
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas) * len(gammas)))
    color_idx = 0
    
    plt.figure(figsize=(12, 6))
    
    for alpha in alphas:
        for gamma in gammas:
            label = f"SARSA (alpha={alpha}, gamma={gamma})"
            print(f"Running {label}...")
            
            agent = SarsaAgent(n_states, n_actions, alpha=alpha, gamma=gamma)
            metrics = run_experiment(env, agent, n_episodes=n_episodes, max_steps=1000)
            
            # Guardar resultados
            results[label] = {
                'rewards': metrics['rewards'],
                'final_avg_reward': np.mean(metrics['rewards'][-50:])
            }
            
            # Plotear suavizado
            window = 50
            smooth_rewards = np.convolve(metrics['rewards'], np.ones(window)/window, mode='valid')
            plt.plot(smooth_rewards, label=label, color=colors[color_idx], linewidth=1.5)
            color_idx += 1
            
    plt.title(f"Comparativa de Parámetros SARSA (Recompensa Suavizada, w={window})")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Promedio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parameter_study.png')
    print("Gráfica guardada en 'parameter_study.png'")
    
    # Guardar JSON
    clean_results = {k: {'final_avg_reward': v['final_avg_reward']} for k, v in results.items()}
    with open('experiment_results.json', 'w') as f:
        json.dump(clean_results, f, indent=4)
        
    env.close()

if __name__ == "__main__":
    run_parameter_study()
