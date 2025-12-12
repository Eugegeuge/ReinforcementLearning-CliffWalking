import gymnasium as gym
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg') # Back-end no interactivo para evitar bloqueos
import matplotlib.pyplot as plt
from src.agent import SarsaAgent
from src.utils import plot_metrics, print_policy
import time

def run_experiment(env, agent, n_episodes=500, max_steps=1000):
    rewards = []
    steps_list = []
    
    # Límite estricto para evitar bucles (ahora parametrizable)
    # max_steps ahora viene como argumento
    
    # Límite de tiempo global de seguridad (5 minutos)
    MAX_GLOBAL_TIME = 300 
    start_time = time.time()
    
    print(f"Iniciando entrenamiento con seguridad: Timeout global de {MAX_GLOBAL_TIME}s")
    
    for episode in range(n_episodes):
        # Chequeo de seguridad de tiempo global
        current_total_time = time.time() - start_time
        if current_total_time > MAX_GLOBAL_TIME:
            print(f"\n[ALERTA DE SEGURIDAD] Tiempo máximo excedido ({current_total_time:.2f}s > {MAX_GLOBAL_TIME}s).")
            print("Abortando entrenamiento y guardando progreso actual...")
            break
            
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        # Log de progreso más frecuente (cada 5%)
        log_freq = max(1, n_episodes // 20)
        if episode % log_freq == 0:
            elapsed = time.time() - start_time
            print(f"Episodio {episode}/{n_episodes} - Tiempo: {elapsed:.2f}s - Epsilon: {agent.epsilon:.2f}")
        
        while not (terminated or truncated) and steps < max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            
            # Actualizar agente
            agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
        rewards.append(total_reward)
        steps_list.append(steps)
        
        # Callback para finalizar episodio (necesario para Monte Carlo)
        agent.on_episode_end()
        
    return {'rewards': rewards, 'steps': steps_list}

def main():
    try:
        print("Iniciando ejecución exclusiva de SARSA...")
        
        # Crear entorno
        try:
            env = gym.make('CliffWalking-v1', is_slippery=True)
        except:
            print("Advertencia: 'is_slippery' no aceptado, usando config por defecto.")
            env = gym.make('CliffWalking-v1')
        
        # Parámetros
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        n_episodes = 1000 # Aumentado para más datos
        
        # Inicializar solo SARSA
        agent_name = 'SARSA'
        agent = SarsaAgent(n_states, n_actions)
        
        print(f"\n--- Iniciando {agent_name} ({n_episodes} episodios) ---")
        metrics = run_experiment(env, agent, n_episodes)
        
        # Guardar datos brutos
        print("\nGuardando datos...")
        
        # 1. Guardar tabla Q
        np.save('sarsa_q_table.npy', agent.q_table)
        print("- Tabla Q guardada en 'sarsa_q_table.npy'")
        
        # 2. Guardar métricas en JSON
        import json
        metrics_data = {
            'rewards': metrics['rewards'],
            'steps': metrics['steps'],
            'params': {
                'n_episodes': n_episodes,
                'alpha': agent.alpha,
                'gamma': agent.gamma,
                'epsilon_initial': agent.epsilon # Nota: si epsilon decae, esto es el inicial o el final segun implementacion
            }
        }
        with open('sarsa_metrics.json', 'w') as f:
            json.dump(metrics_data, f)
        print("- Métricas guardadas en 'sarsa_metrics.json'")
            
        # Mostrar política aprendida al finalizar
        print_policy(agent)

        # Graficar resultados (solo SARSA)
        # Adaptamos el diccionario para que plot_metrics funcione igual
        all_metrics = {agent_name: metrics}
        plot_metrics(all_metrics)
        print(f"\nGráfica guardada en 'metrics_comparison.png'")
        
    except Exception as e:
        print(f"\n[ERROR CRITICO] Ocurrió un error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
            print("Entorno cerrado correctamente.")
            sys.stdout.flush()
            sys.exit(0)

if __name__ == "__main__":
    main()
