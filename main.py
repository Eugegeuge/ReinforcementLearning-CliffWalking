import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.agent import QLearningAgent, SarsaAgent, MonteCarloAgent
from src.utils import plot_rewards, print_policy

def run_experiment(env, agent, n_episodes=500):
    rewards = []
    
    for episode in tqdm(range(n_episodes), desc=f"Entrenando {agent.__class__.__name__}"):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 5000:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            
            # Actualizar agente
            agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            
        # Hook para agentes que actualizan al final (MC)
        agent.on_episode_end()
        
        rewards.append(total_reward)
        
    return rewards

def main():
    # Crear entorno
    env = gym.make('CliffWalking-v1', is_slippery=True)
    
    # Parámetros
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    n_episodes = 500
    
    # Inicializar agentes
    q_agent = QLearningAgent(n_states, n_actions)
    sarsa_agent = SarsaAgent(n_states, n_actions)
    mc_agent = MonteCarloAgent(n_states, n_actions)
    
    # Entrenar Q-Learning
    print("--- Iniciando Q-Learning (Off-policy Control) ---")
    q_rewards = run_experiment(env, q_agent, n_episodes)
    
    # Entrenar SARSA
    print("\n--- Iniciando SARSA (On-policy Control) ---")
    sarsa_rewards = run_experiment(env, sarsa_agent, n_episodes)

    # Entrenar Monte Carlo
    # print("\n--- Iniciando Monte Carlo (First-Visit) ---")
    # mc_rewards = run_experiment(env, mc_agent, n_episodes)
    
    # Graficar resultados
    plot_rewards({
        'Q-Learning': q_rewards,
        'SARSA': sarsa_rewards,
        # 'Monte Carlo': mc_rewards
    })
    print(f"\nGráfica guardada en 'rewards.png'")
    
    # Mostrar políticas finales
    print_policy(q_agent)
    print_policy(sarsa_agent)
    # print_policy(mc_agent)
    
    env.close()

if __name__ == "__main__":
    main()
