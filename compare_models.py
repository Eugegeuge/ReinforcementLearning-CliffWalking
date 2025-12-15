import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.agent import SarsaAgent, QLearningAgent, MonteCarloAgent
from src.utils import plot_metrics, print_policy
import time
import json

def run_experiment(env, agent, n_episodes=500, max_steps=1000):
    rewards = []
    steps_list = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < max_steps:
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Choose next action (needed for SARSA, optional for others but good for uniformity)
            next_action = agent.choose_action(next_state)
            
            # Update agent
            # Note: Q-Learning and MC handle 'next_action' differently or ignore it inside update
            agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
        rewards.append(total_reward)
        steps_list.append(steps)
        
        # Callback for episode end (Crucial for Monte Carlo)
        agent.on_episode_end()
        
    return {'rewards': rewards, 'steps': steps_list}

def main():
    try:
        # Create environment
        try:
            env = gym.make('CliffWalking-v1', is_slippery=True)
        except:
            print("Warning: 'is_slippery' not accepted, using default.")
            env = gym.make('CliffWalking-v1')
            
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        n_episodes = 1000
        
        # Define agents to compare
        agents = {
            'SARSA': SarsaAgent(n_states, n_actions),
            'Q-Learning': QLearningAgent(n_states, n_actions),
            'Monte Carlo': MonteCarloAgent(n_states, n_actions)
        }
        
        all_metrics = {}
        
        for name, agent in agents.items():
            print(f"\n--- Running {name} ---")
            start_time = time.time()
            metrics = run_experiment(env, agent, n_episodes)
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f}s")
            
            all_metrics[name] = metrics
            
            # Show policy
            print_policy(agent)
            
        # Plot comparison
        print("\nGenerating comparison plot...")
        plot_metrics(all_metrics)
        print("Saved 'metrics_comparison.png'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()
