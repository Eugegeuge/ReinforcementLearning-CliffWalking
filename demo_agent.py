import gymnasium as gym
import numpy as np
import time
import sys
from src.agent import SarsaAgent, QLearningAgent

def demo_agent(agent_type='SARSA', q_table_path=None):
    """
    Runs a visual demo of the agent.
    If q_table_path is provided, loads the Q-table.
    Otherwise, trains a fresh agent quickly.
    """
    print(f"Preparing demo for {agent_type}...")
    
    # Create environment for training (no render)
    env = gym.make('CliffWalking-v1', is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    if agent_type == 'SARSA':
        agent = SarsaAgent(n_states, n_actions)
    elif agent_type == 'Q-Learning':
        agent = QLearningAgent(n_states, n_actions)
    else:
        raise ValueError("Unknown agent type")

    if q_table_path:
        print(f"Loading Q-table from {q_table_path}...")
        try:
            agent.q_table = np.load(q_table_path)
        except FileNotFoundError:
            print("File not found. Training a fresh agent instead...")
            q_table_path = None

    if not q_table_path:
        print("Training agent...")
        # Train quickly
        for i in range(1000):
            state, _ = env.reset()
            action = agent.choose_action(state)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
        print("Training complete.")
    
    env.close()
    
    # Create environment for rendering
    # Note: render_mode='human' might open a window. 
    # If running in a headless env, this might fail or need 'rgb_array' + video saving.
    # Assuming user has a display since they asked for a presentation demo.
    try:
        env_render = gym.make('CliffWalking-v1', is_slippery=True, render_mode='human')
        
        print("\nStarting Visual Demo...")
        print("Watch the popup window!")
        
        state, _ = env_render.reset()
        terminated = False
        truncated = False
        steps = 0
        total_reward = 0
        
        # Disable exploration for demo
        agent.epsilon = 0.0
        
        while not (terminated or truncated) and steps < 100:
            # Render is handled by gym automatically with render_mode='human'
            
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env_render.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.5) # Slow down to see movement
            
        print(f"Demo finished. Total Reward: {total_reward}")
        env_render.close()
        
    except Exception as e:
        print(f"Could not render in 'human' mode: {e}")
        print("Trying to save as video instead (requires moviepy/ffmpeg)...")
        # Fallback to video saving if possible, or just warn
        
if __name__ == "__main__":
    # Default to SARSA
    demo_agent('SARSA', 'sarsa_q_table.npy')
