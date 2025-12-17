import gymnasium as gym
import numpy as np
import time
from src.agent import SarsaAgent, QLearningAgent, MonteCarloAgent

def analyze_agent(env, agent, n_episodes=1000):
    rewards = []
    success_count = 0
    convergence_episode = -1
    
    # Moving average window for convergence check
    window = 50
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 1000:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
        rewards.append(total_reward)
        agent.on_episode_end()
        
        # Check success (reached goal = reward > -100 usually, but here cliff is -100)
        # In CliffWalking, reaching goal usually means not falling (-100).
        # Optimal path is -13. Any reward > -100 implies we didn't fall? 
        # Actually, falling gives -100 and resets. 
        # If we reach goal, we get -1 per step.
        # Let's assume success if total_reward > -100 (approx, depends on path length)
        # Better: check if terminated and not truncated? 
        # But falling also terminates? No, falling resets to start in some versions or terminates.
        # In standard CliffWalking-v1: "The episode terminates when the player enters the cliff or the goal."
        # Wait, if it terminates on cliff, reward is -100. If goal, reward is -1.
        # So success is reaching goal.
        
        # Let's rely on reward threshold. Optimal is -13. Safe is around -15/-17.
        if total_reward > -50: 
            success_count += 1
            
        # Check convergence
        if convergence_episode == -1 and episode > window:
            avg_recent = np.mean(rewards[-window:])
            # If variance is low and reward is high enough?
            # Or just when it hits a stable high value.
            if avg_recent > -20: # Arbitrary threshold for "good behavior"
                convergence_episode = episode
                
    return {
        'avg_reward_last_100': np.mean(rewards[-100:]),
        'success_rate': (success_count / n_episodes) * 100,
        'convergence_episode': convergence_episode
    }

def main():
    env = gym.make('CliffWalking-v1', is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    agents = {
        'SARSA': SarsaAgent(n_states, n_actions),
        'Q-Learning': QLearningAgent(n_states, n_actions),
        'Monte Carlo': MonteCarloAgent(n_states, n_actions)
    }
    
    print(f"{'Agent':<15} | {'Success Rate':<15} | {'Avg Reward (Last 100)':<25} | {'Conv. Ep':<10}")
    print("-" * 75)
    
    for name, agent in agents.items():
        metrics = analyze_agent(env, agent)
        print(f"{name:<15} | {metrics['success_rate']:>14.1f}% | {metrics['avg_reward_last_100']:>24.1f} | {metrics['convergence_episode']:>10}")

if __name__ == "__main__":
    main()
