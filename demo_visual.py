#!/usr/bin/env python3
"""
Demo Visual de Agentes RL en CliffWalking
Ejecuta cada algoritmo con su configuración óptima y muestra visualmente el movimiento.
Ideal para grabar presentaciones.

Uso:
  python demo_visual.py sarsa      # Demo de SARSA
  python demo_visual.py qlearning  # Demo de Q-Learning
  python demo_visual.py montecarlo # Demo de Monte Carlo
  python demo_visual.py todos      # Demo de los 3
"""

import gymnasium as gym
import numpy as np
import time
import sys
import os
import random

# Configuración
TRAIN_EPISODES = 5000  # Episodios para entrenar antes de demo
DEMO_EPISODES = 5      # Episodios a mostrar en demo
STEP_DELAY = 0.3       # Segundos entre pasos (ajustar para grabar)
SLIP_PROB = 0.1


class SlipperyCliffWalking(gym.ActionWrapper):
    def __init__(self, env, slip_probability=0.1):
        super().__init__(env)
        self.slip_probability = slip_probability

    def action(self, action):
        if random.random() < self.slip_probability:
            return self.env.action_space.sample()
        return action


class SarsaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.01):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        
    def choose_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
        
    def update(self, s, a, r, s_, a_):
        td = r + self.gamma * self.q_table[s_, a_] - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.01):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        
    def choose_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
        
    def update(self, s, a, r, s_):
        td = r + self.gamma * np.max(self.q_table[s_]) - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td


class MonteCarloAgent:
    def __init__(self, n_states, n_actions, alpha=0.01, gamma=0.99, epsilon=0.01):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        self.history = []
        
    def choose_action(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
        
    def store(self, s, a, r):
        self.history.append((s, a, r))
        
    def update(self):
        G = 0
        for s, a, r in reversed(self.history):
            G = r + self.gamma * G
            self.q_table[s, a] += self.alpha * (G - self.q_table[s, a])
        self.history = []


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def state_to_pos(state):
    return state // 12, state % 12


def render_grid(state, episode, step, total_reward, agent_name, action=None):
    """Renderiza el grid en la terminal."""
    clear_screen()
    row, col = state_to_pos(state)
    actions = ['↑', '→', '↓', '←']
    action_str = actions[action] if action is not None else ''
    
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  {agent_name:^50}  ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Episodio: {episode:<5} │ Paso: {step:<5} │ Acción: {action_str:<3}     ║")
    print(f"  ║  Recompensa Acumulada: {total_reward:<27} ║")
    print(f"  ╚══════════════════════════════════════════════════════╝\n")
    
    print("      0   1   2   3   4   5   6   7   8   9  10  11")
    print("    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐")
    
    for r in range(4):
        line = f"  {r} │"
        for c in range(12):
            if r == row and c == col:
                cell = " 🤖"  # Agente
            elif r == 3 and c == 11:
                cell = " 🏁"  # Meta
            elif r == 3 and 0 < c < 11:
                cell = " 💀"  # Acantilado
            elif r == 3 and c == 0:
                cell = " 🚩"  # Inicio
            else:
                cell = " · "
            line += f"{cell}│"
        print(line)
        if r < 3:
            print("    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤")
    
    print("    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘")
    print("\n    🚩 = Start  │  🏁 = Goal  │  💀 = Cliff  │  🤖 = Agent")
    print("\n    [Ctrl+C para salir]")


def train_agent(agent, agent_type, episodes=TRAIN_EPISODES):
    """Entrena el agente sin visualización."""
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), SLIP_PROB)
    
    print(f"\n  Entrenando {agent_type}... ", end='', flush=True)
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        
        if agent_type == 'SARSA':
            action = agent.choose_action(state)
            while not done and steps < 500:
                next_state, reward, term, trunc, _ = env.step(action)
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                done = term or trunc
                steps += 1
                
        elif agent_type == 'Q-Learning':
            while not done and steps < 500:
                action = agent.choose_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                done = term or trunc
                steps += 1
                
        elif agent_type == 'Monte Carlo':
            while not done and steps < 500:
                action = agent.choose_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                agent.store(state, action, reward)
                state = next_state
                done = term or trunc
                steps += 1
            agent.update()
        
        if (ep + 1) % (episodes // 5) == 0:
            print(f"{(ep+1)*100//episodes}%.. ", end='', flush=True)
    
    print("✓")
    env.close()
    return agent


def run_demo(agent, agent_type, episodes=DEMO_EPISODES):
    """Ejecuta demo visual del agente entrenado."""
    env = SlipperyCliffWalking(gym.make('CliffWalking-v1'), SLIP_PROB)
    
    print(f"\n  Iniciando demo de {agent_type}...")
    print(f"  Delay entre pasos: {STEP_DELAY}s")
    print(f"  Presiona Enter para comenzar...", end='')
    input()
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        render_grid(state, ep, step, total_reward, agent_type)
        time.sleep(STEP_DELAY)
        
        while not done and step < 100:
            action = agent.choose_action(state, explore=False)  # Sin exploración
            next_state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            step += 1
            state = next_state
            done = term or trunc
            
            render_grid(state, ep, step, total_reward, agent_type, action)
            time.sleep(STEP_DELAY)
        
        # Resultado final
        if state == 47:  # Meta
            print("\n    ✅ ¡META ALCANZADA!")
        else:
            print("\n    ❌ Cayó al acantilado")
        
        print(f"    Recompensa total: {total_reward}")
        
        if ep < episodes:
            print("\n    Siguiente episodio en 2s...")
            time.sleep(2)
    
    env.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    algo = sys.argv[1].lower()
    
    agents = {
        'sarsa': (SarsaAgent, 'SARSA', 5000),
        'qlearning': (QLearningAgent, 'Q-Learning', 5000),
        'montecarlo': (MonteCarloAgent, 'Monte Carlo', 10000)
    }
    
    if algo == 'todos':
        for name in ['sarsa', 'qlearning', 'montecarlo']:
            run_progression_demo(*agents[name])
            print("\n" + "="*60)
            input("  Presiona Enter para el siguiente algoritmo...")
    elif algo in agents:
        run_progression_demo(*agents[algo])
    else:
        print(f"  Error: Algoritmo '{algo}' no reconocido")
        print("  Opciones: sarsa, qlearning, montecarlo, todos")
        sys.exit(1)


def run_progression_demo(AgentClass, agent_type, full_episodes):
    """Muestra progresión: 100 ep, 2500 ep, max episodios."""
    
    print(f"\n{'='*60}")
    print(f"  DEMO DE PROGRESIÓN: {agent_type}")
    print(f"{'='*60}")
    
    agent = AgentClass(48, 4)
    
    # FASE 1: 100 episodios
    print(f"\n  📍 FASE 1: Entrenando 100 episodios...")
    train_agent(agent, agent_type, episodes=100)
    print(f"\n  Agente con 100 episodios de entrenamiento")
    print(f"  (Todavía muy inexperto)")
    input("  Presiona Enter para ver demo...")
    run_demo(agent, f"{agent_type} - 100 episodios", episodes=1)
    
    # FASE 2: 2500 episodios (2400 más)
    print(f"\n  📍 FASE 2: Entrenando hasta 2500 episodios...")
    train_agent(agent, agent_type, episodes=2400)
    print(f"\n  Agente con 2500 episodios de entrenamiento")
    print(f"  (Empezando a aprender)")
    input("  Presiona Enter para ver demo...")
    run_demo(agent, f"{agent_type} - 2500 episodios", episodes=1)
    
    # FASE 3: Max episodios
    remaining = full_episodes - 2500
    print(f"\n  📍 FASE 3: Entrenando hasta {full_episodes} episodios...")
    train_agent(agent, agent_type, episodes=remaining)
    print(f"\n  Agente con {full_episodes} episodios de entrenamiento")
    print(f"  (Completamente entrenado - Política óptima)")
    input("  Presiona Enter para ver demo...")
    run_demo(agent, f"{agent_type} - {full_episodes} episodios (ÓPTIMO)", episodes=1)
    
    print(f"\n  ✅ Demo de {agent_type} completada!")


if __name__ == "__main__":
    main()
