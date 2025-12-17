import numpy as np

class RLAgent:
    """Clase base para agentes de RL tabular."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Romper empates aleatoriamente
            values = self.q_table[state, :]
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, state, action, reward, next_state, next_action=None):
        raise NotImplementedError("Debe ser implementado por la subclase")

    def on_episode_end(self):
        """Método opcional para agentes que actualizan al final del episodio (ej. Monte Carlo)."""
        pass

class SarsaAgent(RLAgent):
    """Agente SARSA (On-policy)."""
    def update(self, state, action, reward, next_state, next_action=None):
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        if next_action is None:
             raise ValueError("SARSA requiere next_action para actualizar")
        
        next_q_val = self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q_val
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

class QLearningAgent(RLAgent):
    """Agente Q-Learning (Off-policy)."""
    def update(self, state, action, reward, next_state, next_action=None):
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        # No necesitamos next_action, usamos el max over actions para next_state
        
        max_next_q_val = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * max_next_q_val
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

class MonteCarloAgent(RLAgent):
    """Agente Monte Carlo con Epsilon Decay (Every-Visit)."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, 
                 min_epsilon=0.01, n_episodes=15000):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon)
        self.episode_history = []
        self.min_epsilon = min_epsilon
        self.max_epsilon = epsilon
        self.decay_rate = 5 / n_episodes  # Como en el notebook
        self.episode_count = 0
        
    def update(self, state, action, reward, next_state, next_action=None):
        # En Monte Carlo, guardamos la transición y actualizamos al final
        self.episode_history.append((state, action, reward))
        
    def on_episode_end(self):
        # Calcular retornos G y actualizar Q-table (Every-Visit, como en el notebook)
        if len(self.episode_history) > 0:
            G = 0
            
            # Recorrer el episodio hacia atrás (Every-Visit: actualiza todos los pares)
            for t in range(len(self.episode_history) - 1, -1, -1):
                state, action, reward = self.episode_history[t]
                G = reward + self.gamma * G
                
                # Actualización incremental
                # Q(S, A) <- Q(S, A) + alpha * [G - Q(S, A)]
                self.q_table[state, action] += self.alpha * (G - self.q_table[state, action])
        
        # Epsilon decay exponencial (clave para convergencia!)
        self.episode_count += 1
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.episode_count)
                
        # Limpiar historia
        self.episode_history = []
