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

class QLearningAgent(RLAgent):
    """Agente Q-Learning (Off-policy)."""
    def update(self, state, action, reward, next_state, next_action=None):
        # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
        best_next_action_val = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * best_next_action_val
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

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

class MonteCarloAgent(RLAgent):
    """Agente Monte Carlo (First-Visit, Constant Alpha)."""
    def __init__(self, n_states, n_actions, alpha=0.01, gamma=0.99, epsilon=0.1):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon)
        self.episode = []

    def update(self, state, action, reward, next_state, next_action=None):
        # Solo almacenar la transición. La actualización ocurre al final.
        self.episode.append((state, action, reward))

    def on_episode_end(self):
        G = 0
        visited_in_episode = set()
        
        # Recorrer episodio hacia atrás
        for i in range(len(self.episode) - 1, -1, -1):
            state, action, reward = self.episode[i]
            G = self.gamma * G + reward
            
            # First-Visit Check
            # Comprobar si este par (s,a) apareció antes en el episodio
            previous_occurences = [
                (self.episode[j][0], self.episode[j][1]) 
                for j in range(i)
            ]
            
            if (state, action) not in previous_occurences:
                # Actualizar Q-Table: Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a))
                self.q_table[state, action] += self.alpha * (G - self.q_table[state, action])
        
        # Limpiar historial
        self.episode = []
