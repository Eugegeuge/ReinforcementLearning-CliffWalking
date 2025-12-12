import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards_dict, window=100):
    """
    Plotea las recompensas suavizadas.
    rewards_dict: diccionario {nombre_agente: lista_recompensas}
    """
    plt.figure(figsize=(10, 6))
    
    for name, rewards in rewards_dict.items():
        # Suavizar curva (Moving Average)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=name)
        
    plt.title(f"Recompensas por Episodio (Suavizado: {window})")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rewards.png")
    plt.close()

def print_policy(agent, shape=(4, 12)):
    """
    Imprime la política aprendida en consola.
    0: Up, 1: Right, 2: Down, 3: Left
    """
    actions = ['^', '>', 'v', '<']
    grid = np.full(shape, ' ')
    
    print(f"\nPolítica aprendida ({agent.__class__.__name__}):")
    for s in range(agent.n_states):
        row = s // shape[1]
        col = s % shape[1]
        
        # Ignorar acantilado (fila 3, cols 1-10)
        if row == 3 and 0 < col < 11:
            grid[row, col] = 'C'
            continue
            
        if s == 47: # Goal
            grid[row, col] = 'G'
            continue
            
        action_idx = np.argmax(agent.q_table[s, :])
        grid[row, col] = actions[action_idx]
        
    # Imprimir
    for row in range(shape[0]):
        print(f"{row}: " + " ".join(grid[row]))
