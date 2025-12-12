import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics, window=100):
    """
    Plotea Recompensas y Pasos por episodio suavizados.
    metrics: dict {'Algorithm': {'rewards': [], 'steps': []}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for alg_name, data in metrics.items():
        # Rewards
        rewards = data['rewards']
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smooth_rewards, label=alg_name)
        
        # Steps
        steps = data['steps']
        smooth_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(smooth_steps, label=alg_name)
        
    ax1.set_title(f"Recompensa Media (Ventana: {window})")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f"Pasos por Episodio (Ventana: {window})")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Pasos")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
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
