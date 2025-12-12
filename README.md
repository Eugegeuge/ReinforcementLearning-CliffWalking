# Práctica de Aprendizaje por Refuerzo: Cliff Walking

Este repositorio contiene la implementación de soluciones para el entorno **Cliff Walking** de Gymnasium, utilizando algoritmos de Aprendizaje por Refuerzo (RL).

Proyecto realizado para la asignatura de **Manipuladores (Grado en Ingeniería Robótica)**.

## 📋 Descripción

El objetivo es entrenar agentes capaces de navegar desde un punto de inicio hasta una meta evitando un "acantilado". Se exploran y comparan tres algoritmos:

*   **Q-Learning** (Off-policy, TD-Control)
*   **SARSA** (On-policy, TD-Control)
*   **Monte Carlo** (First-Visit)

El entorno está configurado como **estocástico** (`is_slippery=True`), lo que añade incertidumbre a las transiciones.

## 🚀 Instalación

1.  Clona este repositorio:
    ```bash
    git clone https://github.com/Eugegeuge/ReinforcementLearning-CliffWalking.git
    cd ReinforcementLearning_CliffWalking
    ```
2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Uso

Para entrenar al agente SARSA con los mecanismos de seguridad activados, ejecuta el script principal:

```bash
python main.py
```

Esto realizará lo siguiente:
1.  Entrenará al agente **SARSA** durante **1000 episodios**.
2.  Generará una gráfica de recompensas (`metrics_comparison.png`).
3.  Guardará la tabla Q (`sarsa_q_table.npy`) y métricas (`sarsa_metrics.json`).

## 🛡️ Mecanismos de Seguridad

Para evitar que el entrenamiento se quede colgado (ej. agente dando vueltas en círculos infinitamente), se han implementado:

*   **Timeout Global (300s)**: Si el script tarda más de 5 minutos, se aborta y guarda el progreso.
*   **Max Steps (1000)**: Si un episodio supera los 1000 pasos, se fuerza la terminación de ese episodio.

## 📂 Estructura del Proyecto

*   `src/`: Código fuente de los agentes (`agent.py`) y utilidades (`utils.py`).
*   `main.py`: Script principal de ejecución y orquestación.
*   `Explicacion_Practica.md`: Documentación detallada de los algoritmos y justificación teórica.
*   `Enunciado.md`: Descripción original de la práctica.

## 📊 Resultados Esperados

*   **Q-Learning**: Tiende a aprender el camino óptimo (pegado al acantilado), pero arriesgado durante el entrenamiento.
*   **SARSA**: Tiende a aprender un camino más seguro (alejado del acantilado) debido a la penalización por caídas durante la exploración.

## 👥 Autores

*   [Hugo]
*   Hugo López
*   Juan Diego Serrato
