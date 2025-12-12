# Guía de Estudio: Práctica Cliff Walking

Este documento explica **qué** hemos hecho, **por qué** y **cómo** funciona. Úsalo para entender la práctica y redactar tu memoria.

## 1. El Objetivo (El Entorno)
El problema es el **Cliff Walking** (Caminar junto al acantilado).
- **Meta**: Ir del punto `START` (abajo-izquierda) al `GOAL` (abajo-derecha).
- **Peligro**: Todo el borde inferior (entre Start y Goal) es un **acantilado**.
- **Castigos (Recompensas negativas)**:
  - **-1** por cada paso dado (incentiva llegar rápido).
  - **-100** si caes al acantilado (te devuelve al inicio).

El dilema es: **¿Tomo el camino corto pegado al borde (peligroso) o doy un rodeo seguro (lento)?**

## 2. Los Algoritmos (Los Agentes)

Hemos programado dos "cerebros" distintos. Ambos usan una tabla (Q-Table) para recordar qué tan buena es cada acción en cada casilla, pero actualizan esa tabla de forma diferente.

### Q-Learning (El "Optimista")
- **Filosofía**: *"Asumo que en el futuro no cometeré errores."*
- **Cómo aprende**: Cuando actualiza su tabla, mira la **mejor** acción posible del siguiente estado (`max Q`), ignorando que a veces explora y se mueve al azar.
- **Resultado**: Aprende el **Camino Óptimo Teórico**. Va pegado al borde del acantilado porque es el camino más corto (-13 pasos).
- **Riesgo**: Mientras está aprendiendo (y explorando), se cae muchísimo porque camina por la cuerda floja. Pero su "mapa mental" (Q-Table) converge a la ruta perfecta.

### SARSA (El "Realista" o "Prudente")
- **Filosofía**: *"Aprendo de lo que realmente hago, incluyendo mis errores."*
- **Cómo aprende**: Mira la acción que **realmente** va a tomar a continuación (que a veces es aleatoria por exploración).
- **Resultado**: Aprende el **Camino Seguro**. Se da cuenta de que caminar pegado al borde es mala idea porque, si de repente decide explorar (moverse al azar) hacia abajo, se cae (-100).
- **Consecuencia**: Prefiere irse por arriba, lejos del borde. Tarda más pasos (menos recompensa por episodio), pero es más robusto a fallos durante el entrenamiento.

## 3. Estructura del Código

### `src/agent.py` (Los Cerebros)
Aquí están las clases `QLearningAgent` y `SarsaAgent`.
- **`choose_action`**: Decide qué hacer. Usa "Epsilon-Greedy": la mayoría de veces hace lo mejor que sabe, pero un % de veces (epsilon) hace algo loco para probar cosas nuevas.
- **`update`**: La fórmula matemática.
  - Q-Learning usa `max()`.
  - SARSA usa la acción real siguiente.

### `main.py` (El Entrenador)
Es el gimnasio.
1. Crea el mundo (`CliffWalking-v1`).
2. Pone a entrenar a Q-Learning 500 veces.
3. Pone a entrenar a SARSA 500 veces.
4. Guarda las notas (recompensas) y dibuja el mapa final.

### `src/utils.py` (Los Ojos)
- **`plot_rewards`**: Pinta la gráfica `rewards.png`. Si la línea sube, el agente está aprendiendo.
- **`print_policy`**: Imprime las flechitas en la consola para que veas qué camino ha decidido tomar cada uno.

---

## Resumen para la Entrega
Si te preguntan en la práctica:
> "¿Por qué Q-Learning y SARSA aprenden caminos distintos?"

**Respuesta**: Porque **Q-Learning** aprende el valor de la política *óptima* (sin exploración), mientras que **SARSA** aprende el valor de la política *que está ejecutando* (incluyendo la exploración aleatoria). Por eso SARSA es más miedoso al principio: "sabe" que puede tropezar y prefiere no arriesgarse cerca del borde.
