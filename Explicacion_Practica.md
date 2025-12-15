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

## 4. Desafíos de Implementación: Bucles Infinitos

En RL, especialmente al inicio del entrenamiento, es común que el agente entre en **bucles infinitos**.
- **Causa**: El agente descubre que moverse entre dos casillas seguras (ej. Arriba <-> Abajo) tiene coste -1 + -1 = -2, pero caer al acantilado es -100. Si su política actual cree que el acantilado está en todas partes menos ahí, se quedará moviéndose para siempre.
- **Solución (Safety Locks)**: Hemos implementado un límite de **1000 pasos por episodio**. Si el agente no llega a la meta en 1000 pasos, cortamos el episodio. Esto le obliga a reiniciar y probar otra cosa, evitando que el programa se cuelgue.

---

## Resumen para la Entrega
Si te preguntan en la práctica:
> "¿Por qué Q-Learning y SARSA aprenden caminos distintos?"

**Respuesta**: Porque **Q-Learning** aprende el valor de la política *óptima* (sin exploración), mientras que **SARSA** aprende el valor de la política *que está ejecutando* (incluyendo la exploración aleatoria). Por eso SARSA es más miedoso al principio: "sabe" que puede tropezar y prefiere no arriesgarse cerca del borde.

### Monte Carlo (El "Historiador")
- **Filosofía**: *"No aprendo hasta que termino el episodio completo."*
- **Cómo aprende**: Guarda todo lo que ha pasado en un episodio (estado, acción, recompensa) y, cuando termina, repasa la historia hacia atrás para calcular el retorno real (G) obtenido desde cada punto.
- **Resultado**: Converge a la solución óptima si se le da suficiente tiempo, pero tiene **alta varianza** (porque cada episodio es diferente) y es más lento de actualizar (episodio a episodio, no paso a paso).

## 5. Análisis de Parámetros (Resultados del Estudio)

Hemos realizado un estudio de sensibilidad (`parameter_study.py`) variando Alpha y Epsilon:

### Efecto de Alpha (Tasa de Aprendizaje)
- **Alpha bajo (0.1)**: Aprendizaje lento pero estable. La curva de recompensa sube suavemente.
- **Alpha alto (0.9)**: Aprendizaje muy rápido al principio, pero oscila mucho. El agente cambia de opinión drásticamente con cada nueva experiencia.

### Efecto de Epsilon (Exploración)
- **Epsilon bajo (0.1)**: Converge rápido a una buena política, pero corre el riesgo de quedarse en un óptimo local si no explora suficiente (aunque en CliffWalking esto es menos probable).
- **Epsilon alto (0.5)**: Nunca termina de estabilizarse en una recompensa alta porque sigue haciendo movimientos aleatorios el 50% del tiempo, cayendo al acantilado constantemente. **SARSA sufre mucho con epsilon alto** porque asume que esos fallos son parte de la política.
