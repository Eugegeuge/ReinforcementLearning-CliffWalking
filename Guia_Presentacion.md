# Guía para la Presentación (10-12 minutos)

Esta guía estructura tu presentación para cubrir todos los puntos del `Enunciado.md`.

## Estructura General

| Sección | Tiempo | Contenido Clave | Material de Apoyo |
| :--- | :--- | :--- | :--- |
| **1. Introducción** | 2 min | El problema del Acantilado y el dilema Riesgo vs. Recompensa. | Diapositiva con imagen del grid. |
| **2. Algoritmos** | 3 min | Explicación breve de Q-Learning, SARSA y Monte Carlo. | `Explicacion_Practica.md` |
| **3. Resultados** | 3 min | Comparativa de curvas y métricas. | `metrics_comparison.png`, Output de `analyze_metrics.py` |
| **4. Análisis** | 2 min | Efecto de parámetros (Alpha/Epsilon). | `parameter_study.png` |
| **5. Demo** | 1 min | Ver al agente en acción. | Script `demo_agent.py` |
| **6. Conclusiones** | 1 min | ¿Cuál es mejor? Justificación final. | Resumen verbal. |

---

## Guion Detallado

### 1. Introducción (2 min)
*   **El Entorno**: Cliff Walking.
    *   **Objetivo**: Ir de A a B.
    *   **Restricción**: El suelo resbala (`is_slippery=True`), lo que añade incertidumbre.
    *   **El Dilema**: El camino corto bordea el precipicio (-13 pasos). El camino seguro da un rodeo (-15/17 pasos).
*   **Objetivo de la práctica**: Comparar cómo diferentes "personalidades" de IA (agentes) afrontan este riesgo.

### 2. Algoritmos Implementados (3 min)
*   **Q-Learning (El Optimista)**:
    *   Aprende el valor de la acción *óptima*.
    *   Ignora que a veces explora (se mueve al azar).
    *   *Resultado*: Intenta ir por el borde, pero se cae mucho mientras entrena.
*   **SARSA (El Prudente)**:
    *   Aprende el valor de la acción *que realmente hace*.
    *   Tiene en cuenta que puede equivocarse.
    *   *Resultado*: Prefiere el camino seguro (arriba), lejos del borde.
*   **Monte Carlo (El Historiador)**:
    *   Aprende al final de cada episodio completo.
    *   *Resultado*: Converge, pero es más lento y tiene más varianza.

### 3. Resultados Experimentales (3 min)
*   **Mostrar Gráfica**: `metrics_comparison.png`
    *   Señala cómo **SARSA** (línea naranja/azul) tiene menos caídas (picos negativos) al principio.
    *   Señala cómo **Q-Learning** llega a una recompensa teórica mejor, pero es más inestable.
*   **Datos Duros** (Usa los datos de `analyze_metrics.py`):
    *   *"Como vemos en los datos, SARSA tiene una tasa de éxito del X% durante el entrenamiento, mientras que Q-Learning sufre más caídas..."*

### 4. Análisis de Parámetros (2 min)
*   **Mostrar Gráfica**: `parameter_study.png`
*   **Alpha (Tasa de aprendizaje)**:
    *   Bajo (0.1): Lento pero seguro.
    *   Alto (0.9): Rápido pero inestable (oscila).
*   **Epsilon (Exploración)**:
    *   Alto (0.5): Desastroso en este entorno. Demasiada aleatoriedad cerca de un acantilado es fatal.

### 5. Demostración en Vivo (1 min)
*   *"Ahora veremos al agente SARSA navegando el entorno..."*
*   **Ejecutar**: `python demo_agent.py`
*   Comentar mientras se mueve: *"Veis cómo evita la fila de abajo para no correr riesgos..."*

### 6. Conclusiones (1 min)
*   **Mejor Algoritmo**: Para este entorno peligroso y estocástico, **SARSA** es superior en seguridad durante el entrenamiento.
*   **Q-Learning**: Es mejor si solo nos importa el camino óptimo final y no el coste de aprenderlo (podemos permitirnos caer mil veces en simulación, pero no en un robot real).
*   **Justificación**: Hemos elegido SARSA como el agente principal por su robustez.

---

## Preguntas Preparadas (Por si acaso)

*   **¿Por qué Monte Carlo tarda más?**: Porque solo actualiza al final del episodio. Si el episodio es largo, tarda mucho en recibir feedback.
*   **¿Qué pasa si quitas `is_slippery`?**: El entorno se vuelve determinista. Q-Learning aprendería el camino óptimo sin caerse tanto, porque "siempre que decido ir a la derecha, voy a la derecha".
