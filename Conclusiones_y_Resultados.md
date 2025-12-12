# 📊 Conclusiones y Resultados: Cliff Walking

Este documento resume los hallazgos principales tras la experimentación en el entorno **Cliff Walking** con los algoritmos **Q-Learning** y **SARSA**.

## 1. Comparativa de Comportamiento

| Característica | Q-Learning (Off-Policy) | SARSA (On-Policy) |
| :--- | :--- | :--- |
| **Objetivo** | Aprende la política *óptima absoluta*. | Aprende la política *más segura* dado el comportamiento actual. |
| **Riesgo** | **Alto**. Camina pegado al acantilado. | **Bajo**. Se aleja del borde ("Safety Buffer"). |
| **Convergercia** | Converge al camino más corto (-13 pasos). | Converge a un camino más largo pero seguro. |
| **Rendimiento (Entrenamiento)** | Peor. Sufre muchas caídas (-100) por exploración. | Mejor. Evita caídas drásticas al ser "consciente" de su torpeza exploratoria. |

## 2. Análisis de Resultados

### ¿Por qué toman caminos diferentes?
La diferencia clave radica en la ecuación de actualización:

*   **Q-Learning**: `Q(s,a) <-- ... + max Q(s', a')`. Al usar el `max`, el agente asume que en el siguiente paso **no fallará** y tomará la mejor decisión posible. Por eso ve el camino junto al acantilado como el mejor (-13 pasos), ignorando que su exploración epsilon-greedy podría hacerle caer.
*   **SARSA**: `Q(s,a) <-- ... + Q(s', a')`. Usa la acción que **realmente** va a tomar (que puede ser aleatoria). Si al caminar junto al borde, la exploración le hace saltar al vacío, SARSA asocia "caminar junto al borde" con "dolor", y aprende a evitarlo.

### Gráficas Esperadas
En las gráficas de entrenamiento (`metrics_comparison.png`), deberíamos observar:
1.  **SARSA**: Curva de recompensa más estable y alta durante el entrenamiento (converge a aprox -30/-50).
2.  **Q-Learning**: Curva con muchos picos hacia abajo (caídas) y un promedio peor durante el entrenamiento, aunque su *política final* (si quitamos la exploración) sea teóricamente mejor.

## 3. Desafíos Superados: Bucles Infinitos

Durante el desarrollo, nos encontramos con que los agentes a veces se quedaban atrapados caminando en círculos en zonas seguras.
*   **Motivo**: Para un agente inexperto, moverse de un lado a otro (-1 por paso) es mejor que arriesgarse a caer al acantilado (-100). Si no encuentra la meta rápido, prefiere quedarse dando vueltas.
*   **Solución**: Implementación de **Safety Locks**:
    *   **Max Steps (1000)**: Fuerza el fin del episodio si se tarda demasiado, obligando al agente a reiniciar y explorar nuevas rutas.

## 4. Conclusión General

En entornos críticos donde un fallo es catastrófico (como un robot real o un acantilado), **SARSA** es preferible para el entrenamiento online porque evita situaciones peligrosas mientras aprende. **Q-Learning**, aunque encuentra la solución óptima teórica, es demasiado arriesgado para aprender "sobre la marcha" en sistemas físicos reales sin simulador.
