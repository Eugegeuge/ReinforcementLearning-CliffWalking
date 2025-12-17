# Q-Learning Óptimo - Justificación de Parámetros

## Configuración Óptima

| Parámetro | Valor | Rango Probado |
|-----------|-------|---------------|
| **Alpha (α)** | 0.1 | 0.01 - 0.9 |
| **Gamma (γ)** | 0.99 | 0.5 - 1.0 |
| **Epsilon (ε)** | 0.01 | 0.01 - 0.5 |
| **Episodios** | 10,000 | - |

---

## Alpha (α) = 0.1 - Learning Rate

### Fórmula Q-Learning
```
Q(s,a) ← Q(s,a) + α * [r + γ*max_a'Q(s',a') - Q(s,a)]
```

### Resultados del Estudio

| Alpha | Avg últ.100 | Éxito |
|-------|-------------|-------|
| 0.01 | -26.60 | 84.2% |
| 0.05 | -25.18 | 90.1% |
| **0.1** | -29.87 | **91.4%** ✓ |
| 0.3 | -36.63 | 90.2% |
| 0.9 | -29.32 | 80.1% |

**¿Por qué 0.1?** Es el balance óptimo entre velocidad de aprendizaje y estabilidad. Q-Learning tolera alphas más altos que Monte Carlo porque usa el máximo Q(s',a'), que es más estable.

---

## Gamma (γ) = 0.99 - Factor de Descuento

### Resultados del Estudio

| Gamma | Avg últ.100 | Éxito |
|-------|-------------|-------|
| 0.5 | -79.74 | 53.0% |
| 0.7 | -32.93 | 88.8% |
| 0.9 | -24.51 | 90.8% |
| **0.99** | -28.33 | **91.2%** ✓ |
| 1.0 | -28.88 | 91.5% |

**¿Por qué 0.99?** Permite al agente "ver" las consecuencias futuras de caer al acantilado. γ=1.0 también funciona pero puede causar inestabilidad en algunos casos.

---

## Epsilon (ε) = 0.01 - Exploración

### Resultados del Estudio

| Epsilon | Avg últ.100 | Éxito |
|---------|-------------|-------|
| **0.01** | **-24.18** | **95.3%** ✓ |
| 0.05 | -28.37 | 94.2% |
| 0.1 | -29.42 | 91.6% |
| 0.3 | -78.13 | 73.9% |
| 0.5 | -233.07 | 40.8% |

**¿Por qué 0.01?** Mínima exploración maximiza el rendimiento. Q-Learning es off-policy, por lo que puede aprender la política óptima incluso con poca exploración.

---

## Q-Learning vs SARSA vs Monte Carlo

| Aspecto | Q-Learning | SARSA | Monte Carlo |
|---------|------------|-------|-------------|
| **Tipo** | Off-policy | On-policy | On-policy |
| **Actualización** | max Q(s',a') | Q(s',a') | G (retorno) |
| **Alpha óptimo** | 0.1 | 0.1 | 0.01 |
| **Robustez** | Alta | Alta | Baja |
| **Tiempo** | ~2s | ~2s | ~15s |
| **Éxito típico** | ~95% | ~95% | ~65% |

### Diferencia clave: Off-policy

Q-Learning aprende la **política óptima** independientemente de la política de exploración. Esto lo hace más eficiente pero potencialmente más agresivo (puede subestimar el riesgo del acantilado).

---

## Configuración Final

```python
QLearningAgent(
    alpha=0.1,      # Balance óptimo
    gamma=0.99,     # Alta importancia al futuro
    epsilon=0.01,   # Mínima exploración
)
# 10,000 episodios
```

### Resultados Esperados
- **Recompensa (últ. 100)**: ~-22 a -28
- **Tasa de Éxito**: ~95%
- **Tiempo**: ~2 segundos

---

*Documento generado automáticamente basado en estudios de hiperparámetros*
