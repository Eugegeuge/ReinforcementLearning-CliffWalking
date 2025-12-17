# SARSA Óptimo - Justificación de Parámetros

## Resumen de Configuración Óptima

| Parámetro | Valor | Rango Probado |
|-----------|-------|---------------|
| **Alpha (α)** | 0.1 | 0.01 - 0.9 |
| **Gamma (γ)** | 0.99 | 0.5 - 1.0 |
| **Epsilon (ε)** | 0.01 | 0.01 - 0.5 |
| **Episodios** | 10,000 | - |

---

## Alpha (α) = 0.1 - Learning Rate

### ¿Qué hace?
Controla cuánto "confiamos" en la nueva información vs el conocimiento previo.

### Fórmula SARSA
```
Q(s,a) ← Q(s,a) + α * [r + γQ(s',a') - Q(s,a)]
```

### ¿Por qué 0.1?

| Alpha | Avg últ.100 | Éxito | Observación |
|-------|-------------|-------|-------------|
| 0.01 | -26.96 | 84.4% | Muy lento para converger |
| 0.05 | -25.50 | 90.8% | Bueno pero lento |
| **0.1** | -32.14 | **91.3%** | ✓ Balance óptimo |
| 0.3 | -39.60 | 88.1% | Algo inestable |
| 0.9 | -369.14 | 23.4% | Catastrófico |

**Diferencia con Monte Carlo**: SARSA actualiza **paso a paso** (no al final del episodio), por lo que tolera alphas más altos. α=0.1 es el estándar en literatura y funciona muy bien.

---

## Gamma (γ) = 0.99 - Factor de Descuento

### ¿Qué hace?
Determina cuánto le importa al agente el futuro vs las recompensas inmediatas.

### ¿Por qué 0.99?

| Gamma | Avg últ.100 | Éxito | Comportamiento |
|-------|-------------|-------|----------------|
| 0.5 | -96.88 | 23.5% | "Miope" - no planifica |
| 0.7 | -61.36 | 47.0% | Cortoplacista |
| 0.9 | -35.61 | 90.2% | Bueno |
| 0.95 | -25.76 | 90.8% | Muy bueno |
| **0.99** | -26.54 | **91.7%** | ✓ Óptimo |
| 1.0 | -27.27 | 91.6% | Igual de bueno |

**Conclusión**: γ=0.99 permite que SARSA "vea" las consecuencias de caer al acantilado y aprenda a evitarlo.

---

## Epsilon (ε) = 0.01 - Exploración

### ¿Qué hace?
Probabilidad de tomar una acción aleatoria en lugar de seguir la política aprendida.

### ¿Por qué 0.01?

| Epsilon | Avg últ.100 | Éxito | Problema |
|---------|-------------|-------|----------|
| **0.01** | **-21.59** | **95.5%** | ✓ Óptimo |
| 0.05 | -23.13 | 94.2% | Muy bueno |
| 0.1 | -36.15 | 91.1% | Bueno |
| 0.3 | -46.24 | 77.0% | Demasiada exploración |
| 0.5 | -98.71 | 45.6% | Casi aleatorio |

**Nota**: SARSA funciona bien incluso con ε=0.1, pero ε=0.01 maximiza el rendimiento final.

---

## Episodios = 10,000

### ¿Por qué este número?

SARSA converge **mucho más rápido** que Monte Carlo porque:
- Actualiza después de cada paso (no al final del episodio)
- Propaga el conocimiento inmediatamente

Con 10,000 episodios, SARSA alcanza ~95% de éxito.

---

## SARSA vs Monte Carlo

| Aspecto | SARSA | Monte Carlo |
|---------|-------|-------------|
| **Actualización** | Cada paso (TD) | Fin de episodio |
| **Alpha óptimo** | 0.1 | 0.01 |
| **Tolerancia a α alto** | ✓ Buena | ✗ Catastrófico |
| **Velocidad** | Rápido | Lento |
| **Episodios necesarios** | ~10K | ~15K |
| **Tasa de éxito típica** | ~95% | ~65% |

### ¿Por qué SARSA es más robusto?

1. **Error TD pequeño**: Solo considera un paso adelante, no todo el episodio
2. **On-policy**: Aprende de lo que realmente hace (incluido el riesgo de explorar)
3. **Propagación rápida**: El conocimiento se difunde inmediatamente

---

## Configuración Final Recomendada

```python
SarsaAgent(
    alpha=0.1,      # Balance velocidad/estabilidad
    gamma=0.99,     # Alta importancia al futuro
    epsilon=0.01,   # Mínima exploración
)
# Entrenar con 10,000 episodios
# max_steps=500 para evitar bucles
```

### Resultados Esperados
- **Recompensa (últ. 100)**: ~-22 a -28
- **Tasa de Éxito**: ~95%
- **Tiempo**: ~4-6 segundos

---

*Documento generado automáticamente basado en estudios de hiperparámetros*
