# Monte Carlo Óptimo - Justificación de Parámetros

## Resumen de Configuración Óptima

| Parámetro | Valor | Rango Probado |
|-----------|-------|---------------|
| **Alpha (α)** | 0.01 | 0.01 - 0.9 |
| **Gamma (γ)** | 0.99 | 0.5 - 1.0 |
| **Epsilon (ε)** | 0.01 | 0.01 - 0.5 + Decay |
| **Episodios** | 15,000 | 10K - 30K |

---

## Alpha (α) = 0.01 - Learning Rate

### ¿Qué hace?
Controla cuánto "confiamos" en la nueva información vs el conocimiento previo.

### Fórmula
```
Q(s,a) ← Q(s,a) + α * [G - Q(s,a)]
```

### ¿Por qué 0.01?

| Alpha | Avg últ.100 | Éxito | Problema |
|-------|-------------|-------|----------|
| 0.01 | **-40.95** | **80.5%** | ✓ Óptimo |
| 0.1 | -107.56 | 73.0% | Más varianza |
| 0.5 | -140.53 | 28.0% | Inestable |
| 0.9 | -1311.51 | 9.2% | Catastrófico |

**Conclusión**: En Monte Carlo, el retorno G se calcula al final del episodio y puede ser muy negativo. Un alpha alto sobreescribe el conocimiento anterior demasiado rápido, causando "olvido catastrófico".

---

## Gamma (γ) = 0.99 - Factor de Descuento

### ¿Qué hace?
Determina cuánto le importa al agente el futuro vs las recompensas inmediatas.

### Fórmula
```
G = r₁ + γ*r₂ + γ²*r₃ + ... + γⁿ*rₙ
```

### ¿Por qué 0.99?

| Gamma | Avg últ.100 | Éxito | Comportamiento |
|-------|-------------|-------|----------------|
| 0.5 | -570.54 | 12.8% | "Miope" - no planifica |
| 0.7 | -540.59 | 22.6% | Cortoplacista |
| 0.9 | -481.41 | 37.1% | Insuficiente |
| **0.99** | -273.71 | **65.3%** | ✓ Planifica bien |
| 1.0 | -54.03 | 57.1% | Inestable |

**Conclusión**: En CliffWalking, el camino seguro (arriba-derecha-abajo) es más largo pero evita el acantilado. Un gamma alto hace que el agente valore llegar al final sin caer, en lugar de buscar el camino más corto (y peligroso).

---

## Epsilon (ε) = 0.01 - Exploración

### ¿Qué hace?
Probabilidad de tomar una acción aleatoria (explorar) en lugar de seguir la política aprendida (explotar).

### ¿Por qué 0.01?

| Epsilon | Avg últ.100 | Éxito | Problema |
|---------|-------------|-------|----------|
| **0.01** | **-27.85** | **63.2%** | ✓ Óptimo |
| 0.1 | -50.98 | 62.9% | Más caídas |
| 0.3 | -316.49 | 32.3% | Demasiada exploración |
| 0.5 | -447.32 | 7.7% | Casi aleatorio |

**Conclusión**: Para Monte Carlo, la exploración es muy costosa porque el agente completa todo el episodio antes de aprender. Caer al acantilado por exploración penaliza todo el episodio.

### ¿Y el Epsilon Decay?
El decay (1.0→0.01) tuvo resultados mixtos:
- **Ventaja**: Explora mucho al principio, aprende el espacio
- **Desventaja**: Pierde tiempo con alta exploración inicial

Para CliffWalking, ε=0.01 fijo funciona mejor porque el espacio es pequeño (48 estados).

---

## Episodios = 15,000

### ¿Por qué este número?

| Episodios | Convergencia | Tiempo |
|-----------|--------------|--------|
| 5,000 | Incompleta | ~10s |
| 10,000 | Buena | ~18s |
| **15,000** | **Óptima** | ~25s |
| 30,000 | Marginal mejora | ~50s |

Con α=0.01 (aprendizaje lento), se necesitan más episodios para convergencia completa. 15,000 es el punto óptimo costo/beneficio.

---

## Resumen Visual

```
IMPORTANCIA DE CADA PARÁMETRO:

Alpha (Learning Rate)
├─ Muy bajo (0.01): ████████████████████ Estable, lento
├─ Medio (0.1):     ████████████         Balance
└─ Alto (0.5+):     ████                 Inestable

Gamma (Descuento)
├─ Bajo (0.5-0.7):  ████                 Miope
├─ Medio (0.9):     ████████             Insuficiente
└─ Alto (0.99):     ████████████████████ Planifica bien

Epsilon (Exploración)
├─ Muy bajo (0.01): ████████████████████ Estable
├─ Bajo (0.1):      ████████████████     Bueno
└─ Alto (0.3+):     ████                 Caótico
```

---

## Configuración Final Recomendada

```python
MonteCarloAgent(
    alpha=0.01,     # Aprendizaje muy gradual
    gamma=0.99,     # Alta importancia al futuro
    epsilon=0.01,   # Mínima exploración
)
# Entrenar con 15,000 episodios
# max_steps=500 para evitar bucles
```

---

*Documento generado automáticamente basado en estudios de hiperparámetros*
