# Teoría de Aprendizaje por Refuerzo

## 1. Diferencias entre Algoritmos

### On-Policy vs Off-Policy

| Característica | On-Policy (SARSA) | Off-Policy (Q-Learning) |
|----------------|-------------------|-------------------------|
| **Política de comportamiento** | = Política objetivo | ≠ Política objetivo |
| **Qué aprende** | Lo que realmente hace | Lo óptimo (ignorando exploración) |
| **Actualización** | Q(s',a') (acción tomada) | max Q(s',a') (mejor acción) |
| **Comportamiento** | Conservador | Arriesgado |

### Fórmulas de Actualización

**SARSA:**
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                        ↑
                 Acción que REALMENTE tomó
```

**Q-Learning:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
                        ↑
                 MEJOR acción posible
```

### ¿Por qué SARSA es conservador?

SARSA aprende considerando que **explorará** con probabilidad ε. Si está cerca del acantilado y puede explorar, la acción aleatoria podría hacerlo caer. Por eso aprende a alejarse.

Q-Learning ignora la exploración en el aprendizaje. Asume que siempre tomará la mejor acción, así que no le importa estar cerca del acantilado.

---

## 2. Temporal Difference (TD) vs Monte Carlo

| Aspecto | TD (SARSA/Q-Learning) | Monte Carlo |
|---------|----------------------|-------------|
| **Cuándo actualiza** | Cada paso | Fin del episodio |
| **Qué usa** | r + γQ(s',a') (bootstrap) | G (retorno real) |
| **Sesgo** | Tiene (usa estimación) | No tiene |
| **Varianza** | Baja | Alta |
| **Velocidad** | Rápida | Lenta |

### Bootstrapping

TD "adivina" el valor futuro usando su propia estimación (Q-table). Monte Carlo espera hasta ver el resultado real.

```
TD:    V(s) ← V(s) + α[r + γV(s') - V(s)]
              ↑           ↑
         Recompensa   Estimación del futuro
              real       (bootstrap)

MC:    V(s) ← V(s) + α[G - V(s)]
                       ↑
               Retorno REAL del episodio
```

---

## 3. Exploración vs Explotación

### El Dilema

- **Explorar**: Probar acciones nuevas para descubrir mejores opciones
- **Explotar**: Usar el conocimiento actual para maximizar recompensa

### Estrategia ε-greedy

```
Con probabilidad ε:     Acción aleatoria (explorar)
Con probabilidad 1-ε:   Mejor acción conocida (explotar)
```

### Valores típicos de ε

| Valor | Comportamiento |
|-------|---------------|
| ε=1.0 | 100% aleatorio |
| ε=0.5 | 50% aleatorio |
| ε=0.1 | 10% aleatorio (común) |
| ε=0.01 | Casi siempre explota |
| ε=0.0 | 100% greedy |

### Epsilon Decay

Empezar con alta exploración y reducirla gradualmente:
```
ε = max(ε_min, ε_inicial * decay^episodio)
```

---

## 4. La Q-Table

### ¿Qué es?

Una tabla que almacena el **valor esperado** de tomar cada acción en cada estado.

```
Q[estado][acción] = Recompensa esperada total si:
                    1. Estoy en 'estado'
                    2. Tomo 'acción'
                    3. Sigo la política óptima después
```

### Ejemplo en CliffWalking

```
Estado 36 (inicio):
  Q[36][↑] = -15.2  ← Subir es bueno
  Q[36][→] = -100   ← Ir al acantilado es malo
  Q[36][↓] = -50    ← Inválido
  Q[36][←] = -50    ← Inválido
```

### Extracción de Política

```python
mejor_accion = argmax(Q[estado])  # Acción con mayor valor Q
```

---

## 5. Parámetros Clave

### Alpha (α) - Learning Rate

**Qué hace**: Cuánto confiamos en la nueva información vs conocimiento previo.

```
α = 0:    Nunca aprende (ignora nueva info)
α = 1:    Solo usa nueva info (olvida todo)
α = 0.1:  Balance típico
```

**Fórmula**:
```
Q_nuevo = Q_viejo + α * (objetivo - Q_viejo)
        = (1-α)*Q_viejo + α*objetivo
```

### Gamma (γ) - Factor de Descuento

**Qué hace**: Cuánto le importa el futuro vs presente.

```
γ = 0:    Solo le importa recompensa inmediata (miope)
γ = 1:    Valora igual futuro y presente
γ = 0.99: Valora mucho el futuro (común)
```

**Ejemplo**:
```
Recompensas: [r0, r1, r2, r3, ...]
Retorno: G = r0 + γ*r1 + γ²*r2 + γ³*r3 + ...

Con γ=0.9:  G = r0 + 0.9*r1 + 0.81*r2 + 0.73*r3 + ...
Con γ=0.5:  G = r0 + 0.5*r1 + 0.25*r2 + 0.125*r3 + ...
```

### Epsilon (ε) - Exploración

**Qué hace**: Probabilidad de tomar acción aleatoria.

```
ε alto:   Mucha exploración, descubre más, pero ineficiente
ε bajo:   Poca exploración, eficiente, pero puede quedarse en óptimo local
```

---

## 6. Convergencia

### Condiciones para Garantizar Convergencia

1. **Visitar todos los pares (s,a)** infinitas veces
2. **Learning rate decrece** pero no demasiado rápido
3. **Entorno es MDP** (Markov Decision Process)

### Velocidad de Convergencia

```
Monte Carlo:  ~15,000 episodios
SARSA:        ~5,000 episodios  
Q-Learning:   ~5,000 episodios
```

¿Por qué TD es más rápido?
- Actualiza cada paso (más actualizaciones)
- Propaga información inmediatamente
- Menor varianza en las estimaciones

---

## 7. CliffWalking Específico

### Entorno

```
    0   1   2   3   4   5   6   7   8   9  10  11
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
0 │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │
1 │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │
2 │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │
3 │ S │ C │ C │ C │ C │ C │ C │ C │ C │ C │ C │ G │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

### Recompensas

| Evento | Recompensa |
|--------|------------|
| Cada paso | -1 |
| Caer al acantilado | -100 + volver a S |
| Llegar a G | 0 (fin) |

### Caminos

- **Óptimo teórico**: S→↑→→→→→→→→→→→↓→G = 13 pasos = -13
- **Seguro**: S→↑→↑→→→→→→→→→→↓↓→G = 15+ pasos
