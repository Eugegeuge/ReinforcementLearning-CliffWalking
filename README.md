# ✅ PROYECTO TERMINADO Y DOCUMENTACIÓN ENTREGADA

# 🏔️ Cliff Walking - Reinforcement Learning

Implementación y análisis comparativo de algoritmos de Aprendizaje por Refuerzo en el entorno **Cliff Walking** de Gymnasium.

**Práctica de Manipuladores - Grado en Ingeniería Robótica**

## � Resultados Principales

| Algoritmo | α óptimo | γ óptimo | ε óptimo | Tasa Éxito | Tiempo |
|-----------|----------|----------|----------|------------|--------|
| **SARSA** | 0.1 | 0.99 | 0.01 | **95.6%** | 1.7s |
| **Q-Learning** | 0.1 | 0.99 | 0.01 | **95.4%** | 2.1s |
| **Monte Carlo** | 0.01 | 0.99 | 0.01 | 66.4% | 17.2s |

## 🎯 Descripción

El objetivo es entrenar agentes que naveguen desde el inicio (S) hasta la meta (G) evitando el acantilado (C):

```
    0   1   2   3   4   5   6   7   8   9  10  11
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
0 │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
1 │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
2 │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │ · │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
3 │ S │ C │ C │ C │ C │ C │ C │ C │ C │ C │ C │ G │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  
  S = Start | G = Goal | C = Cliff (caer = -100)
```

**Entorno estocástico**: 10% de probabilidad de acción aleatoria (slippery).

## 🧠 Algoritmos Implementados

### SARSA (On-policy TD)
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```
- Aprende de la acción que **realmente toma** (incluyendo exploración)
- Más conservador, evita el acantilado

### Q-Learning (Off-policy TD)
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
```
- Aprende la política **óptima** independiente de exploración
- Más agresivo, puede subestimar riesgos

### Monte Carlo (First-Visit)
```
G ← retorno acumulado desde el final del episodio
Q(s,a) ← Q(s,a) + α[G - Q(s,a)]
```
- Actualiza solo al **final del episodio**
- Alta varianza, lento en entornos estocásticos

## 📁 Estructura del Proyecto

```
├── src/
│   ├── agent.py              # Clases base de agentes RL
│   └── utils.py              # Utilidades y visualización
│
├── scripts/
│   ├── estudios/             # Estudios de hiperparámetros
│   │   ├── *_epsilon_study.py
│   │   ├── *_alpha_study.py
│   │   └── *_gamma_study.py
│   │
│   ├── optimos/              # Entrenamientos con config óptima
│   │   ├── montecarlo_optimo.py
│   │   ├── sarsa_optimo.py
│   │   └── qlearning_optimo.py
│   │
│   ├── comparaciones/        # Comparaciones entre modelos
│   └── utilidades/           # Scripts auxiliares
│
├── graphs/
│   ├── montecarlo/           # Gráficos + documentación MC
│   ├── sarsa/                # Gráficos + documentación SARSA
│   ├── qlearning/            # Gráficos + documentación Q-Learning
│   └── comparacion_todos/    # Comparaciones generales
│
└── main.py                   # Script principal
```

## 🚀 Instalación

```bash
git clone https://github.com/Eugegeuge/ReinforcementLearning-CliffWalking.git
cd ReinforcementLearning-CliffWalking
pip install -r requirements.txt
```

## 🛠️ Uso

### Ejecutar estudios de parámetros
```bash
python scripts/estudios/sarsa_epsilon_study.py
python scripts/estudios/montecarlo_alpha_study.py
```

### Entrenar con configuración óptima
```bash
python scripts/optimos/sarsa_optimo.py
python scripts/optimos/qlearning_optimo.py
python scripts/optimos/montecarlo_optimo.py
```

### Comparar todos los modelos
```bash
python scripts/comparaciones/run_full_training.py
```

## � Hallazgos Clave

### Por qué SARSA/Q-Learning superan a Monte Carlo en CliffWalking:

1. **Actualización paso a paso**: Los métodos TD propagan el conocimiento inmediatamente
2. **Menor varianza**: No acumulan error de todo el episodio
3. **Tolerancia a α más alto**: Pueden usar α=0.1 vs α=0.01 de MC

### Parámetros óptimos encontrados:

| Parámetro | SARSA | Q-Learning | Monte Carlo |
|-----------|-------|------------|-------------|
| **Alpha** | 0.1 | 0.1 | 0.01 |
| **Gamma** | 0.99 | 0.99 | 0.99 |
| **Epsilon** | 0.01 | 0.01 | 0.01 |
| **Episodios** | 10K | 10K | 15K |

## 📖 Documentación

Ver justificación detallada de cada parámetro en:
- [`graphs/sarsa/JUSTIFICACION_PARAMETROS_SARSA.md`](graphs/sarsa/JUSTIFICACION_PARAMETROS_SARSA.md)
- [`graphs/qlearning/JUSTIFICACION_PARAMETROS_QLEARNING.md`](graphs/qlearning/JUSTIFICACION_PARAMETROS_QLEARNING.md)
- [`graphs/montecarlo/JUSTIFICACION_PARAMETROS_MC.md`](graphs/montecarlo/JUSTIFICACION_PARAMETROS_MC.md)

## 👥 Autores

- Hugo Sevilla
- Hugo López
- Juan Diego Serrato

---

**Universidad de Alicante - Grado en Ingeniería Robótica**
