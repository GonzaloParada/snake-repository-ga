# Snake Game - Genetic Algorithm AI

Un proyecto de inteligencia artificial que entrena una red neuronal para jugar Snake usando algoritmos genéticos.

## 🎮 Características

- **Juego Snake completo** implementado con Pygame
- **Red neuronal** para controlar la serpiente
- **Algoritmo genético** para entrenar la IA
- **Múltiples modos de juego**: IA entrenada, humano, demo aleatorio
- **Entrenamiento paralelo** usando multiprocesamiento
- **Guardado/carga de modelos** entrenados
- **Interfaz de línea de comandos** fácil de usar

## 🚀 Instalación

1. Clona o descarga este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## 📖 Uso

### Entrenar una nueva IA

```bash
# Entrenamiento básico (50 generaciones, población de 100)
python main.py train

# Entrenamiento personalizado
python main.py train --generations 100 --population-size 150 --games-per-individual 5
```

**Parámetros de entrenamiento:**
- `--generations`: Número de generaciones (default: 50)
- `--population-size`: Tamaño de la población (default: 100)
- `--games-per-individual`: Juegos por individuo para evaluación (default: 3)
- `--mutation-rate`: Tasa de mutación (default: 0.1)
- `--mutation-strength`: Fuerza de mutación (default: 0.5)
- `--elite-percentage`: Porcentaje de élite (default: 0.2)
- `--crossover-rate`: Tasa de cruzamiento (default: 0.8)
- `--no-multiprocessing`: Deshabilitar multiprocesamiento

### Jugar con IA entrenada

```bash
# Jugar con el mejor modelo
python main.py play

# Jugar con un modelo específico
python main.py play models/best_snake_gen_20.pkl

# Ejecutar múltiples juegos sin visualización (para estadísticas)
python main.py play --games 10 --no-visual
```

### Otros modos

```bash
# Jugar como humano (usa las flechas del teclado)
python main.py human

# Demo con IA aleatoria
python main.py demo

# Listar modelos disponibles
python main.py list
```

## 🧠 Arquitectura de la IA

### Red Neuronal
- **Entrada**: 11 características del estado del juego
  - Peligro adelante, izquierda, derecha
  - Dirección actual (4 booleanos)
  - Dirección de la comida (4 booleanos)
- **Capas ocultas**: [16, 12] neuronas (configurable)
- **Salida**: 3 acciones (recto, girar derecha, girar izquierda)
- **Activación**: ReLU para capas ocultas, Softmax para salida

### Algoritmo Genético
- **Selección**: Torneo de 5 individuos
- **Cruzamiento**: Un punto de cruce
- **Mutación**: Ruido gaussiano
- **Elitismo**: Mantiene el 20% de los mejores individuos
- **Función de fitness**: Combina puntuación y supervivencia

## 📁 Estructura del Proyecto

```
snake-game-ga/
├── main.py              # Script principal
├── snake_game.py        # Implementación del juego Snake
├── neural_network.py    # Red neuronal y clase SnakeAI
├── genetic_algorithm.py # Algoritmo genético
├── requirements.txt     # Dependencias
├── README.md           # Este archivo
└── models/             # Modelos entrenados (se crea automáticamente)
    ├── best_snake_final.pkl
    ├── best_snake_gen_*.pkl
    └── training_history.pkl
```

## 🎯 Función de Fitness

La función de fitness recompensa:
- **Comer comida**: +100 puntos por cada comida + bonificación exponencial
- **Supervivencia**: +0.1 puntos por cada paso
- **Penalización por colisión**: -10 puntos

```python
fitness = score * 100 + steps * 0.1
if score > 0:
    fitness += score ** 2 * 10
```

## 🔧 Personalización

### Modificar la red neuronal
Edita `neural_network.py` para cambiar la arquitectura:

```python
nn = NeuralNetwork(
    input_size=11,
    hidden_sizes=[20, 16, 12],  # Más capas/neuronas
    output_size=3
)
```

### Ajustar parámetros del juego
Modifica `snake_game.py`:

```python
game = SnakeGame(
    width=800,      # Ancho de la ventana
    height=600,     # Alto de la ventana
    block_size=20,  # Tamaño de cada bloque
    speed=10        # Velocidad del juego
)
```

## 📊 Monitoreo del Entrenamiento

Durante el entrenamiento verás:
```
Generation 0: Best=245.30, Avg=89.45
Generation 1: Best=312.80, Avg=156.23
...
```

Los modelos se guardan automáticamente:
- Cada 10 generaciones: `models/best_snake_gen_X.pkl`
- Al final: `models/best_snake_final.pkl`
- Historial: `models/training_history.pkl`

## 🚀 Consejos para Mejor Rendimiento

1. **Entrenamiento más largo**: Usa más generaciones (100-200)
2. **Población más grande**: Aumenta `--population-size` a 200-300
3. **Más evaluaciones**: Incrementa `--games-per-individual` a 5-10
4. **Ajustar mutación**: Experimenta con diferentes `--mutation-rate` (0.05-0.2)

## 🐛 Solución de Problemas

### Error de multiprocesamiento
Si tienes problemas con multiprocesamiento:
```bash
python main.py train --no-multiprocessing
```

### Modelo no encontrado
Asegúrate de entrenar un modelo primero:
```bash
python main.py train
python main.py play
```

### Pygame no se cierra correctamente
En algunos sistemas, cierra manualmente la ventana o usa Ctrl+C.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Algunas ideas:
- Mejores funciones de fitness
- Diferentes arquitecturas de red neuronal
- Visualización del entrenamiento en tiempo real
- Algoritmos de selección alternativos
- Interfaz gráfica para configuración

## 📝 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo, modificarlo y distribuirlo.

## 🎉 ¡Diviértete!

¡Experimenta con diferentes parámetros y observa cómo evoluciona tu IA Snake! 🐍🧠
