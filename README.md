# Regresión Lineal con Descenso de Gradiente

Este repositorio contiene una implementación de **descenso de gradiente** usando dos enfoques distintos para el entrenamiento del modelo:  
1. **Implementación manual** del algoritmo de descenso de gradiente.  
2. **Uso de librerías** (Scikit-Learn con `SGDRegressor`).  

El objetivo es comparar ambos métodos sobre un dataset sintético generado con NumPy.

---

## Estructura del repositorio

- **[`gradiente_manual.py`](gradiente_manual.py)**  
  Contiene la implementación "desde cero" del descenso de gradiente para ajustar los parámetros de una recta a datos sintéticos.

- **[`gradiente_librerias.py`](gradiente_librerias.py)**  
  Implementación usando `SGDRegressor` de Scikit-Learn. Incluye escalamiento de datos y visualización de los resultados.  

- **[`implementacion.ipynb`](implementacion.ipynb)**  
  Notebook principal que integra ambos métodos (`gradiente_manual` y `gradiente_librerias`). Aquí se genera el dataset, se configuran hiperparámetros (α y número de iteraciones), y se muestran los resultados comparativos.

---

##  Requisitos

Para ejecutar los scripts y el notebook, se recomienda crear un entorno de Python (>=3.9) e instalar las siguientes dependencias:

```bash
pip install numpy matplotlib scikit-learn
```

## Uso

1. Clonar el repositorio
```bash
git clone https://github.com/Valeria-Aguilar/Portafolio-Implementation.git
cd Portafolio-Implementation
```

2. Ejecutar el Notebook
Abre el notebook y corre los ejemplos paso a paso:
```bash
jupyter notebook implementacion.ipynb
```
4. Ejecutar desde consola (Python)
También puedes importar y correr las funciones directamente en un script o consola interactiva:
```python
import numpy as np
from gradiente_manual import gradiente_manual
from gradiente_librerias import gradiente_librerias

# Dataset sintético
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Hiperparámetros
alpha = 0.1
n_iterations = 1000

# Descenso de gradiente manual
gradiente_manual(X, y, alpha, n_iterations)

# Descenso de gradiente con librerías
gradiente_librerias(X, y, alpha, n_iterations)

```

