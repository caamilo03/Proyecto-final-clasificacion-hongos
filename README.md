# Proyecto Final: Clasificación de Hongos

**Entrega final del proyecto para Modelos II**  
**Dataset:** UCI Mushroom Database  
**Objetivo:** Clasificar hongos como comestibles o venenosos usando 5 modelos de Machine Learning y técnicas de reducción de dimensionalidad

---
## Link del vídeo: https://youtu.be/jaczX1fN5Sc


## Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para clasificación binaria de hongos (comestibles vs venenosos) utilizando el dataset UCI Mushroom. Se implementan **5 modelos de clasificación** con validación cruzada estratificada y se exploran **técnicas de reducción de dimensionalidad** (PCA y UMAP) para analizar el rendimiento con menos características.

### Modelos Implementados:

1. **Regresión Logística** 
2. **K-Nearest Neighbors (k-NN)** 
3. **Random Forest** 
4. **Support Vector Machine (SVM)** 
5. **Red Neuronal Artificial (MLP)** 

### Técnicas de Reducción de Dimensionalidad:

- **PCA (Principal Component Analysis)** - 95% y 99% varianza
- **UMAP (Uniform Manifold Approximation and Projection)** - 2, 5, 10, 20 componentes

---

## Dataset

- **Fuente:** UCI Machine Learning Repository - Mushroom Database
- **Archivo:** `dataset_24_mushroom.arff`
- **Muestras:** 8,124 hongos
- **Características:** 22 atributos categóricos (cap-shape, odor, gill-color, etc.)
- **Variable objetivo (class):**
  - **0 = Comestible (edible)** - Clase negativa
  - **1 = Venenoso (poisonous)** - Clase positiva (IMPORTANTE)
- **División:** 70% entrenamiento (5,686 muestras), 30% prueba (2,438 muestras)
- **Estratificación:** Preserva proporción de clases en train/test

### Codificación de Variables:

El proyecto utiliza **dos esquemas de codificación** según el modelo:

1. **Label Encoding** (22 variables):
   - Para: Random Forest
   - Convierte categorías a números ordinales (0, 1, 2, ...)
   
2. **One-Hot Encoding** (117 variables):
   - Para: Regresión Logística, k-NN, SVM, MLP
   - Crea columnas binarias para cada categoría

---


## Cómo Ejecutar el Proyecto

### Opción 1: Google Colab (Recomendado)

1. **Abrir en Colab:**
   - Ve a [Google Colab](https://colab.research.google.com/)
   - Sube el archivo `Clasificación_Hongos.ipynb` o clona el repositorio

2. **Ejecutar todo el notebook:**
   ```python
   # Opción A: Usar el menú
   Runtime → Run all
   
   
   ```

3. **Instalación automática de UMAP:**
   - El notebook instala automáticamente `umap-learn` si no está disponible
   - No requiere configuración adicional

### Opción 2: Ejecución Local

#### Prerrequisitos:

```bash
# Python 3.8 o superior
python --version

# Instalar Jupyter
pip install jupyter notebook
# O Jupyter Lab
pip install jupyterlab
```

#### Instalación de dependencias:

```bash
# Navegar al directorio del proyecto
cd "Proyecto-final-clasificación de hongos"

# Instalar librerías necesarias
pip install numpy pandas matplotlib seaborn scikit-learn scipy umap-learn

# Versiones recomendadas:
# - scikit-learn >= 1.3, < 1.7
# - umap-learn >= 0.5.0
```

#### Ejecutar el notebook:

```bash
# Opción A: Jupyter Notebook
jupyter notebook Clasificación_Hongos.ipynb

# Opción B: Jupyter Lab
jupyter lab Clasificación_Hongos.ipynb

# Opción C: VS Code
# Abre el archivo .ipynb directamente en VS Code con la extensión Python
```

#### Ejecutar todas las celdas:

1. En Jupyter Notebook/Lab: `Cell → Run All`
2. En VS Code: `Run All` en la parte superior del notebook

### Opción 3: Ejecutar desde Python Script

```python
# Convertir notebook a script (opcional)
jupyter nbconvert --to script Clasificación_Hongos.ipynb

# Ejecutar como script Python
python Clasificación_Hongos.py
```

---

## Tiempo de Ejecución

**Tiempo total aproximado:** ~15-20 minutos (depende del hardware)

| Sección | Tiempo Estimado |
|---------|----------------|
| Carga y preprocesamiento | < 1 min |
| Modelo 1 (Regresión Logística) | ~1 min |
| Modelo 2 (k-NN + validación) | ~6 min |
| Modelo 3 (Random Forest + validación) | ~1 min |
| Modelo 4 (SVM + validación) | ~2 min |
| Modelo 5 (MLP + validación) | ~7 min |
| PCA (análisis y entrenamiento) | ~1 min |
| UMAP (análisis y entrenamiento) | ~3 min |

**Nota:** Los tiempos pueden variar según CPU. UMAP requiere más tiempo en la primera ejecución.

---

## Estructura del Proyecto

```
Proyecto-final-clasificación de hongos/
│
├── Clasificación_Hongos.ipynb           # Notebook principal COMPLETO (68 celdas)
├── Clasificacion_Hongos_5_Modelos.ipynb # Versión anterior (referencia)
├── dataset_24_mushroom.arff             # Dataset UCI Mushroom
├── README.md                            # Este archivo
└── RESUMEN_PROGRESO.md                  # Historial de desarrollo
```

### Estructura del Notebook:

**Sección 0: Preparación de Datos (Celdas 1-9)**
- Carga del dataset desde GitHub
- Análisis exploratorio
- Label Encoding y One-Hot Encoding
- Split estratificado 70/30

**Sección 1: Regresión Logística (Celdas 10-15)**
- Implementación manual
- Gradiente descendente
- Visualización de frontera de decisión

**Sección 2: K-Nearest Neighbors (Celdas 16-19)**
- Optimización de k con validación cruzada
- Evaluación con k=3 óptimo

**Sección 3: Random Forest (Celdas 20-23)**
- Grid search de n_estimators y max_features
- Análisis de importancia de variables

**Sección 4: Support Vector Machine (Celdas 24-28)**
- Comparación de kernels (linear, poly, rbf, sigmoid)
- Optimización de C y gamma

**Sección 5: Redes Neuronales MLP (Celdas 29-32)**
- Búsqueda de arquitectura óptima
- Validación con múltiples configuraciones

**Sección 6: Comparación de Modelos (Celdas 33-35)**
- Tabla resumen de los 5 modelos
- Análisis comparativo

**Sección 7: Reducción de Dimensionalidad (Celdas 36-68)**
- **PCA:** Análisis de varianza, aplicación a k-NN y RF
- **UMAP:** Reducción no-lineal, comparación con PCA
- Tablas y gráficas comparativas

---

## Tecnologías y Librerías

### Lenguaje:
- **Python 3.8+**

### Librerías principales:

```python
# Manipulación de datos
import numpy as np              # Operaciones matemáticas y arrays
import pandas as pd             # DataFrames y análisis de datos

# Visualización
import matplotlib.pyplot as plt # Gráficas
import seaborn as sns          # Visualizaciones estadísticas

# Machine Learning (scikit-learn)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

# Reducción dimensional no-lineal
from umap import UMAP

# Lectura de archivos ARFF
from scipy.io import arff
```





### Limitaciones

- Dataset sintético y perfectamente separable
- No representa complejidad del mundo real
- Características pre-procesadas y limpias
- No hay ruido ni valores atípicos significativos

---

## Referencias

- **Dataset:** [UCI Machine Learning Repository - Mushroom Database](https://archive.ics.uci.edu/ml/datasets/mushroom)
- **Repositorio:** [GitHub - Proyecto-final-clasificacion-hongos](https://github.com/caamilo03/Proyecto-final-clasificacion-hongos)
- **Documentación scikit-learn:** [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- **UMAP:** [UMAP Documentation](https://umap-learn.readthedocs.io/)

---





---

