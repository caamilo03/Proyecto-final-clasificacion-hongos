# 🍄 Proyecto Final: Clasificación de Hongos

**Entrega final del proyecto para Modelos II**  
**Dataset:** UCI Mushroom Database  
**Objetivo:** Clasificar hongos como comestibles o venenosos usando 5 modelos de Machine Learning

---

## 📋 Descripción del Proyecto

Implementación de **5 modelos de clasificación** siguiendo la metodología del profesor con validación cruzada estratificada (StratifiedKFold):

1. ✅ **Regresión Logística** (Implementación manual) - 94.58% accuracy
2. ✅ **K-Nearest Neighbors (KNN)** - 100% accuracy
3. ✅ **Random Forest** - 100% accuracy  
4. ✅ **Red Neuronal Artificial (MLP)** - 100% accuracy
5. ⏳ **Support Vector Machine (SVM)** - Pendiente

---

## 📊 Dataset

- **Fuente:** UCI Machine Learning Repository - Mushroom Database
- **Archivo:** `dataset_24_mushroom.arff`
- **Muestras:** 8,124 hongos
- **Características:** 22 atributos categóricas (cap-shape, odor, gill-color, etc.)
- **Clases:** 
  - 0 = Comestible (edible)
  - 1 = Venenoso (poisonous)
- **División:** 70% entrenamiento, 30% prueba

---

## 🚀 Resultados Actuales

| Modelo | Técnica | Accuracy | Parámetros Óptimos | Validación |
|--------|---------|----------|-------------------|------------|
| 1 | Regresión Logística | 94.58% | η=10.0 | Manual |
| 2 | KNN | 100% | k=1 | StratifiedKFold (4 folds) |
| 3 | Random Forest | 100% | 50 trees, 5 vars | StratifiedKFold (4 folds) |
| 4 | MLP | 100% | (10,) - 1 capa, 10 neuronas | StratifiedKFold (4 folds) + K-fold robustness (3-20 folds) |
| 5 | SVM | Pendiente | - | - |

**Nota:** El dataset UCI Mushroom es perfectamente separable. Los resultados de 100% accuracy están validados y son correctos.

---

## 📁 Estructura del Proyecto

```
Proyecto-final-clasificación de hongos/
│
├── Clasificacion_Hongos_5_Modelos.ipynb  # Notebook principal (51 celdas)
├── dataset_24_mushroom.arff              # Dataset UCI
├── README.md                             # Este archivo
└── RESUMEN_PROGRESO.md                   # Resumen detallado para compartir
```

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Librerías:**
  - NumPy (operaciones matemáticas)
  - Pandas (manipulación de datos)
  - Matplotlib & Seaborn (visualización)
  - scikit-learn (KNN, Random Forest, MLP, PCA, StandardScaler)
  - SciPy (lectura ARFF)

---

## 📈 Metodología

### Implementación siguiendo patrones del profesor:

1. **Funciones `experimentar_X()`** personalizadas para cada modelo
2. **StratifiedKFold(n_splits=4)** para validación cruzada
3. **DataFrames de resultados** con métricas:
   - Error de entrenamiento (media)
   - Desviación estándar entrenamiento
   - Error de prueba (media)
   - Intervalo de confianza
4. **Visualizaciones 2D/3D con PCA** para cada modelo
5. **Matrices de confusión** (entrenamiento y prueba)

---

## 🎯 Hallazgos Principales

### Dataset Perfectamente Separable
- Características altamente discriminativas: `odor`, `gill-color`, `spore-print-color`
- Modelos modernos alcanzan 100% accuracy naturalmente
- **Validación robusta:** MLP validado con 53 entrenamientos adicionales (K-folds: 3, 5, 10, 15, 20)

### Análisis de Complejidad
- **Modelo más simple que alcanza 100%:** KNN con k=1
- **Red neuronal óptima:** 1 capa, 10 neuronas (convergencia en 113 iteraciones)
- **Random Forest óptimo:** 50 árboles, 5 variables

### Regresión Logística
- **94.58% accuracy** - razonable para modelo lineal en problema no-lineal
- Implementación manual completa (gradiente descendente, función de costo)
- Frontera de decisión lineal visualizada

---

## 📊 Visualizaciones Incluidas

Cada modelo incluye:
- Matriz de confusión (Train/Test)
- Visualización 2D con PCA
- Visualización 3D con PCA
- Gráficas de métricas específicas

**Especiales:**
- **Modelo 1:** Evolución del costo
- **Modelo 3:** Importancia de características
- **Modelo 4:** Heatmap comparativo, validación K-fold

---

## 🔬 Próximos Pasos

- [ ] Implementar Modelo 5: Support Vector Machine (SVM)
- [ ] Tabla comparativa final de los 5 modelos
- [ ] Análisis de tiempo de entrenamiento
- [ ] Recomendaciones para deployment
- [ ] Conclusiones finales del proyecto

---

## 📝 Notas del Desarrollo

### ¿Por qué 100% accuracy es válido?
- Dataset UCI Mushroom documentado como perfectamente separable
- Validación exhaustiva con múltiples configuraciones de K-fold
- Gap Train-Test = 0% (no hay sobreajuste)
- Confusion matrices perfectas: sin falsos positivos/negativos

### Validaciones realizadas:
✅ StratifiedKFold con 4 folds en todos los modelos  
✅ Validación adicional MLP: 3, 5, 10, 15, 20 folds (53 entrenamientos)  
✅ Métricas perfectas: Precision=100%, Recall=100%, F1=100%  
✅ PCA 2D/3D confirma separabilidad visual  

---

## 👨‍💻 Autor

**Camilo**  
Curso: Modelos II  
Fecha: Noviembre 2025

---

## 📄 Licencia

Proyecto académico - Universidad

---

**Última actualización:** 20 de Noviembre de 2025  
**Estado:** 4/5 modelos completados ✅  
**Notebook ejecutado:** ✅ Todas las celdas funcionando
