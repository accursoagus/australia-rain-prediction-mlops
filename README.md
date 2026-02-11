# Predicción de Lluvia en Australia: E2E ML Pipeline & MLOps

Este proyecto implementa un flujo de trabajo completo (End-to-End) de Machine Learning para predecir la probabilidad de lluvia en Australia al día siguiente.

El foco no estuvo solo en el modelado, sino en la ingeniería de características, la interpretabilidad del modelo y el despliegue containerizado listo para producción.

## Características Principales

### 1. Ingeniería de Características (Feature Engineering)
* **Clustering Geoespacial:** Se aplicó K-Means para agrupar ubicaciones en regiones climáticas, reduciendo la dimensionalidad de la variable `Location`.
* **Manejo de Temporalidad:** Transformación de fechas en variables cíclicas y estacionales para capturar patrones climáticos.
* **Imputación Avanzada:** Se utilizó IterativeImputer (MICE) para completar valores faltantes basándose en la correlación multivariada entre variables, método superior a la imputación simple por media o mediana.

### 2. Modelado y Optimización
* **Selección de Modelos:** Se realizó un benchmark comparativo utilizando PyCaret (AutoML) y modelos personalizados.
* **Redes Neuronales (Deep Learning):** Arquitectura densa en TensorFlow/Keras con capas de Dropout y Early Stopping para prevenir el sobreajuste. Este modelo superó a la Regresión Logística y LightGBM, logrando un AUC-ROC de 0.90.
* **Optimización Bayesiana:** Ajuste fino de hiperparámetros con Optuna para maximizar el rendimiento del modelo.

### 3. Interpretabilidad (XAI)
* **SHAP Values:** Análisis de las predicciones de la red neuronal ("caja negra") para entender qué variables (como Humedad a las 3pm o Presión) tienen mayor peso en la decisión del modelo.

### 4. MLOps y Despliegue
* **Docker:** El entorno de inferencia se encuentra totalmente containerizado. Esto asegura la reproducibilidad y elimina problemas de versiones o dependencias entre distintos entornos.
* **Script de Inferencia:** Pipeline automatizado (`inferencia.py`) que carga el preprocesador (`joblib`) y el modelo (`.h5`) para realizar predicciones sobre nuevos datos crudos.

## Stack Tecnológico
* **Lenguaje:** Python 3.9
* **Deep Learning:** TensorFlow, Keras
* **AutoML & Tuning:** PyCaret, Optuna
* **Procesamiento:** Pandas, Scikit-learn (Pipeline, ColumnTransformer)
* **Infraestructura:** Docker

## Estructura del Proyecto
* `TP-clasificacion-AA1_Entrega_3.ipynb`: Notebook con el entrenamiento, EDA y validación.
* `Dockerfile`: Configuración para construir la imagen del contenedor.
* `inferencia.py`: Script de punto de entrada para realizar predicciones en producción.
* `best_model.h5`: Modelo de Red Neuronal entrenado y serializado.
* `preprocessor.joblib`: Pipeline de preprocesamiento ajustado.

## Ejecución con Docker

Para levantar la imagen y probar el modelo en un entorno aislado:

```bash
# 1. Construir la imagen
docker build -t rain-predictor .

# 2. Ejecutar el contenedor (realizará una inferencia de prueba)
docker run rain-predictor
