# TP Aprendizaje Automático 1 - Clasificación: Predicción de Lluvia

Este contenedor encapsula el modelo de Red Neuronal optimizado para predecir la lluvia en Australia.

## Contenido

* **inferencia.py**: Script que carga el modelo y realiza la predicción.
* **best\_model.keras**: Modelo de Red Neuronal entrenado con TensorFlow/Keras.
* **preprocessor.joblib**: Pipeline de scikit-learn para imputación y escalado.

## Instrucciones de Ejecución

### 1\. Construir la imagen de Docker

Abrir la terminal en esta carpeta y ejecutar:

docker build -t prediccion-lluvia .



### 2\. Ejecutar el contenedor

Para correr una predicción de prueba:

docker run --rm prediccion-lluvia

El script imprimirá en consola el resultado de la predicción 

