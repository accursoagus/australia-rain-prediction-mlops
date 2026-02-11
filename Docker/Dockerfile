# 1. Usamos una imagen base oficial de Python (liviana)
FROM python:3.9-slim

# 2. Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos los requerimientos e instalamos las dependencias
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# 4. Copiamos el código y los modelos al contenedor
COPY inferencia.py .
COPY preprocessor.joblib .
COPY best_model.h5 .

# 5. Comando por defecto para ejecutar el contenedor
# Ejecutamos el script de inferencia
CMD ["python", "inferencia.py"]