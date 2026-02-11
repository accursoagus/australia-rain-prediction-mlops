import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import sys
import os

def cargar_artefactos():
    try:
        # Cargamos modelo y preprocesador
        
        model = tf.keras.models.load_model('best_model.h5', compile=False) # compile=False evita cargar el optimizador, que nos daba error de versiones.
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        print(f"Error cargando artefactos: {e}")
        sys.exit(1)

def preprocesar_datos(df_entrada, preprocessor):

    df = df_entrada.copy()
    
    # Convertimos fecha
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df = df.drop(columns=['Date'])
    
    # Aplicamos el ColumnTransformer (Imputación + Escalar + OneHot)
    try:
        X_transformado = preprocessor.transform(df)
    except Exception as e:
        raise ValueError(f"Error en transformación de datos: {e}")

    # Convertimos a denso. El preprocesador devolvía matrices dispersas, así que usamos toarray() para convertirlas en matrices densas
    if hasattr(X_transformado, 'toarray'):
        X_transformado = X_transformado.toarray()
        
    return X_transformado

def predecir(datos_dict):
    model, preprocessor = cargar_artefactos()
    
    # Convertimos diccionario a DataFrame
    df_entrada = pd.DataFrame([datos_dict])
    
    # Preprocesamos
    try:
        X_listo = preprocesar_datos(df_entrada, preprocessor)
    except Exception as e:
        return {"error": str(e)}

    # Inferencia
    try:
        # .predict devuelve probabilidad (0 a 1)
        probabilidad = model.predict(X_listo, verbose=0).flatten()[0]
        
        # Aplicamos umbral de 0.44
        clase = 1 if probabilidad > 0.44 else 0
        resultado_texto = "SI Llueve" if clase == 1 else "NO Llueve"
        
        return {
            "Prediccion": resultado_texto,
            "Probabilidad": float(probabilidad),
            "Información": "ALERTA DE LLUVIA" if clase == 1 else "Clima estable"
        }
    except Exception as e:
        return {"error": f"Fallo en inferencia: {e}"}

if __name__ == "__main__":
    # Simulación de datos
    dato_ejemplo = {
        'Date': '2023-11-24',
        'Location': 'Canberra',
        'MinTemp': 13.0,
        'MaxTemp': 25.0,
        'Rainfall': 0.0,
        'Evaporation': 5.0,
        'Sunshine': 10.0,
        'WindGustDir': 'NW',
        'WindGustSpeed': 30.0,
        'WindDir9am': 'N',
        'WindDir3pm': 'W',
        'WindSpeed9am': 15.0,
        'WindSpeed3pm': 20.0,
        'Humidity9am': 60.0,
        'Humidity3pm': 40.0,
        'Pressure9am': 1015.0,
        'Pressure3pm': 1012.0,
        'Cloud9am': 2.0,
        'Cloud3pm': 4.0,
        'Temp9am': 18.0,
        'Temp3pm': 24.0,
        'RainToday': 'No',
        'Region': 2 
    }

    print(f"Procesando datos para: {dato_ejemplo['Location']} - {dato_ejemplo['Date']}")
    resultado = predecir(dato_ejemplo)
    
    print("RESULTADO:")
    print(resultado)