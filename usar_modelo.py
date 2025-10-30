"""
Script para cargar y usar el modelo MLP entrenado
"""
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
print("Cargando modelo...")
model = load_model('mlp_model.keras')
print("✓ Modelo cargado exitosamente")

# Cargar el escalador
print("Cargando escalador...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Escalador cargado exitosamente")

# Umbral óptimo encontrado durante el entrenamiento
UMBRAL_OPTIMO = 0.1800

print("\n" + "="*60)
print("MODELO LISTO PARA HACER PREDICCIONES")
print("="*60)

# EJEMPLO DE USO:
# ---------------
# Cargar nuevos datos
# df_new = pd.read_csv("nuevos_datos.csv")
# X_new = df_new.drop(columns=["EntregaATiempo"]).values  # Si tiene la columna
# O simplemente: X_new = df_new.values

# Escalar los datos
# X_new_scaled = scaler.transform(X_new)

# Hacer predicciones (probabilidades)
# probabilidades = model.predict(X_new_scaled).ravel()

# Convertir a predicciones binarias usando el umbral óptimo
# predicciones = (probabilidades >= UMBRAL_OPTIMO).astype(int)

# Mostrar resultados
# print(f"Predicciones: {predicciones}")
# print(f"Probabilidades: {probabilidades}")

print("\nEjemplo de uso:")
print("-" * 60)
print("df_new = pd.read_csv('nuevos_datos.csv')")
print("X_new = df_new.values")
print("X_new_scaled = scaler.transform(X_new)")
print("probabilidades = model.predict(X_new_scaled).ravel()")
print(f"predicciones = (probabilidades >= {UMBRAL_OPTIMO:.4f}).astype(int)")
print("-" * 60)
