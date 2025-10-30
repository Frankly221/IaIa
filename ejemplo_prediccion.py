"""
Script de ejemplo para hacer predicciones con el modelo MLP entrenado
Usa el archivo datos_prueba.csv para demostrar cómo hacer predicciones
"""
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

print("="*70)
print("EJEMPLO DE PREDICCIÓN CON MODELO MLP")
print("="*70)

# 1. Cargar el modelo entrenado
print("\n[1/4] Cargando modelo...")
model = load_model('mlp_model.keras')
print("✓ Modelo cargado exitosamente")

# 2. Cargar el escalador
print("\n[2/4] Cargando escalador...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✓ Escalador cargado exitosamente")

# 3. Cargar los datos de prueba
print("\n[3/4] Cargando datos de prueba...")
df_prueba = pd.read_csv('datos_prueba.csv')
print(f"✓ Se cargaron {len(df_prueba)} registros")

print("\nPrimeras 5 filas de los datos:")
print(df_prueba.head())

# Preparar los datos (no incluir la columna objetivo si existe)
if 'EntregaATiempo' in df_prueba.columns:
    X_prueba = df_prueba.drop(columns=['EntregaATiempo']).values
    y_real = df_prueba['EntregaATiempo'].values
    tiene_etiquetas = True
else:
    X_prueba = df_prueba.values
    tiene_etiquetas = False

# 4. Escalar los datos
print("\n[4/4] Escalando datos y haciendo predicciones...")
X_prueba_scaled = scaler.transform(X_prueba)

# Hacer predicciones (probabilidades)
probabilidades = model.predict(X_prueba_scaled, verbose=0).ravel()

# Umbral óptimo encontrado durante el entrenamiento
UMBRAL_OPTIMO = 0.5000  # Este valor será reemplazado por el script principal

# Convertir a predicciones binarias
predicciones = (probabilidades >= UMBRAL_OPTIMO).astype(int)

# 5. Mostrar resultados
print("\n" + "="*70)
print("RESULTADOS DE LAS PREDICCIONES")
print("="*70)

print(f"\nUmbral de decisión: {UMBRAL_OPTIMO:.4f}")
print("\nPredicciones detalladas:")
print("-" * 70)
print(f"{'#':<5} {'Probabilidad':<15} {'Predicción':<15} {'Interpretación'}")
print("-" * 70)

for i, (prob, pred) in enumerate(zip(probabilidades, predicciones), 1):
    interpretacion = "✓ Entrega a tiempo" if pred == 1 else "✗ NO entrega a tiempo"
    print(f"{i:<5} {prob:.4f} ({prob*100:5.2f}%)  {pred:<15} {interpretacion}")

# Resumen de predicciones
total = len(predicciones)
entregas_a_tiempo = np.sum(predicciones == 1)
entregas_tarde = np.sum(predicciones == 0)

print("\n" + "="*70)
print("RESUMEN")
print("="*70)
print(f"Total de registros:           {total}")
print(f"Predicciones 'A tiempo':      {entregas_a_tiempo} ({entregas_a_tiempo/total*100:.1f}%)")
print(f"Predicciones 'NO a tiempo':   {entregas_tarde} ({entregas_tarde/total*100:.1f}%)")

# Si hay etiquetas reales, calcular precisión
if tiene_etiquetas:
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_real, predicciones)
    print(f"\n✓ Accuracy en datos de prueba: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nReporte de clasificación:")
    print("-" * 70)
    print(classification_report(y_real, predicciones, 
                               target_names=['NO a tiempo', 'A tiempo'],
                               digits=3))

# Guardar resultados en CSV
print("\n" + "="*70)
print("GUARDANDO RESULTADOS")
print("="*70)

df_resultados = df_prueba.copy()
df_resultados['Probabilidad'] = probabilidades
df_resultados['Prediccion'] = predicciones
df_resultados['Interpretacion'] = df_resultados['Prediccion'].map({
    0: 'NO entrega a tiempo',
    1: 'Entrega a tiempo'
})

output_file = 'resultados_prediccion.csv'
df_resultados.to_csv(output_file, index=False)
print(f"✓ Resultados guardados en: {output_file}")

print("\n" + "="*70)
print("¡PROCESO COMPLETADO EXITOSAMENTE!")
print("="*70)
