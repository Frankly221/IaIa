"""
Script para cargar y usar el modelo MLP entrenado con variables categóricas
"""
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

print("="*70)
print("SISTEMA DE PREDICCIÓN - ENTREGA A TIEMPO")
print("="*70)

# ==================== CONFIGURACIÓN DE RUTAS ====================
# Obtener el directorio donde está este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"\n📁 Directorio de trabajo: {SCRIPT_DIR}")

# Rutas a los archivos
MODEL_PATH = os.path.join(SCRIPT_DIR, 'mlp_model.keras')
SCALER_PATH = os.path.join(SCRIPT_DIR, 'scaler.pkl')
INPUT_CSV = os.path.join(SCRIPT_DIR, 'new.csv')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'predicciones_resultado.csv')

# ==================== CARGAR MODELO Y ESCALADOR ====================
print("\n[1/4] Cargando modelo y escalador...")

try:
    model = load_model(MODEL_PATH)
    print(f"✓ Modelo cargado exitosamente desde: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el modelo en {MODEL_PATH}")
    exit(1)

try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Escalador cargado exitosamente desde: {SCALER_PATH}")
except FileNotFoundError:
    print(f"❌ Error: No se encontró el escalador en {SCALER_PATH}")
    exit(1)

# Umbral óptimo encontrado durante el entrenamiento
UMBRAL_OPTIMO = 0.5  # Ajusta este valor según tu entrenamiento

# ==================== FUNCIÓN DE PREPROCESAMIENTO ====================
def preprocesar_datos(df_new):
    """
    Aplica la misma transformación que durante el entrenamiento
    """
    print("\n[2/4] Preprocesando datos nuevos...")
    
    # Mostrar datos originales
    print("\n--- Datos originales ---")
    print(df_new)
    
    # Variables nominales (One-Hot Encoding)
    nominales = ['Clima', 'TipoCarga', 'HorarioSalida']
    
    # Variables ordinales (Label Encoding)
    ordinales = ['TraficoPico', 'RiesgoRuta']
    
    # Variables binarias (Label Encoding)
    binarias = ['FallasMecanicas']
    
    # Crear copia para no modificar el original
    df_processed = df_new.copy()
    
    # 1. Aplicar Label Encoding a variables ordinales PRIMERO
    print("\n1. Label Encoding para variables ordinales:")
    for col in ordinales:
        if col in df_processed.columns:
            orden_map = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
            valores_unicos = df_processed[col].unique()
            print(f"   {col}: {valores_unicos} → Bajo=0, Medio=1, Alto=2")
            df_processed[col] = df_processed[col].map(orden_map)
            
            # Verificar valores nulos
            if df_processed[col].isnull().any():
                raise ValueError(f"❌ Valores no reconocidos en {col}. Use: Bajo, Medio, Alto")
    
    # 2. Aplicar Label Encoding a variables binarias
    print("\n2. Label Encoding para variables binarias:")
    for col in binarias:
        if col in df_processed.columns:
            binary_map = {'No': 0, 'Si': 1}
            valores_unicos = df_processed[col].unique()
            print(f"   {col}: {valores_unicos} → No=0, Si=1")
            df_processed[col] = df_processed[col].map(binary_map)
            
            # Verificar valores nulos
            if df_processed[col].isnull().any():
                raise ValueError(f"❌ Valores no reconocidos en {col}. Use: No, Si")
    
    # 3. Aplicar One-Hot Encoding a variables nominales AL FINAL
    print("\n3. One-Hot Encoding para variables nominales:")
    for col in nominales:
        if col in df_processed.columns:
            unique_vals = df_processed[col].unique()
            print(f"   {col}: {unique_vals}")
    
    df_encoded = pd.get_dummies(df_processed, columns=nominales, prefix=nominales, drop_first=False)
    
    print(f"\n✓ Datos procesados: {df_encoded.shape[1]} features")
    print("\n--- Columnas después del procesamiento ---")
    for i, col in enumerate(df_encoded.columns):
        print(f"   {i+1}. {col}")
    
    return df_encoded.values

# ==================== CARGAR Y PROCESAR NUEVOS DATOS ====================
print("\n" + "="*70)
print("PREDICCIÓN CON NUEVOS DATOS")
print("="*70)

try:
    # Cargar datos nuevos desde CSV
    df_new = pd.read_csv(INPUT_CSV)
    print(f"\n✓ Datos nuevos cargados desde: {INPUT_CSV}")
    print(f"✓ Registros encontrados: {df_new.shape[0]}")
    print(f"✓ Columnas encontradas: {df_new.shape[1]}")
    
    # Mostrar columnas disponibles
    print(f"\n--- Columnas en el archivo CSV ---")
    for i, col in enumerate(df_new.columns, 1):
        print(f"   {i}. {col}")
    
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo {INPUT_CSV}")
    print(f"\n💡 Asegúrate de que el archivo 'new.csv' esté en: {SCRIPT_DIR}")
    exit(1)
except Exception as e:
    print(f"❌ Error al cargar el archivo CSV: {e}")
    exit(1)

try:
    # Preprocesar datos
    X_new = preprocesar_datos(df_new)
except ValueError as e:
    print(f"\n❌ Error en el preprocesamiento: {e}")
    exit(1)
except Exception as e:
    print(f"\n❌ Error inesperado en el preprocesamiento: {e}")
    exit(1)

# ==================== ESCALAR DATOS ====================
print("\n[3/4] Escalando datos...")
try:
    X_new_scaled = scaler.transform(X_new)
    print(f"✓ Datos escalados correctamente")
    print(f"   Media: {X_new_scaled.mean():.6f}")
    print(f"   Desviación estándar: {X_new_scaled.std():.6f}")
except Exception as e:
    print(f"❌ Error al escalar datos: {e}")
    exit(1)

# ==================== REALIZAR PREDICCIONES ====================
print("\n[4/4] Realizando predicciones...")
try:
    probabilidades = model.predict(X_new_scaled, verbose=0).ravel()
    predicciones = (probabilidades >= UMBRAL_OPTIMO).astype(int)
    print("✓ Predicciones completadas exitosamente")
except Exception as e:
    print(f"❌ Error al realizar predicciones: {e}")
    exit(1)

# ==================== MOSTRAR RESULTADOS ====================
print("\n" + "="*70)
print("RESULTADOS DE LAS PREDICCIONES")
print("="*70)

# Crear DataFrame con resultados
resultados = df_new.copy()
resultados['Probabilidad_EntregaATiempo'] = probabilidades
resultados['Prediccion'] = predicciones
resultados['Prediccion_Texto'] = resultados['Prediccion'].map({0: 'NO a tiempo', 1: 'SÍ a tiempo'})

print("\n--- Resumen de Predicciones ---")
print(resultados[['Clima', 'TraficoPico', 'RiesgoRuta', 'Distancia_km', 
                  'Probabilidad_EntregaATiempo', 'Prediccion_Texto']])

# Estadísticas
print("\n--- Estadísticas ---")
print(f"Total de entregas: {len(predicciones)}")
print(f"Entregas A TIEMPO (predichas): {np.sum(predicciones == 1)} ({np.sum(predicciones == 1)/len(predicciones)*100:.1f}%)")
print(f"Entregas NO A TIEMPO (predichas): {np.sum(predicciones == 0)} ({np.sum(predicciones == 0)/len(predicciones)*100:.1f}%)")
print(f"Probabilidad promedio: {probabilidades.mean():.4f}")
print(f"Umbral de decisión: {UMBRAL_OPTIMO:.4f}")

# Detalles por cada predicción
print("\n--- Detalles por Predicción ---")
for i, (idx, row) in enumerate(df_new.iterrows()):
    prob = probabilidades[i]
    pred = predicciones[i]
    pred_texto = "SÍ a tiempo ✓" if pred == 1 else "NO a tiempo ✗"
    
    print(f"\n{'-'*70}")
    print(f"ENTREGA #{i+1}")
    print(f"{'-'*70}")
    print(f"📍 Distancia: {row['Distancia_km']} km")
    print(f"⏱️  Tiempo estimado: {row['TiempoEstimado_min']} min")
    print(f"🌤️  Clima: {row['Clima']}")
    print(f"🚦 Tráfico: {row['TraficoPico']}")
    print(f"⚠️  Riesgo Ruta: {row['RiesgoRuta']}")
    print(f"📦 Tipo Carga: {row['TipoCarga']}")
    print(f"⚙️  Fallas Mecánicas: {row['FallasMecanicas']}")
    print(f"\n🎯 PREDICCIÓN: {pred_texto}")
    print(f"📊 Probabilidad: {prob:.2%}")
    print(f"   └─ Confianza: {'ALTA' if abs(prob - 0.5) > 0.3 else 'MEDIA' if abs(prob - 0.5) > 0.15 else 'BAJA'}")

# Guardar resultados
try:
    resultados.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Resultados guardados en: {OUTPUT_CSV}")
    print("="*70)
except Exception as e:
    print(f"❌ Error al guardar resultados: {e}")

# ==================== INTERPRETACIÓN DE FACTORES ====================
print("\n" + "="*70)
print("ANÁLISIS DE FACTORES DE RIESGO")
print("="*70)

for i, (idx, row) in enumerate(df_new.iterrows()):
    print(f"\n--- Entrega #{i+1} ---")
    factores_riesgo = []
    
    # Analizar factores de riesgo
    if row['Clima'] == 'Tormenta':
        factores_riesgo.append("⛈️  Clima adverso (Tormenta)")
    elif row['Clima'] == 'Lluvia':
        factores_riesgo.append("🌧️  Clima moderado (Lluvia)")
    
    if row['TraficoPico'] == 'Alto':
        factores_riesgo.append("🚗 Tráfico intenso")
    elif row['TraficoPico'] == 'Medio':
        factores_riesgo.append("🚙 Tráfico moderado")
    
    if row['RiesgoRuta'] == 'Alto':
        factores_riesgo.append("⚠️  Ruta de alto riesgo")
    elif row['RiesgoRuta'] == 'Medio':
        factores_riesgo.append("⚡ Ruta de riesgo moderado")
    
    if row['FallasMecanicas'] == 'Si':
        factores_riesgo.append("🔧 Fallas mecánicas reportadas")
    
    if row['Distancia_km'] > 300:
        factores_riesgo.append(f"📏 Distancia larga ({row['Distancia_km']} km)")
    
    if row['TipoCarga'] == 'Peligrosa':
        factores_riesgo.append("☢️  Carga peligrosa")
    elif row['TipoCarga'] == 'Fragil':
        factores_riesgo.append("📦 Carga frágil")
    
    if factores_riesgo:
        print("Factores de riesgo identificados:")
        for factor in factores_riesgo:
            print(f"  • {factor}")
    else:
        print("✓ Sin factores de riesgo significativos")
    
    print(f"Nivel de riesgo: {'ALTO' if len(factores_riesgo) >= 4 else 'MEDIO' if len(factores_riesgo) >= 2 else 'BAJO'}")

print("\n" + "="*70)
print("✅ PROCESO COMPLETADO")
print("="*70)