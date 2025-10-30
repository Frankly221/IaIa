# 📋 Formato del CSV de Prueba para el Modelo MLP

## 🔍 Estructura del Archivo

El archivo CSV debe contener **14 columnas** con las siguientes características (en este orden):

### Columnas Requeridas:

| # | Columna | Descripción | Tipo | Valores Posibles |
|---|---------|-------------|------|------------------|
| 1 | **Clima** | Condiciones climáticas | Categórico | 0=Bueno, 1=Malo |
| 2 | **TraficoPico** | Nivel de tráfico | Categórico | 0=Bajo, 1=Medio, 2=Alto |
| 3 | **RiesgoRuta** | Nivel de riesgo de la ruta | Categórico | 0=Bajo, 1=Medio, 2=Alto, 3=Muy Alto |
| 4 | **Distancia_km** | Distancia en kilómetros | Numérico | Decimal > 0 |
| 5 | **TiempoEstimado_min** | Tiempo estimado en minutos | Numérico | Decimal > 0 |
| 6 | **TiempoReal_min** | Tiempo real de entrega | Numérico | Decimal > 0 |
| 7 | **Demora_min** | Minutos de demora | Numérico | Decimal ≥ 0 |
| 8 | **TipoCarga** | Tipo de carga transportada | Categórico | 0=Tipo A, 1=Tipo B, 2=Tipo C |
| 9 | **Peso_kg** | Peso de la carga en kg | Numérico | Entero > 0 |
| 10 | **ExperienciaConductor_anios** | Años de experiencia | Numérico | Entero ≥ 0 |
| 11 | **AntiguedadCamion_anios** | Antigüedad del camión | Numérico | Entero ≥ 0 |
| 12 | **FallasMecanicas** | Presencia de fallas | Binario | 0=No, 1=Sí |
| 13 | **NivelCombustible_pct** | Nivel de combustible | Categórico | 0=Bajo, 1=Medio, 2=Alto |
| 14 | **HorarioSalida** | Horario de salida | Categórico | 0=Madrugada, 1=Mañana, 2=Tarde |

### Columna Opcional (para validación):

| # | Columna | Descripción | Tipo | Valores Posibles |
|---|---------|-------------|------|------------------|
| 15 | **EntregaATiempo** | Entrega a tiempo (etiqueta real) | Binario | 0=No, 1=Sí |

> **Nota**: Si incluyes la columna `EntregaATiempo`, el script de predicción calculará la precisión automáticamente.

---

## 📄 Ejemplo de CSV (datos_prueba.csv)

```csv
Clima,TraficoPico,RiesgoRuta,Distancia_km,TiempoEstimado_min,TiempoReal_min,Demora_min,TipoCarga,Peso_kg,ExperienciaConductor_anios,AntiguedadCamion_anios,FallasMecanicas,NivelCombustible_pct,HorarioSalida
0,0,1,150.5,180.0,185.0,5.0,1,5000,8,3,0,1,1
1,2,2,300.0,350.0,420.0,70.0,2,12000,3,8,1,0,0
0,1,1,100.0,120.0,125.0,5.0,0,3000,10,2,0,1,2
```

---

## 🚀 Cómo Usar el Modelo

### Opción 1: Usar el script de ejemplo

```bash
python ejemplo_prediccion.py
```

Este script:
- ✅ Carga el modelo automáticamente
- ✅ Lee `datos_prueba.csv`
- ✅ Hace predicciones
- ✅ Muestra resultados detallados
- ✅ Guarda resultados en `resultados_prediccion.csv`

### Opción 2: Código personalizado

```python
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Cargar modelo y escalador
model = load_model('mlp_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Cargar datos
df = pd.read_csv('tus_datos.csv')
X = df.values  # O df.drop(columns=['EntregaATiempo']).values si tiene etiquetas

# Escalar y predecir
X_scaled = scaler.transform(X)
probabilidades = model.predict(X_scaled).ravel()
predicciones = (probabilidades >= 0.5).astype(int)

# Ver resultados
print("Predicciones:", predicciones)
print("Probabilidades:", probabilidades)
```

---

## 📊 Interpretación de Resultados

### Predicción:
- **0** = NO entrega a tiempo (se predice demora)
- **1** = Entrega a tiempo (se predice puntualidad)

### Probabilidad:
- Valor entre **0.0** y **1.0**
- Más cercano a **1.0** = Mayor confianza en entrega a tiempo
- Más cercano a **0.0** = Mayor confianza en demora

### Ejemplo de salida:

```
#     Probabilidad     Predicción      Interpretación
1     0.8524 (85.24%)  1               ✓ Entrega a tiempo
2     0.2341 (23.41%)  0               ✗ NO entrega a tiempo
3     0.6789 (67.89%)  1               ✓ Entrega a tiempo
```

---

## 💡 Consejos

1. **Formato correcto**: Asegúrate de que el CSV tenga exactamente 14 columnas en el orden especificado
2. **Sin encabezado faltante**: La primera fila debe ser el encabezado con los nombres de las columnas
3. **Valores válidos**: Verifica que los valores categóricos estén dentro del rango especificado
4. **Sin valores nulos**: El modelo no acepta valores NaN o vacíos
5. **Separador**: Usa coma (`,`) como separador

---

## 📁 Archivos Generados

Después de ejecutar `ejemplo_prediccion.py`:

- **resultados_prediccion.csv** - Datos originales + Probabilidad + Predicción + Interpretación

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo usar datos sin la columna EntregaATiempo?**  
R: Sí, es opcional. Si no la incluyes, simplemente no se calculará la precisión.

**P: ¿Qué umbral usa el modelo?**  
R: El umbral óptimo es 0.5 por defecto, pero fue optimizado durante el entrenamiento.

**P: ¿Cuántos registros puedo predecir a la vez?**  
R: El modelo puede procesar cualquier cantidad de registros simultáneamente.

**P: ¿Los valores deben estar normalizados?**  
R: No, el escalador se encarga automáticamente de normalizar los datos.

---

## 📞 Soporte

Si tienes problemas:
1. Verifica que el CSV tenga el formato correcto
2. Asegúrate de que `mlp_model.keras` y `scaler.pkl` estén en el mismo directorio
3. Revisa que todas las columnas estén presentes y en el orden correcto
