# 📊 CODIFICACIÓN DE VARIABLES DEL MODELO MLP

Este documento explica el significado de cada valor numérico en las variables del dataset.

---

## 🌤️ **1. Clima**
Condiciones meteorológicas durante la entrega.

| Código | Significado | Descripción |
|--------|-------------|-------------|
| **0** | Despejado | Buen clima, sin precipitaciones |
| **1** | Lluvia | Condiciones lluviosas |
| **2** | Tormenta| Nevadas  |

**Impacto**: Clima adverso (1, 2) aumenta probabilidad de retrasos.

---

## 🚗 **2. TraficoPico**
Nivel de congestión vehicular en la ruta.

| Código | Significado | Descripción |
|--------|-------------|-------------|
| **0** | Sin tráfico | Tráfico fluido |
| **1** | Tráfico moderado | Congestión leve |
| **2** | Tráfico pesado | Alta congestión vehicular |

**Impacto**: Tráfico pesado (2) correlaciona fuertemente con demoras.

---

## ⚠️ **3. RiesgoRuta**
Nivel de peligrosidad o complejidad de la ruta.

| Código | Significado | Descripción |
|--------|-------------|-------------|
| **0** | Riesgo bajo | Carreteras principales, bien mantenidas |
| **1** | Riesgo moderado | Rutas secundarias o con algunas complicaciones |
| **2** | Riesgo alto | Rutas de montaña, caminos difíciles |

**Impacto**: Mayor riesgo implica velocidades menores y mayor probabilidad de incidentes.

---

## 📏 **4. Distancia_km**
Distancia total del recorrido en kilómetros.

| Rango | Clasificación |
|-------|---------------|
| 0-100 km | Corta distancia |
| 100-200 km | Distancia media |
| 200-400 km | Larga distancia |


---

## ⏱️ **5. TiempoEstimado_min**
Tiempo planificado para completar la entrega (en minutos).

**Formato**: Número decimal basado en:
- Distancia
- Condiciones previstas
- Velocidad promedio esperada

**Ejemplo**: 180.0 minutos = 3 horas

---

## ⏰ **6. TiempoReal_min**
Tiempo real que tomó la entrega (en minutos).

**Formato**: Número decimal
**Relación**: `TiempoReal - TiempoEstimado = Demora`

---

## ⌛ **7. Demora_min**
Diferencia entre tiempo real y estimado (en minutos).

| Valor | Significado |
|-------|-------------|
| **0** | Sin demora (entrega puntual) |
| **>0** | Minutos de retraso |
| **<0** | Llegada anticipada (poco común) |

**Ejemplo**: 
- Demora = 5 min → Retraso leve
- Demora = 70 min → Retraso significativo

---

## 📦 **8. TipoCarga**
Clasificación del tipo de mercancía transportada.

| Código | Significado | Descripción |
|--------|-------------|-------------|
| **0** | Media| 
| **1** | Ligera|
| **2** | Pesada | Maquinaria, materiales de construcción |



---

## ⚖️ **9. Peso_kg**
Peso total de la carga en kilogramos.

| Rango | Clasificación |
|-------|---------------|
| 0-3,000 kg | Carga ligera |
| 3,000-8,000 kg | Carga media |
| 8,000-15,000 kg | Carga pesada |
| >15,000 kg | Carga muy pesada |

**Impacto**: Mayor peso afecta velocidad y consumo de combustible.

---

## 👨‍✈️ **10. ExperienciaConductor_anios**
Años de experiencia del conductor.

| Rango | Nivel de experiencia |
|-------|---------------------|
| 0-2 años | baja |
| 3-5 años | media |
| > 6 años | alta |


---

## 🚛 **11. AntiguedadCamion_anios**
Edad del vehículo en años.

| Rango | Estado del vehículo |
|-------|---------------------|
| 0-3 años | Nuevo |
| 4-6 años | media|
| >7 años | vieja |

---

## 🔧 **12. FallasMecanicas**
Ocurrencia de fallas mecánicas durante el trayecto.

| Código | Significado | Descripción |
|--------|-------------|-------------|
| **0** | Sin fallas | Viaje sin problemas mecánicos |
| **1** | Con fallas | Problemas técnicos durante el trayecto |

**Impacto**: Fallas mecánicas (1) causan demoras significativas.

---

## ⛽ **13. NivelCombustible_pct**
Nivel de combustible al inicio (como porcentaje normalizado).

| Código | Nivel de combustible |
|--------|---------------------|
| **0** | adecuado (>95%)  |
| **1** | bajo (<5%) |


**Impacto**: Nivel bajo puede requerir paradas adicionales.

---

## 🕐 **14. HorarioSalida**
Franja horaria de salida del viaje.

| Código | Horario | Rango aproximado |
|--------|---------|------------------|
| **0** | Mañana | 
| **1** | Tarde | 
| **2** | Noche | |

**Impacto**: Horarios pico (1) tienen más tráfico.

---

## 🎯 **Variable Objetivo: EntregaATiempo**

| Código | Significado | Descripción |
|--------|-------------|-------------|
| **0** | NO entrega a tiempo | Demora significativa |
| **1** | SÍ entrega a tiempo | Entrega puntual o con mínima demora |

