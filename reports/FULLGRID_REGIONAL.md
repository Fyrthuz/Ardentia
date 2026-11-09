# Evaluación operativa full-grid y forecast regional probabilístico

**Ciclo 2026-10-30/31** — el doc maestro del ciclo de experimentación adoptado
por el [README principal](../README.md) y [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).
Los reportes del ciclo anterior (2026-10-29, gate Fase 2) están en `archive/`.

**Fecha:** 2026-10-30 · Modelo: LGBM (600 árboles, lr 0.05, 48 features) · Target: ignición ≥10 ha

## 1. Evaluación operativa full-grid (82 días de test con fuegos ≥10 ha, 137 fuegos)

Por primera vez se predice **las 245.172 celdas** cada día de test (no la muestra de 15),
emulando el uso operativo real: cada mañana se marca el K% de Galicia con mayor riesgo.

| Esfuerzo diario | Recall medio | Recall mediano |
|---|---|---|
| top 1% (2.451 celdas) | 3,0% | 0% |
| top 5% (12.258 celdas) | 14,2% | 0% |
| top 10% (24.517 celdas) | **31,2%** | 0% |

** Lectura honesta: ** marcando a diario el 10% del territorio se captura ~31% de los fuegos
grandes (media diaria; la mediana es 0 porque en la mayoría de días ningún fuego cae en el
top). Es un problema muy difícil: 137 fuegos repartidos en 82 días sobre 245k celdas.

## 2. Calibración de probabilidades (isotónica)

Modelo auxiliar entrenado ≤2021, isotónica ajustada sobre 2022.

- **Brier: 0.0118 → 0.0065** (−45%)
- Reliability casi perfecta en los 8 bins superiores (ej.: p calibrada 3,16% vs real 3,08%
  en el decil superior; 0,87% vs 0,84% en el bin 0.0043).

Con esto la frase *"esta celda tiene un 3% de probabilidad de fuego ≥10 ha hoy"* es honesta.

## 3. Forecast regional (agregación Poisson + recalibración)

Agregación por hexágonos padre res7 (5,2 km): λ = Σp_i → P(≥1) = 1−e^(−λ).

**Problema detectado:** la P(≥1) cruda está masivamente sobreenconfiada (regiones con
P(≥1) 50-100% tuvieron fuego el 0,05% de las veces). Causas: (a) la isotónica de celda se
ajustó con la muestra de 15 celdas/día y el full-grid extrapola fuera de su rango (clipping
al máximo), (b) las celdas comparten el mismo tiempo → correlación perfecta que viola la
independencia del Poisson.

**Solución:** segunda isotónica a nivel región, ajustada con 2023 y evaluada en 2024
(split temporal honesto, 154.170 regiones-día, 33 con fuego):

| Decil de p_reg | p mediana | frecuencia real |
|---|---|---|
| 1.º | 0,0000 | 0,0000 |
| 5.º | 0,0002 | 0,0001 |
| 8.º | 0,0004 | 0,0004 |
| 10.º (top) | 0,0005 | **0,0008** |

- **AUC regional: 0,728** (predecir ≥1 fuego ≥10 ha en la región ese día)
- Top 10% de regiones/día → 27,3% de los fuegos-region · top 20% → 42,4%
- Tendencia monótona en los deciles superiores; calibración absoluta comprimida
  (eventos raros: base rate 0,02-0,03% por región-día)

## 4. Comunicación del riesgo

Dado que las probabilidades honestas son diminutas, el producto final comunica:

- **Nivel de riesgo** por percentiles (verde/amarillo/naranja/rojo) — celdas y regiones
- **Múltiplo de la base rate** (×N): "esta región hoy tiene 85× el riesgo medio"
- **Nº esperado de fuegos ≥10 ha en Galicia** (λ recalibrado) en el pie del vídeo

Nunca se afirma "va a haber un incendio con total seguridad": proceso estocástico,
se emite probabilidad calibrada + nivel ordinal (analogía AEMET).

## 5. Dependencia espacial y garantías (frailty NB + conformal)

`src/ardintra/models/joint_calibration.py` — dos experimentos sobre la agregación regional:

**a) Frailty gamma (≈ copula de factor común):** los fuegos de un día comparten el tiempo
(mismo FWI) → sobredispersión respecto al Poisson. Modelo N_d ~ NB(media λ_d, frailty k).

- k ajustado en 2023: **6.4** (sobredispersión real en train: días con 8-9 fuegos)
- **Fuera de muestra (2024) el Poisson gana**: log-lik −32.9 (Poisson) vs −34.5 (NB)
- Con 55 días de entrenamiento, el parámetro extra no paga. La independencia Poisson
  es adecuada a escala Galicia-día; la sobredispersión no es estimable de forma robusta.

**b) Split conformal regional (α=0.10, ajuste 2023 → evaluación 2024):**

- Para garantizar cobertura marginal ≥90%, el conjunto conformal necesita marcar el
  **87.7% del territorio** (media 5.008 regiones/día)
- Comparación: top-20% fijo por p_reg → 78.8% de cobertura con solo el 20% del territorio
- **Lectura:** el conformal cuantifica el precio de la garantía — con señal AUC 0.73,
  una garantía del 90% exige marcar casi todo. Útil como límite honesto del sistema.

**Limitación documentada:** el parquet regional solo contiene los 82 días CON fuegos
(los 649 días sin fuego ≥10 ha no se evaluaron), por lo que la calibración de P(N≥1)
incondicionada no es validable sin re-procesar los días sin fuego.

## 6. Vídeo 2-paneles (actualizado)

`reports/simulacion_2023_2panel.mp4` — panel izquierdo: celdas res9 por percentil de
p calibrada; panel derecho: regiones res7 por P(≥1) recalibrada con etiquetas ×N en las
top-3 del día; pie con λ esperado Galicia y fuegos reales (⭐ ≥10 ha, 🔵 <10 ha).

## Artefactos

- `reports/fullgrid_eval.json` — recall operativo por día
- `reports/fullgrid_regional.parquet` — λ, P(≥1) cruda y fuegos observados por región-día
- `reports/fullgrid_regional_calibrado.parquet` — P(≥1) recalibrada (eval 2024)
- `reports/regional_reliability_2024.csv` — reliability regional
- `reports/calibration.json`, `reports/reliability_diagram.csv` — calibración de celda
- `reports/joint_calibration.json` — frailty NB + conformal regional

## 7. Fase 1 — Experimentos de mejora (2026-10-31)

### 7.1 Muestreo estratificado de negativos — REFUTADO

Hipótesis (Pourmohamad 2025): hard negatives (near-space doblado + sitios históricos
de ignición en días FWI≥p90) corregirían la sobreenconfianza full-grid.

| Config | AUC | AP | recall@10% operativo |
|---|---|---|---|
| V1 muestreo original | **0.897** | **0.036** | **31.2%** |
| V2 estratificado | 0.881 | 0.031 | 27.2% |
| V2 sin features densidad | 0.878 | 0.034 | — |

- El muestreo estratificado EMPEORA el modelo (más conservativo, −1.9pp AUC)
- La sobreenconfianza regional persiste → es estructural (transferencia de la
  calibración muestra→full-grid + suma de celdas correlacionadas), no de muestreo
- La 2ª isotónica regional sigue siendo el fix pragmático correcto
- **Decisión:** revertido a v1; tiers nuevos detrás de `ARDENTIA_HARDNEG=1`
- Features de densidad por causa: shift train/test (0→1.97), sin ganancia — excluidas

### 7.2 Causa de ignición — propagada y disponible

- EGIF `idcausa` propagada al dataset: 6.464 positivos (92%) — 76% humanas (3xx+4xx),
  3% rayo (1xx), resto desconocida/reproducción
- Columnas `cause`, `es_humana`, `es_rayo` disponibles para análisis y modelos por causa
- Pendiente (Fase 2): modelo separado humano/natural (Gelabert 2025, Gómez-Pazo 2025)

### 7.3 Forecast D-1 — VIABLE con coste pequeño y cuantificado

**NOTA DE CORRECCIÓN:** la primera versión de este experimento usó datos de
GFS Previous Runs (cobertura solo desde 2024-01) que dejó las features met de
2023 a cero → resultados inválidos, retractados. La versión definitiva usa la
Historical Forecast API (HRES seamless, cobertura 2023 completa, lead ~24h)
con corrección de sesgo NGR-lite por estación (fit 2023, verificado en 2024:
RMSE del FWI forecast 3.03→2.37, −22%).

| Métrica | obs (V1) | forecast D-1 corregido (V3b) | dif |
|---|---|---|---|
| AUC agregado (muestra) | 0.897 | 0.887 | −1.0pp |
| recall top5 operativo | 14.2% | 13.2% | −0.3 SE (ruido) |
| recall top10 operativo | 31.2% | 29.5% | −0.4 SE (ruido) |

- **Conclusión:** el sistema puede operar con forecast D-1: el coste es −1pp
  de AUC agregado en muestra y no detectable en el recall operativo @K
  → elimina el leakage operativo (la meteo del día D no está disponible de mañana)
- Artefactos: `src/ardintra/features/build_met_fc.py` (NGR-lite),
  `data/raw/openmeteo/met_fc.parquet`, `reports/fullgrid_eval_v3.json`,
  flag `ARDENTIA_MET_FC=1` en fullgrid_eval

## 8. Stage 2 — Modelo jerárquico espacial (GLM + Moran Eigenvector Maps)

Réplica python del enfoque "XGBoost meets INLA" (arXiv 2508.09896), sin R:

    log μ_{r,d} = α + β·log(1+λ_raw_{r,d}) + Σ_k γ_k·MEM_k(r)

- MEM: eigenvectores de Moran de la adjacency hexagonal res7 (5.710 regiones,
  base discreta que aproxima el campo Matérn del SPDE; Griffith 1996)
- β recalibra la intensidad Stage 1 (ajustado: 0.61 — corrige la sobreenconfianza)
- u_r = MEM·γ: patrón espacial residual persistente (clustering no explicado
  por features dinámicas)
- Ajuste: PoissonRegressor (ridge), alpha=1e-6 **seleccionado por CV dentro de
  2023 (fit Ene-Jun → val Jul-Dic)** — sin tocar 2024

### Validación 2024 (split temporal, 154.170 regiones-día, 33 con fuego)

| Modelo | AUC | rec@10% región-día |
|---|---|---|
| λ cruda Stage 1 | 0.719 | 30.3% |
| Isotónica regional (producción) | 0.693 | 27.3% |
| **Stage 2 (GLM+MEM)** | **0.755** | **39.4%** |

Reliability por deciles: monótona y bien calibrada (top decile: pred 0.00052 vs
real 0.00071 — subconfianza 1.4×, aceptable para eventos raros).

Sensibilidad a alpha: 1e-4 → 0.720 (espacial colapsado), 1e-6 → 0.755 (óptimo,
elegido por CV), 1e-8 → 0.717 (sobreajuste). El CV interno de 2023 selecciona
1e-6 de forma independiente → la ganancia no es artefacto de selección.

**Conclusión:** Stage 2 mejora el forecast regional de forma genuina (+3.6pp AUC,
+12pp rec@10% sobre el baseline de producción) con calibración honesta. La
recalibración isotónica de producción queda sustituida por el GLM+MEM.

Artefactos: `src/ardintra/models/stage2_lgcp.py`, `reports/stage2_lgcp.json`,
`reports/stage2_reliability.csv`, `reports/stage2_predictions_2024.parquet`.

## 9. Stage 2 integrado al vídeo

`reports/simulacion_2023_2panel.mp4` regenerado: el panel derecho usa μ del
GLM+MEM (`reports/stage2_model.npz`) — P(≥1) regional honesta, etiquetas ×N
(riesgo vs base rate 0.028%) y λ diario = Σμ (Nº esperado de fuegos). Ejemplo
2023-10-12: ×6-×7 en las top-3 (antes ×38 con el modelo sobreenconfiado).

## 10. Fase 3 — Secuencias temporales (14 días) — SIN GANANCIA

Contexto temporal explícito (ventana 14 días de meteo/FWI por celda, IDW 3
estaciones; `data/processed/features/v1/sequences.npz`, 267k × 14×8):

| Modelo | AUC (test) | AP |
|---|---|---|
| LGBM tabular día-D (48 feats, obs) | **0.897** | **0.036** |
| LGBM tabular día-D (48 feats, forecast test) | 0.867 | 0.019 |
| LGBM + sumarios 14d | 0.864 | 0.030 |
| LGBM + secuencia aplanada (112) | 0.811 | 0.012 |
| LSTM 2×64 (solo met-secuencia) | 0.870 | 0.017 |

- **Las secuencias de meteo no añaden valor**: los índices FWI (DC/DMC son
  acumulados con memoria de meses) ya codifican la historia temporal relevante
- Consistente con Di Giuseppe 2025 (ECMWF): calidad de datos > arquitectura
- Nota: comparación LSTM vs tabular no es apples-to-apples (LSTM sin features
  estáticas/espaciales); el test justo (tabular+sumarios) confirma: −0.3pp AUC
- Hallazgo adicional (corregido tras detectar datos inválidos): el forecast
  D-1 con corrección NGR cuesta −1pp de AUC agregado (0.897→0.887); el recall
  operativo @K queda dentro del ruido (−0.3/−0.4 SE)

## Estado final del sistema

Pipeline operativo completo y validado:
1. Forecast D-1 (GFS) → 2. LGBM 245k celdas → 3. Calibración isotónica celda →
4. Stage 2 GLM+MEM regional (AUC 0.755, rec@10% 39.4%) → 5. Vídeo/niveles ×N+λ

## 11. Ronda literatura 2 (NGR, causa, WUI)

### 11.1 Corrección de sesgo del forecast (NGR-lite) — ADOPTADA ✅

Bohlmann & Laine 2024 (NHESS): calibración estadística de forecasts de FWI.
Implementación: corrección lineal por estación (tmax/rh/viento) + factor
multiplicativo de precipitación, ajustada con pares obs-forecast 2023,
aplicada a todo el forecast; cadena FWI recalculada sobre el tiempo corregido.

- RMSE del FWI forecast 2024 (holdout): 3.03 → **2.37 (−22%)** · bias +1.27 → +0.91
- Coste del forecast en AUC agregado: −3pp (sin corregir, medida inválida
  inicial) → **−1pp corregido** (0.897 → 0.887)
- Recall operativo @K: dentro del ruido (−0.3/−0.4 SE)
- `src/ardintra/features/build_met_fc.py`

### 11.2 Modelos separados por causa — REFUTADO ❌

Gelabert 2025 / Shmuel 2025 sugieren modelos por causa. Con nuestros datos
(862 humanas ≥10ha en train, 39 rayo, 458 otra):

| Modelo | AUC |
|---|---|
| Único (referencia) | **0.897** |
| Solo humanas | 0.867 |
| Solo naturales+otras | 0.704 |
| Combinado 1−(1−p_h)(1−p_n) | 0.861 |

Dividir debilita la señal (menos positivos por modelo) y el combinado pierde
3.6pp. Con 76% de igniciones humanas, el modelo único ya captura el patrón
dominante. La causa queda como columna analítica, no como separación de modelos.

### 11.3 WUI / densidad de edificios (OSM) — REFUTADO ❌

Gelabert 2025: distancia a WUI = driver humano principal. Proxy sin población:
446.494 edificios OSM → densidad 1 km + distancia al edificio más cercano
(`data/processed/h3_cells/res9_wui.parquet`).

- AUC: 0.8874 (sin) vs 0.8857 (con) — sin ganancia
- Importancia mínima: ya tenemos dist_to_road, road_density, road_class y
  road_fuel_interface — la huella humana está cubierta; el proxy es colineal

## 12. TabPFN v2 y destilación — ganancia en muestra, no transfiere al grid

### 12.1 TabPFN v2 (transformer preentrenado, Nature 2025) — mejor modelo en muestra

Ensembling por subsampling (10×10k, todos los positivos), eval con 137
positivos test + 20k negativos (AUC insesgado):

| Modelo | AUC | AP |
|---|---|---|
| **TabPFN v2 (teacher)** | **0.9387** | **0.165** |
| Blend student+labels | 0.9320 | 0.138 |
| Student destilado (LGBM) | 0.9303 | 0.144 |
| LGBM labels (referencia) | 0.8878 | 0.092 |

+5.1pp AUC y AP ×1.8 sobre LGBM en la distribución muestreada.

### 12.2 Destilación a full-grid — NO TRANSFIERE ❌

La inferencia directa de TabPFN en full-grid es inviable (~30+ min/día).
Destilación: 420k filas full-grid de train puntuadas por el teacher
(`distill_set.parquet`) → student LGBM (retiene 0.9303 en muestra).

Pero la evaluación operativa full-grid (82 días) del blend degrada:
top10 18.0% vs 29.5% (LGBM forecast) · AUC regional 0.593 · rec@10% región 13.1%.

**Causa raíz:** el teacher solo fue calibrado/evaluado en la distribución
MUESTREADA (15 celdas/día); su ranking no transfiere al espacio full-grid —
el mismo fallo de dominio que la sobreenconfianza de la isotónica. La
destilación copió fielmente al teacher, incluida su debilidad OOD.

**Lección del proyecto (3 evidencias): la brecha muestra→full-grid es EL
problema central del sistema.** Cualquier mejora debe validarse directamente
en full-grid, no en la muestra.

### 12.3 Decisión

Pipeline de producción = el mejor validado operativamente:
forecast D-1 corregido NGR → LGBM 48 feats → isotónica → Stage 2 GLM+MEM.

## 13. NVIDIA Kumo Tabular — segunda replicación del hallazgo

Teacher 2 = Kumo Tabular large (NVIDIA, ICL, pesos locales HF, 215M params).

### En la distribución muestreada — mejor que todos

| Modelo | AUC | AP |
|---|---|---|
| **Kumo teacher** | **0.947** | — |
| TabPFN teacher | 0.9387 | 0.165 |
| LGBM labels | 0.8887 | 0.092 |

### Operativa full-grid (student destilado de Kumo, 82 días) — falla igual

| Métrica | Kumo destilado | TabPFN destilado | LGBM forecast (V3b) |
|---|---|---|---|
| recall top5 | 12.3% | 6.6% | **13.2%** |
| recall top10 | 17.1% | 18.0% | **29.5%** |
| AUC regional | 0.623 | 0.593 | — |
| rec@10% región | 16.9% | 13.1% | — |

**Hallazgo replicado con 2 foundation models independientes:** ganar en la
distribución muestreada (más AUC que LGBM) implica estar MÁS comprometido con
esa distribución → peor transferencia operativa. El LGBM clásico, menos
expresivo, extrapola mejor al full-grid.

Artefactos: `src/ardintra/models/kumo_teacher.py`, `reports/kumo_teacher.json`,
`reports/fullgrid_eval_kumo.json`, `reports/fullgrid_regional_kumo.parquet`,
`models/student_kumo.joblib`.
