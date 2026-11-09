# Pipeline completo de preprocesamiento

Chain raw → interim → processed → features → modelos. Sin GEE y sin API keys
para los datos satelitales (STAC público AWS). Las fases correspondientes al
lake están en [DATA_MODEL.md](DATA_MODEL.md) y la guía operativa en
[PIPELINES.md](PIPELINES.md); aquí está el qué-y-porqué de cada paso.

## 1. Ingesta y limpieza por fuente

| Fuente | Proceso clave | Producto |
|---|---|---|
| **EGIF** (164.620 partes 2016-2024) | normalización a XLSX oficial → parquet largo; geocodificación paraje→lat/lon; filtro `in_bbox` (Galicia); dedupe por `numeroparte`; códigos de causa preservados (`idcausa`: 100=rayo, 3xx=negligencia, 4xx=intencionada, 5xx=reproducción, 6xx=otras) | `data/interim/egif/parts_normalized.parquet` |
| **EFFIS WFS** | perímetros ≥30 ha (WFS 1.0.0), intersección con celdas H3 | `labels/v1/burned_cells.parquet` |
| **AEMET OpenData** | obs horaria en tiempo real (crontab); backfill histórico con pacing (~15k req límite) | `data/raw/aemet/` |
| **Open-Meteo Archive** | backfill diario 2017→hoy por estación (57): tmax/tmin/prec/viento/RH — una llamada bulk por bloque de 10 estaciones | `data/raw/openmeteo/daily_2017_today.parquet` |
| **Open-Meteo Historical Forecast** | forecast D-1 por estación (lead ~24h, pasadas HRES archivadas) 2023→hoy | `data/raw/openmeteo/forecast_daily.parquet` → `met_fc.parquet` |
| **ERA5-Land (CDS)** | humedad de suelo 2 capas + VPD, remuestreo al grid de celdas | `era5_features()` |
| **Sentinel-2 (STAC AWS)** | composites de VERANO por año (NDMI/NDVI/NBR @100 m, 8/8 años 2017-2024), mosaico local sin auth | `data/raw/s2/s2_summer_*.tif` |
| **OSM** | red drivable completa + edificios (446k); derivados: `dist_to_road`, `road_class`, `road_density_1km`, `road_fuel_interface` (vía cuyo arcén toca combustible denso) | `res9_static.parquet` |
| **DEM GLO-30 + MDT25 IGN** | elevación, pendiente, orientación | idem |
| **MFE25** | mapa forestal: `fuel_model` (Rothermel `n_MODCOM` 1-13), `forest_frac` | idem |

## 2. Mallado H3

- **res9 (~105 m, 245.172 celdas forestales)** = producción (predicción por celda)
- res7 (5.2 km, 5.710 hexágonos) = agregación regional (Stage 2)
- Tabla maestra `h3_cells/res9_static.parquet`: todas las estáticas por celda.

## 3. Etiquetas y muestreo (res9)

- Positivo: ignición EGIF/EFFIS en la celda-día (`surface_ha ≥ 0.1`)
- Target principal `big10`: ignición con `surface_ha ≥ 10` (1.496 en 2017-24;
  137 en test 2023-24)
- Negativos (seed 42, determinista): 150/día random en temporada (3 invierno)
  + tier NEAR-SPACE (anillo 1-3 km de fuegos del mismo día, 40/día) + NEAR-TIME
  (la celda que arderá 8-21 días antes — "hard negative" temporal, sin look-ahead)
- Filtro de contaminación: descarte de (celda,día) a <1 km de un fuego ±7 días
  (es propagación del mismo evento, no ausencia real)
- Split temporal estricto: train ≤2022 (206.714 filas) · test 2023-24 (60.498).
  El test jamás entra en ninguna decisión de modelado.

## 4. Features (48 + causa, columnas por fila celda-día)

- **Meteorología/FWI estación-día** (18): tmax, RH, viento, precip, ffmc/dmc/dc/isi/bui/fwi,
  ffmc_7d, fwi_7d, fwi_prev, ffmc/dmc/dc anoma (vs su media 7d), spei30, spei90.
  Por celda vía **IDW de las 3 estaciones más cercanas** (pesos 1/d²).
- **ERA5-Land** (4): swc_l1/l2_mean, vpd_max, vpd_mean
- **Satélite prev-summer** (4): ndmi/ndvi/nbr del verano anterior + ndmi_anom
- **Estáticas** (10): forest_frac, fuel_model, pendiente, sin/cos orientación, elevación,
  log dist a vía, log densidad viaria, interfaz vía-combustible, road_class
- **Recencia** (3): días desde el último fuego ≤5 km, dist a él, ardió ≤1 km en 1 año
- **Lags de ignición** (3): nº igniciones en la celda (3d) y radios 2.5/5 km (2d/7d)
- **Estacional/antrópico** (5): doy sin/cos, es_finde, dens_x_weekend, lat/lon
- **Causa** (analítica, no feature de modelo): `cause`, `es_humana` (76% humanas), `es_rayo`

## 5. Modelado y calibración

1. **LGBM** (600 árboles, lr 0.05, 31 hojas, `scale_pos_weight` ~39) sobre las
   filas muestreadas → `P(ignición ≥10 ha | celda, día)` cruda.
2. **Isotónica** (modelo auxiliar ≤2021, ajuste en 2022) → probabilidad calibrada:
   Brier 0.0118→0.0065, reliability casi perfecta en 8/10 bins.
3. **Agregación regional Stage 2** (res7): λ_r = Σ p_cal por región →
   GLM Poisson jerárquico `log μ = α + β·log(1+λ_r) + MEM·γ` (80 eigenvectores de
   Moran sobre la adjacency hexagonal; α elegido por CV 2023 H1→H2) →
   P(≥1 fuego en la región ese día) y λ Galicia = Σμ.
4. **Niveles**: celdas por percentil vs temporada de train; regiones por
   riesgo ×N (vs base rate 0.028% región-día).

## 6. Forecast D-1 (régimen operativo)

- Sustitución de la meteo del día D por el forecast emitido D-1 (Historical
  Forecast API de Open-Meteo; corrección de sesgo NGR-lite por estación ajustada
  con 2023: RMSE del FWI forecast 3.03→2.37)
- Los índices lentos (SPEI, anomalías) permanecen en modo observación
- Coste medido: −1pp ROC-AUC agregado, recall operativo @K dentro del ruido
  → el sistema predice de mañana con datos de hoy (sin leakage)

## 7. Evaluación operativa (full-grid)

82 días de test con ≥1 fuego ≥10 ha: se construyen TODAS las celdas cada día y
se marca el K% del territorio con mayor riesgo (top recall medio diario):
top-1% → 3-5% · top-5% → 13-19% · **top-10% → 29.5-31.2%** de los 137 fuegos.
A nivel región: AUC 0.755, rec@10% regiones-día 39.4% (Stage 2).
