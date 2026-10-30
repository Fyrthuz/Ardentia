# Roadmap — plan por fases y estado de implementación

> Última actualización: 2026-10-29. Estado global: **Fase 0 ✔ · Fase 1 ~85% · Fases 2–5 pendientes**. Embeddings: cancelados; satélite v2 local en curso.

## Fase 0 — Setup, credenciales y validación de acceso a datos

**Objetivo:** verificar que cada fuente entrega datos reales antes de construir el pipeline.

| # | Tarea | Estado | Notas |
|---|---|---|---|
| 0.1 | Estructura repo + Docker (`python`, `postgis` Fase 3, `duckdb-cli`) | ✅ | `docker-compose.yml`; PostGIS con profile `serving` |
| 0.2 | Altas de credenciales (AEMET, CDS, GEE, CNIG, EGIF manual) | ✅✅✅✅/⏳ | AEMET ✅ CDS ✅ (licencia aceptada). GEE: API key AIza rechazada por EE (OAuth requerida) — **deparked a favor de STAC local** |
| 0.3 | Probes de fuentes + matriz de disponibilidad | ✅ | `data/catalog/availability_matrix.md` |
| — | Reorganización `data/` en lake raw-inmutable con manifests | ✅ | `manifest.json` (sha256 + cover) por fuente |

**Hallazgos de la Fase 0** (importantes, no obvios):
- **EGIF XML export incluye la causa** (`pif_causa`: `idcausa` 400=intencionado) y
  coordenadas decimales ya presentes, además de UTM H29 → mejor que el MVP "sin causa".
- Tramos XML EGIF: 2004–2023 consolidado; 164.620 partes; 83% con coords válidas;
  **23.350 igniciones localizadas 2017–2023** → ventana entrenamiento: train 2017–2022, test 2023–2024.
- **Los .rar del MITECO eran ZIP** (edición `MFE25_50_Dic2021`) con un único SHP
  `MFE_11.shp` = **toda Galicia** (96.665 polígonos) con `n_MODCOM` (modelo Rothermel
  1–13) ya calculado → no hace falta mapear clases de combustible.
- MDT25 IGN: hojas hay que ir una por una → se descartó para MVP: **Copernicus GLO-30**
  (30 m, sin auth, 14 tiles cubren Galicia) hace el rol; MDT25 queda como detalle Fase 5+.
- La hora del parte en `pif_condiciones.hora` a veces trae fechas 2025 → no usar como del parte.
- **Liímites documentados**: AEMET 40 conexiones/min (FAQ §3.5); Open-Meteo 429 por minuto/hora/día "counted per call". Backfill satelital corriendo en local via COGs AWS (sentinel-cogs, ~10 tiles MGRS cubren Galicia incl. 1 en UTM zone 30 → reproyectar al mosaico).
- AEMET OpenData deliberately: rate limit se congela tras ~15k peticiones → backfill histórico
  movido a **Open-Meteo** (1 petición/estación, sin claves). AEMET queda para tiempo real.

## Fase 1 — Ingesta y Feature Engineering (Galicia) — ~70%

| # | Sub-tarea | Estado |
|---|---|---|
| 1.1 | Malla H3 res9 con fracción forestal + modelo Rothermel mayoritario | ✅ 245.172 celdas |
| 1.2 | DEM estático: `elev/slope/aspect` por celda (GLO-30) | ✅ (slope re-calcada en metros reales tras fix) |
| 1.3 | Viaria: `dist_to_road` (polígono-celda→vía), `road_class`, `road_density_1km` (buffer 1 km), `road_fuel_interface` | ✅ |
| 1.4 | Fuel grid: MFE25 `n_MODCOM` rasterizado al grid del DEM | ✅ `data/interim/mfe25/fuel_modcom_30m.tif` |
| 1.5 | Meteorología histórica diaria (FWI/DC/DMC): **Open-Meteo** 58 estaciones | ~25 estaciones completadas (cupo por hora; idempotente, reanudable) |
| 1.6 | Meteorología real-time ob horaria AEMET | ✅ script (58 estaciones validadas; cron por deploy) |
| 1.7 | ERA5-Land SWC/VPD netCDF | 🟢 ~20/32 ficheros caen solos (cola CDS) |
| 1.8 | **S2 LOCALES** vía STAC earth-search (AWS sin auth): composites de verano NDMI/NDVI/NBR 2017–2024 @100m → sampler local | 🟢 implementado `s2_stac.py`; batch años en cola (2022 primero) |
| 1.8b | ~~Embeddings (AlphaEarth/Prithvi)~~ | ❌ **CANCELADOS (decisión 2026-10-29)**: la ablación F2 se hace con met + índices espectrales + estáticas; sin GEE/APIs de Google |
| 1.8c | GEE original (warm v1) | 🅿️ *parked*: código defendable (`gee_export.py`) pero pausado — reactivo si STAC local no da cobertura |
| 1.9 | Etiquetas v1 (EGIF h→h9 + EFFIS polígonos→celdas + NBR backup) | ✅ `labels/v1/` (11.8k positivo celd×día) |
| 1.10 | Sanity de las estáticas: distribución dist/ddensidad/nivel de consistencia | ✅ primera pista: mediana `dist_to_road`=0 (las pistas cruzan la celda); la señal diferencial la darán densidad + interfaz arcén-combustible |

## Fase 2 — Entrenamiento y validación de la idea (Gate)

| # | Sub-tarea | Estado |
|---|---|---|
| 2.1 | Dataset etiquetas+features (muestreo de negativos 1:20) | ⏳ |
| 2.2 | Baseline FWI puro (cffdrs port) + logística | ⏳ |
| 2.3 | LightGBM 0–4 (met + satélite espectral + estáticas, **sin embeddings**), focal/weights | ⏳ |
| 2.4 | Ablation: (1) FWI → (+2) satélite → (+3) topografía → (+4) viaria | ⏳ (la demostración central de la validación) |
| 2.5 | Split espacial-temporal: por año (train 2017–2022 / test 2023–2024) + dejar Ourense fuera (transferencia espacial) | ⏳ |
| 2.6 | Métricas y **Gate PR-AUC + recall clase Extremo ≥0.7** | 🟡 draft inicial |
| 2.7 | **Feature recencia** (y-1 días por último fuego <5 km; exenciones de contaminación redujéronse a ±7d — la recencia aprende el reburn) | ✅ añade **test PR-AUC 0.350, recall@5% 56%** (lgbm primer smoke) |
| 2.8 | **Búsqueda multi-modelo Optuna** (`model_search.py`): LGBM · XGB · CatBoost · HGB · RF · ExtraTrees · **MLP** · lineal; split interno 2017-21/val 22 sin tocar test | ✅ catboost/lgbm líderes (0.353 test) |
| 2.9 | **Hard negatives estratificados** (near-space 1-3 km mismo día + near-time 8-21 d antes) — técnica WUS-2025/FireCastRL | ✅ +grabes salto: ROC-AUC 0.87→0.91 |
| 2.10 | **Balanced-ensemble** (25×LGBM con undersamples balanceados, Texas-2025) | ✅ +30% PR-AUC relativo; ROC 0.9098 |
| 2.11 | **Features sequía** (SPEI-30/90 + anomalías FFMC/DMC/DC — México-2021/NHESS-2025) | ✅ integradas; dmc_anom top-4 importancia |
| 2.12 | **LSTM temporal** (ventanas 10 días — Kondylatos) | ❌ descartado con evidencia (5.2× vs 24× LGBM): la ventana ya capturada por features flat |
| 2.13 | **Suavizado espacial H3** (matriz ring-k, producción) + baremo fijo de niveles (P75/92/98/99.7) | ✅ implementado en viz+API |
| 2.14 | Clusters por estáticas + modelo por región | ❌ descartado (peor que global: la señal regional ya la captura el parent-day) |

## Pendiente día siguiente (2026-10-30)

1. ~~TIFFs S2 2023/2024~~ ✅ (8/8, transform arreglado)
2. ~~Rebuild dataset final~~ ✅ (267k res9, 72k res8)
3. ~~Re-evaluar todo~~ ✅ (ROC 0.9179-0.9348)
4. ~~Regenerar vídeos~~ ✅ (6 resoluciones)
5. **FIRMS VIIRS labels**: requiere MAP_KEY gratuito (registro en
   https://firms.modaps.eosdis.nasa.gov/api/map_key/) — el módulo está
   (`firms_viirs.py`) pero sin key no descarga. Pendiente registro.
6. **Spatial block CV** ✅ implementado (`spatial_block_cv.py`): 5-fold KMeans
   blocks, ROC-AUC 0.886±0.041 (temporal 0.892, Ourense 0.783) — el modelo
   generaliza espacialmente.
7. **CNN-2D parches espaciales**: pendiente (Ronda 3, el salto DL real)

1. **TIFFs S2 2023/2024 pendientes** (solo hay 2017–2022; el proceso quedó a medias al
   cancelar): regenerar con `uv run python -m ardintra.ingest.s2_stac 2023 2024` (~25 min).
   ⚠️ El 2023 es necesario para las features satelitales del TEST 2024 (prev-summer).
2. Rebuild dataset final + re-evaluar todo (ablación/ensemble/jerárquico) con satélite completo.
3. Regenerar vídeos res9/res8 + informe Gate definitivo + push.

## Comparativa por resolución (cerrada 2026-10-29/30)

Res9 gana en ROC (0.9121) y lift (21×); res8 empata en recall@5% (56%); las mayores
(res4-7) pierden lift por dilución de features y etiquetas agregadas. **Decisión:
res9 producción · res8 capa de presentación regional.**

## Fase 3 — Serving (Docker local)

- PostGIS+Martin + FastAPI + visor MapLibre; pipeline diario 06:00/12:00. Pendiente.

## Fase 4 — XAI + LLM narrativo (optativa, barata en local)

- TreeSHAP + JSON context + **Ollama Llama 3 8B** en Docker. Pendiente.

## Fase 5 — Cloudflare

- API → Workers (Hono) o CF Containers
- R2 para GeoParquet/catálogos y **pbf estáticos** pregenerados (no Martin dinámico)
- Postgres hosted w/ PostGIS (Neon/Supabase) via Hyperdrive
- Cron Triggers para pipeline diario
