# WildfireRisk-ES (codename: Ardentia)

Sistema de evaluación predictiva y explicable del riesgo de ignición y propagación
de incendios forestales en España a escala de malla hexagonal H3 res9 (~105 m·celda),
con piloto en **Galicia** y despliegue final en **Cloudflare**.

> Documentación detallada en [`docs/`](docs/):
> - [ROADMAP.md](docs/ROADMAP.md) — plan por fases y estado de implementación
> - [DATA_MODEL.md](docs/DATA_MODEL.md) — organización del lake, features y etiquetas
> - [SOURCES.md](docs/SOURCES.md) — fuentes de datos, credenciales, rate limits y quirks

## Esencia del producto

Los índices convencionales (FWI) son meteorológicos y de baja resolución. Este sistema
integra **tres capas** que el FWI ignora:

1. **Estática territorial**: topografía (GLO-30), modelos de combustible Rothermel
   (MFE25, `n_MODCOM` 1–13 directamente disponible), y **factor antrópico** — red
   viaria (OSM/BTN25) con `dist_to_road`, `road_class`, `road_density_1km` y
   `road_fuel_interface` (vía cuyo arcén toca combustible denso).
2. **Dinámica biofísica**: NDMI/NDVI/NBR de Sentinel-2 vía **STAC earth-search (AWS, sin
   auth) con composites locales en GPU/CPU**; meteorología AEMET (tiempo real) + histórico
   oficial; ERA5-Land para humedad de suelo/VPD. propiedad: **sin GEE, sin API keys para
   datos satellite**.
3. **Explicabilidad**: TreeSHAP por celda + boletín operativos generado con LLM.

**Criterio de validación de la idea (Fase 2 Gate):** el modelo multinivél
*piloto* debe superar un baseline puro-FWI en **PR-AUC**, con ablations que
demuestren la ganancia específica de las variables viarias/antrópicas.
(Decisión 2026-10-29: los embeddings satelitales (AlphaEarth/Prithvi) quedan
**fuera del MVP**; la ablación se hace con met + índices espectrales + estáticas.)

## Quick start

```bash
uv sync && uv pip install osmium
cp .env.example .env   # rellenar claves (ver docs/SOURCES.md)
.venv/bin/python -m ardintra.ingest.bootstrap_data --sources egif,effis,dem,osm,mfe25
```

Pipelines independientes (`uv run python -m <módulo>`):

| Módulo | Qué hace |
|---|---|
| `ardintra.ingest.egif.parse_egif` | EGIF XML/zip → `parts_normalized.parquet` + QA |
| `ardintra.ingest.effis_wfs` | Perímetros RDII 2017–2025 vía WFS (sin auth) |
| `ardintra.ingest.dem_glo30` | Copernicus GLO-30 → mosaico + pendiente + orientación |
| `ardintra.ingest.osm_roads` | OSM pbf → `roads_driveable.parquet` (clases normalizadas) |
| `ardintra.ingest.mfe25` | MFE25 → SHP (con ModeloComb Rothermel por polígono) |
| `ardintra.ingest.aemet.aemet_obs` | AEMET obs horaria + estaciones (tiempo real) |
| `ardintra.ingest.openmeteo_daily` | Backfill histórico meteorológico (_checkpoint idempotente) — AEMET paced para estaciones missing |
| `ardintra.ingest.s2_stac` | Composites S2 LOCALES (NDMI/NDVI/NBR verano 2017–2024 @100 m, sin auth) |
| `ardintra.features.build_static` | Tabla maestra H3 res9 con todas las estáticas |

## Resultados finales del ciclo de experimentación (2026-10-29)

**Métricas defendibles** (test 2023–24 jamás tocado · ROC-AUC como métrica comparable con literatura):

| Configuración | ROC-AUC | Lift | recall@5% |
|---|---|---|---|
| FWI solo (estado del arte) | 0.67 | 0.6× — peor que azar | 9% |
| LGBM balanced-ensemble ×25 (met+sequía+recencia+satélite+viaria) | **0.9098** | **21-24× azar** | **56%** |
| Jerárquico fusión (fino+regional) | 0.8809 | 12-18× | 50% |
| Comparativa literatura: RF-Kondylatos 0.898 · ConvLSTM 0.926 · NHESS-Europa 0.81 | | | |

Detalles por resolución y experimento en `reports/` (ver docs/ROADMAP.md).

## Simulación visual

`reports/simulacion_2023_res9.mp4` — temporada 2023 día a día: celdas coloreadas
por score (baremo fijo: P75/92/98/99.7 de la temporada), fuegos reales apareciendo
(⭐ ≥10 ha), zoom-inset sobre el mayor fuego del día, 3 s/frame.

## Estado resumido (2026-10-29)

- Fase 0 completa (ingesta y validación de fuentes) ✔
- Fase 1: estáticas construidas ✔ (245.172 celdas H3 res9 forestales);
  meteorología oficial en curso (AEMET paced 18 estaciones restantes; ERA5-Land ~20/32);
  satélite via STAC local (s2_stac) en marcha. **Embeddings cancelados para MVP** (decisión
  2026-10-29); módulo GEE quedará como alternativa a future (parked).
- Fase 2 (modelo vs FWI): pendiente; ya hay etiquetas disponibles
  (23.350 igniciones localizadas de EGIF 2017–2023 + perímetros EFFIS).
- Fase 3 (serving/postgis/martin/visor — prototipo mini-API ya operativo en :8001),
  Fase 4 (XAI+LLM) y Fase 5 (Cloudflare): pendientes.

## Comparativa por resolución de hexágono (balanced-ensemble 25×, target ≥10 ha, test 2023-24)

| res | celda | filas | prev big10 test | PR-AUC | **lift** | ROC-AUC | recall@5% |
|---|---|---|---|---|---|---|---|
| **res9** | 105 m | 267k | 0.23% | 0.0538 | **23.4×** | **0.9179** | 54% |
| **res8** | 740 m | 72k | 0.82% | 0.1247 | 15.2× | 0.9021 | 51% |
| res7 | 5.2 km | 28k | 2.38% | 0.1451 | 6× | 0.8643 | 36% |
| res6 | 36 km² | 30k | 2.04% | 0.1336 | 7× | 0.8737 | 39% |
| res5 | 253 km² | 16k | 4.07% | 0.1622 | 4× | 0.8488 | 29% |
| res4 | 1.770 km² | 11k | 5.11% | 0.2656 | 5× | 0.8757 | 30% |

⚠️ El AP absoluto sube con celdas grandes SOLO porque sube la prevalencia; el **lift honesto**
muestra que res9/res8 son muy superiores (5-21× vs 4-7×). Los vídeos de simulación existen
para las 6 resoluciones (`reports/simulacion_2023_res*.mp4`).

## Próximos pasos (retomar mañana)

1. **Granularidad**: res9 (245k celdas) tiene demasiada densidad visual para el mapa;
   decidir entre res8 como producción o res9 con renderizado optimizado (tiles/vector).
2. **S2**: los 8 TIFFs están listos; el dataset v4 ya los integra (ndmi/ndvi/nbr prev-summer
   al 37.8% y subiendo). Re-evaluar ablación completa.
3. **Ensemble final**: re-run de `ensemble_gate.py` con el dataset v4 limpio.
4. **Push periódico**: mantener el repo sincronizado con los reports.
