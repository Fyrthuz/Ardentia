# Pipelines de ingesta — guía operativa

Cada módulo es un entrypoint `python -m ardintra.ingest.<módulo>` **idempotente**
(todos apagan por manifest/checkpoint y reanudan sin repetir trabajo).

## Orden de arranque de un entorno desde cero

```bash
uv sync && uv pip install osmium            # entorno Geo (Python 3.12)
cp .env.example .env && $EDITOR .env        # AEMET_API_KEY, CDS_API_KEY, EARTH_ENGINE_KEY
uv run python -m ardintra.ingest.openmeteo_daily      # (1) histórico met — sin claves, reanudable
uv run python -m ardintra.ingest.bootstrap_data      # (2) egif+effis+dem+osm+mfe25
uv run python -m ardintra.features.build_static      # (3) tabla maestra H3 (~10 min)
uv run python -m ardintra.ingest.aemet.aemet_obs    # (4) obs horaria — cron cada hora
uv run python -m ardintra.ingest.aemet.backfill_missing  # (4b) paced AEMET de estaciones faltantes
uv run python -m ardintra.ingest.era5.era5land_cds  # (5) netCDF por año/variable (idempotente)
uv run python -m ardintra.ingest.s2_stac -- 2017    # (6) composites S2 @100m por año (sin auth)
```

## Rate-limit policy (aprendido a dolor)

| Servicio | Regla operativa |
|---|---|
| AEMET | Sólo para OPERACIÓN (obs horaria por cron: 1 llamada/h). El HISTÓRICO va por Open-Meteo. Nunca lanzar backfills contra AEMET: su límite congeló la sesión tras ~14k peticiones |
| Open-Meteo | checkpoint por estación + 2 s entre peticiones; ante 429 (minuto/hora), reanudar más tarde — no martillear |
| EFFIS | 1 petición/año; retry exponencial con tenacity |
| CDS | solicitudes en cola: 1 por variable-año, reanudable por fichero existente |

## Manifiestos y reproducibilidad

- Cada carpeta del `data/raw` lleva `manifest.json` (sha256, cobertura, licencia).
- `ardentra.ingest.bootstrap_data` crea/regenera los manifests de las descargas
  manuales (EGIF, MDT25, MFE25, OSM) si no existen.
- `data/interim/` se puede borrar entero: todo se reconstruye con los módulos.

## Comprobaciones post-pipeline (QA rápida)

```bash
uv run python - <<'EOF'
import geopandas as gpd, pandas as pd
st = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
print("celdas:", len(st), "| "
      "slope p50:", st.slope_deg.quantile(.5).round(1), "| "
      "forest_frac p50:", st.forest_frac.quantile(.5).round(2), "| "
      "densidad p50:", st.road_density_1km.quantile(.5).round(2))
eg = pd.read_parquet("data/interim/egif/parts_normalized.parquet")
print("EGIF partes:", len(eg), "| causas informadas:", eg.idcausa.notna().mean())
EOF
```
