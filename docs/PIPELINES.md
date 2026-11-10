# Ingestion pipelines — operational guide

Each module is a `python -m ardintra.ingest.<module>` entrypoint that is
**idempotent** (all of them check manifests/checkpoints and resume without
repeating work).

## From-scratch environment order

```bash
uv sync && uv pip install osmium            # Geo environment (Python 3.12)
cp .env.example .env && $EDITOR .env        # AEMET_API_KEY, CDS_API_KEY, EARTH_ENGINE_KEY
uv run python -m ardintra.ingest.openmeteo_daily      # (1) weather history — no keys, resumable
uv run python -m ardintra.ingest.bootstrap_data      # (2) egif+effis+dem+osm+mfe25
uv run python -m ardintra.features.build_static      # (3) H3 master table (~10 min)
uv run python -m ardintra.ingest.aemet.aemet_obs    # (4) hourly obs — cron every hour
uv run python -m ardintra.ingest.aemet.backfill_missing  # (4b) paced AEMET for missing stations
uv run python -m ardintra.ingest.era5.era5land_cds  # (5) netCDF per year/variable (idempotent)
uv run python -m ardintra.ingest.s2_stac -- 2017    # (6) S2 composites @100m per year (no auth)
```

## Rate-limit policy (learned the hard way)

| Service | Operating rule |
|---|---|
| AEMET | For OPERATION only (hourly obs via cron: 1 call/h). HISTORICAL goes through Open-Meteo. Never run backfills against AEMET: its limit froze the session after ~14k requests |
| Open-Meteo | per-station checkpoints + 2 s between requests; on 429 (minute/hour), resume later — do not hammer |
| EFFIS | 1 request/year; exponential retry with tenacity |
| CDS | queued requests: 1 per variable-year, resumable by existing file |

## Manifests and reproducibility

- Every `data/raw` folder has a `manifest.json` (sha256, coverage, license).
- `ardintra.ingest.bootstrap_data` creates/regenerates manifests for the
  manual downloads (EGIF, MDT25, MFE25, OSM) if missing.
- `data/interim/` can be deleted wholesale: everything is rebuilt by the modules.

## Post-pipeline checks (quick QA)

```bash
uv run python - <<'EOF'
import geopandas as gpd, pandas as pd
st = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
print("cells:", len(st), "| "
      "slope p50:", st.slope_deg.quantile(.5).round(1), "| "
      "forest_frac p50:", st.forest_frac.quantile(.5).round(2), "| "
      "road density p50:", st.road_density_1km.quantile(.5).round(2))
eg = pd.read_parquet("data/interim/egif/parts_normalized.parquet")
print("EGIF reports:", len(eg), "| informed causes:", eg.idcausa.notna().mean())
EOF
```
