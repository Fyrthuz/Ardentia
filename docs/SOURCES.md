# Data sources — access, credentials, limits and quirks

> Checklist as of 2026-10-29 in `data/catalog/availability_matrix.md`.

## Access summary

| Source | Auth | Critical limit | Role in the system |
|---|---|---|---|
| EGIF (MITECO portal) | manual (no public API) | export in batches; XML→reports | labels with cause (idcausa/motivation/cause agent), `deteccion` date, coords |
| EFFIS RDII WFS | **none** | WFS 1.0.0 (1.1.0 swaps lon/lat); ~2 min latency per year | perimeters ≥30 ha 2017–2025 |
| AEMET OpenData | API key (expires 90 d) | **aggressive rate limit** (~15k req freeze the account; see note) | real time: hourly obs `/observacion/convencional/todas` |
| Open-Meteo Archive | **none** | 429 per minute and **per HOUR** with large payloads | daily historical backfill (57 stations; 1 req = 3,589 days × 5 vars) |
| Open-Meteo Historical Forecast | **none** | archive starts 2022-01 (HRES seamless) | day-D forecast issued D-1 (operational regime) |
| Copernicus CDS (ERA5-Land) | account + **explicit dataset license acceptance** on the portal | execution queue | SWC l1/l2, T2m, Td2m (VPD) 2017–2024 |
| **STAC earth-search + AWS sentinel-cogs** | **none** (open data) | COGs with overviews; @100 m reads with capped GDAL cache | Sentinel-2 L2A: NDMI/NDVI/NBR summer composites 2017–2024 (10 MGRS tiles; 30TTM in zone 30 → reproject) |
| ~~GEE / AlphaEarth~~ | ~~OAuth~~ | `AIza…` API key **unsupported** by EE (401 CREDENTIALS_MISSING) | **CANCELLED** (decision 2026-10-29); `gee_export.py` module parked |
| ~~Local Prithvi / Clay~~ | HF weights | RTX 5080 available but unnecessary | **CANCELLED** — the F2 ablation doesn't use embeddings |
| OSM Geofabrik | none | — | road network (482,881 Galician segments incl. 114k forest tracks) |
| MFE25 (MITECO) | manual download | ZIP (not rar) | Rothermel ModeloComb 1–13 per polygon, FCC, species |
| IGN/CNIG | manual download | individual sheets | MDT25 (backup), BCN200 as future road comparison |
| Copernicus GLO-30 | **none** (AWS open data) | — | DEM/slope/aspect (MVP base) |

## Discovered quirks (so you don't trip twice)

1. **AEMET mixed encodings**: the station inventory is served in latin-1
   (0xD3 = "Ó"). Two-step pattern: the first JSON returns the real URL
   (`datos`); the body may come back **empty** for stretches with no data →
   treat as empty, not an error.
2. **AEMET daily climatologies**: dates in `YYYY-MM-DDTHH:MM:SSUTC` format
   (plain ISO date doesn't work). Data with decimal commas (`6,1`).
   **Documented limit (official FAQ §3.5): 40 connections/min per user**; the
   429 code means "exceeds the per-minute flow limit". The backfill uses
   2.5 s/req (~24/min) and on 429 retries the SAME stretch after cooldown
   (never discards data). Maximum tested chunk: 31 days (30 to be safe). The
   ACK and `datos` body sometimes arrive mis-encoded → decode tolerantly
   (latin-1 fallback) and discard the stretch.
3. **AEMET rate limit**: the API freezes after ~14-15k requests in one
   session (a 30-day×2-call×58-station backfill ≈ 14k) → **use Open-Meteo for
   history** and AEMET only for the real-time hourly cron.
4. **Open-Meteo free tier**: "Minutely" 429 is expected; **"Hourly" exists
   too** — the module keeps per-station checkpoints (`per_station/`) and is
   resumable.
5. **EFFIS WFS**: use `WFS 1.0.0` (correct lon,lat axes); filter
   `COUNTRY=ES` and province in `{A Coruña, Lugo, Ourense, Pontevedra}` after
   the bbox query (the bbox catches Portugal).
6. **GLO-30 is EPSG:4326** (degrees): slope requires converting the per-pixel
   gradient to meters (`a_deg·111_320·cos(lat)`) — a first version with the
   unit in degrees gave ~90° slopes (fixed: p95=28° for Galicia).
7. **MFE25**: the official "rar" is a **ZIP** (check `PK` before unrar);
   single layer for all Galicia, CRS EPSG:25830 → reproject to the GLO-30
   4326 grid.
8. **STAC S2 (earth-search)**: `grid:code` per MGRS tile, `eo:cloud_cover`.
   Bands by STAC name (`nir`, `nir08`/B8A, `swir16`/B11, `swir22`/B12, `scl`).
   The COG carries `raster:bands` scale/offset (0.0001 / -0.1) that must be
   applied BEFORE the normalized differences. SCL mask: filter 3/8/9/10/11.
   GDAL_CACHEMAX capped at 512 MB per reading process to avoid blowing RAM
   with COG decimations.
9. **EGIF**: `numeroparte` is the business key; `idpif` for internal joins.
   The `hora` in `pif_condiciones` sometimes reads 2025 → ignore as report time.
10. **H3 res9 with a single geometry type**: for road distances use the
    **cell polygon** (`h3.cell_to_boundary`), not the centroid — bare
    centroids sit ~50 m off the arc and fake `dist_to_road=9999`.
