# Full preprocessing pipeline

Chain raw → interim → processed → features → models. No GEE and no API keys
required for the satellite data (public STAC on AWS). The lake layout is in
[DATA_MODEL.md](DATA_MODEL.md) and the operational run guide in
[PIPELINES.md](PIPELINES.md); this document explains the what-and-why of each
step.

## 1. Per-source ingestion and cleaning

| Source | Key processing step | Product |
|---|---|---|
| **EGIF** (164,620 fire reports 2016-2024) | official XLSX normalization → long parquet; paraje→lat/lon geocoding; `in_bbox` (Galicia) filter; dedupe on `numeroparte`; cause codes preserved (`idcausa`: 100=lightning, 3xx=negligence, 4xx=arson, 5xx=re-ignition, 6xx=other) | `data/interim/egif/parts_normalized.parquet` |
| **EFFIS WFS** | burned perimeters ≥30 ha (WFS 1.0.0), intersected with H3 cells | `labels/v1/burned_cells.parquet` |
| **AEMET OpenData** | hourly obs in real time (crontab); historical backfill with pacing (~15k request limit) | `data/raw/aemet/` |
| **Open-Meteo Archive** | daily backfill 2017→today for 57 stations (tmax/tmin/prec/wind/RH) — one bulk call per block of 10 stations | `data/raw/openmeteo/daily_2017_today.parquet` |
| **Open-Meteo Historical Forecast** | day-D weather issued on D-1 (lead ~24h, archived HRES runs) 2023→today | `data/raw/openmeteo/forecast_daily.parquet` → `met_fc.parquet` |
| **ERA5-Land (CDS)** | soil moisture 2 layers + VPD, resampled to the cell grid | `era5_features()` |
| **Sentinel-2 (STAC AWS)** | SUMMER composites per year (NDMI/NDVI/NBR @100 m, 8/8 years 2017-2024), local mosaic without auth | `data/raw/s2/s2_summer_*.tif` |
| **OSM** | full drivable network + 446k buildings; derived: `dist_to_road`, `road_class`, `road_density_1km`, `road_fuel_interface` (a road whose shoulder touches dense fuel) | `res9_static.parquet` |
| **DEM GLO-30 + MDT25 IGN** | elevation, slope, aspect | idem |
| **MFE25** | forest map: `fuel_model` (Rothermel `n_MODCOM` 1-13), `forest_frac` | idem |

## 2. H3 gridding

- **res9 (~105 m, 245,172 forest cells)** = production (per-cell prediction)
- res7 (5.2 km, 5,710 hexagons) = regional aggregation (Stage 2)
- Master table `h3_cells/res9_static.parquet`: all static features per cell.

## 3. Labels and sampling (res9)

- Positive: EGIF/EFFIS ignition in the cell-day (`surface_ha ≥ 0.1`)
- Primary target `big10`: ignition with `surface_ha ≥ 10` (1,496 in 2017-24;
  137 in test 2023-24)
- Negatives (seed 42, deterministic): 150/day random in fire season (3/day in
  winter) + NEAR-SPACE tier (1-3 km ring around same-day fires, 40/day) +
  NEAR-TIME (the cell that will burn 8-21 days later — temporal "hard
  negative", no look-ahead)
- Contamination filter: discards (cell, day) pairs within 1 km of a fire
  ±7 days (they are propagation of the same event, not true absence)
- Strict temporal split: train ≤2022 (206,714 rows) · test 2023-24 (60,498).
  The test set never enters any modeling decision.

## 4. Features (48 + cause, per cell-day row)

- **Station weather/FWI** (18): tmax, RH, wind, precip, ffmc/dmc/dc/isi/bui/fwi,
  ffmc_7d, fwi_7d, fwi_prev, ffmc/dmc/dc anomalies (vs 7d mean), spei30, spei90.
  Per cell via **IDW over the 3 nearest stations** (weights 1/d²).
- **ERA5-Land** (4): swc_l1/l2_mean, vpd_max, vpd_mean
- **Prev-summer satellite** (4): NDMI/NDVI/NBR of the previous summer + NDMI anomaly
- **Statics** (10): forest_frac, fuel_model, slope, aspect sin/cos, elevation,
  log distance to road, log road density, road-fuel interface, road_class
- **Fire recency** (3): days since last fire ≤5 km, distance to it, burned ≤1 km in past year
- **Ignition lags** (3): ignition count in the cell (3d) and in 2.5/5 km radii (2d/7d)
- **Seasonal/human** (5): doy sin/cos, weekend flag, road-density×weekend, lat/lon
- **Cause** (analytical, not a model feature): `cause`, `es_humana` (76% human), `es_rayo`

## 5. Modeling and calibration

1. **LGBM** (600 trees, lr 0.05, 31 leaves, `scale_pos_weight` ≈ 39) over the
   sampled rows → raw `P(ignition ≥10 ha | cell, day)`.
2. **Isotonic regression** (auxiliary model ≤2021, fitted on 2022) → calibrated
   probability: Brier 0.0118→0.0065, near-perfect reliability in 8/10 bins.
3. **Regional aggregation Stage 2** (res7): λ_r = Σ p_cal per region →
   hierarchical Poisson GLM `log μ = α + β·log(1+λ_r) + MEM·γ` (80 Moran
   eigenvectors of the hex adjacency graph; α selected by CV within 2023) →
   P(≥1 fire in the region that day) and λ Galicia = Σμ.
4. **Risk levels**: cells by within-day percentile (FWI-gated palette);
   regions by ×N risk ratio (vs the 0.028% region-day base rate).

## 6. Day-ahead forecast (operational regime)

- The day-D weather is replaced by the forecast issued on D-1 (Open-Meteo
  Historical Forecast API; per-station NGR-lite bias correction fitted on
  2023 — forecast FWI RMSE 3.03→2.37)
- Slow indices (SPEI, code anomalies) remain in observation mode
- Measured cost: −1pp aggregate ROC-AUC, operational recall@K within noise
  → the system predicts tomorrow with today's data (no leakage)

## 7. Operational evaluation (full grid)

82 test days with ≥1 fire ≥10 ha: ALL 245k cells are built each day and the
top K% of the territory is flagged (mean daily recall):
top-1% → 3-5% · top-5% → 13-19% · **top-10% → 29.5-31.2%** of the 137 fires.
At region level: AUC 0.755, recall@10% region-days 39.4% (Stage 2).
