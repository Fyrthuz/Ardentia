# Data model and lake organization

The whole lake lives in `data/` (excluded from git except `manifest.json`
files and `data/catalog/`). Principle: **raw immutable · interim derivable
(safe to delete) · processed reconstructible**.

```
data/
├── raw/                                # EXACTLY as downloaded
│   ├── egif/manual/2026-10-29/         # Xml_*.zip and Xlsx_*.zip from the EGIF portal (manual download — no API)
│   ├── effis/year={2017..2025}/perimeters.parquet   # WFS per-year (programmatic)
│   ├── dem_glo30/{lat}_{lon}.tif       # 14 tiles of 1° (Copernicus, AWS open data, no auth)
│   ├── osm/galicia-2026-10-28.osm.pbf  # Geofabrik snapshot named by date
│   ├── mfe25/mfe_galicia.rar           # actually a ZIP (Dec 2021 edition) + unpacked/
│   ├── mdt25/manual/*.tif              # 2 sheets (backup; the MVP base is GLO-30)
│   ├── s2/s2_summer_{year}.tif         # NDMI/NDVI/NBR @100m (STAC on AWS, 10-tile mosaic)
│   ├── aemet/
│   │   ├── stations.parquet            # inventory of 58 Galician stations (decimal coords)
│   │   ├── obs_hourly/date={day}/obs.parquet      # hourly crontab of operation (step 2)
│   │   └── missing_backfill/estacion={idema}.parquet  # paced AEMET backfill for the 19 stations OpenMeteo missed
│   ├── openmeteo/
│   │   ├── per_station/estacion={idema}.parquet    # idempotent backfill checkpoint
│   │   ├── daily_2017_today.parquet    # consolidated series 2017→today (tmax/tmin/prec/wind/RH)
│   │   └── forecast_daily.parquet      # day-D forecast issued D-1 (operational regime)
│   └── era5land/{var}_{year}.nc        # netCDF (SWC l1/l2, t2m, td2m → VPD) 32/32 complete
├── interim/                            # reconstructible derivatives
│   ├── egif/parts_normalized.parquet + qa_egif.json
│   ├── dem/{dem,slope,aspect}_galicia.tif
│   ├── osm/roads_driveable.parquet     # 482,881 road segments with normalized class
│   └── mfe25/fuel_modcom_30m.tif
├── processed/
│   ├── h3_cells/res9_static.parquet    # res9 MASTER STATIC TABLE (245,172 cells)
│   ├── h3_cells/res8_static.parquet    # aggregated to res8 (38,013 cells)
│   ├── features/v1/dataset.parquet         # res9 training dataset
│   ├── features/v1_res8/dataset.parquet    # res8 training dataset
│   └── labels/v1/ignitions.parquet     # labels
└── catalog/availability_matrix.md
```

## General conventions

- CRS: WGS84/4326 for storage; ETRS89/UTM29 (EPSG:25829) for metric distances.
- H3 resolutions tested: **res9** (~105 m, 245k cells) and **res8** (~740 m, 38k cells).
- `manifest.json` per source with sha256 and coverage; versioned in git even
  though the data itself is not.

## Model feature blocks

### Statics (per cell, from `res{9,8}_static.parquet`)
| Feature | Description | Source |
|---|---|---|
| `forest_frac` | forest fraction of the cell (60 m sampling; at res8 the mean of children) | MFE25 |
| `fuel_model` | majority Rothermel fuel model (1–13) | MFE25 `n_MODCOM` |
| `elev_m`, `slope_deg` | elevation and slope | Copernicus GLO-30 |
| `aspect_sin`, `aspect_cos` | aspect decomposed (circular → 2 components) | GLO-30 |
| `dist_to_road`, `dist_to_road_log` | distance to nearest road (m and log1p) | OSM |
| `road_class` | class of that road (autovia…pista_forestal/camino_rural) | OSM |
| `road_density_1km`, `road_density_log` | km of road per km² in a 1 km buffer (and log1p) | OSM |
| `road_fuel_interface` | binary: nearest road has dense fuel on its shoulder | MFE25 ∩ OSM |

### Dynamics (per cell-day, in `dataset.parquet`)
| Group | Features | Source |
|---|---|---|
| Daily weather (IDW over 3 nearest stations) | `tmax_c, rh_mean, wind_max_kmh, prec_mm` | AEMET 58 stations (openmeteo + paced backfill) |
| FWI (verbatim cffdrs port) | `fwi, ffmc, dmc, dc, isi, bui` | computed from daily weather |
| Trends | `ffmc_7d, fwi_7d, fwi_prev` | 7d rolling over FWI series |
| Drought | `spei30, spei90` (Hargreaves PET) · `ffmc_anom, dmc_anom, dc_anom` (anomaly vs calendar-day climatology) | AEMET/openmeteo raw |
| Soil moisture / VPD | `swc_l1_mean, swc_l2_mean, vpd_max, vpd_mean` (3-hourly → daily) | ERA5-Land 32/32 |
| Satellite (NO look-ahead) | `ndmi_summer, ndvi_summer, nbr_summer, ndmi_anom` — composite of the PREVIOUS year's summer | Sentinel-2 via public STAC |
| Fire recency (burned-fuel memory) | `dias_ultimo_fuego_cercano` (days since last fire <5 km), `dist_ultimo_fuego_m`, `ardio_1km_1an` | EGIF historical |
| Human calendar | `es_finde` (weekend), `dens_x_weekend` (road×weekend interaction) | date |
| Ignition cause (analytical) | `cause`, `es_humana` (76%), `es_rayo` (3%) | EGIF `idcausa` |

## Labels

- **Any ignition ≥0.1 ha** (micro-fires <0.1 ha = 40% of reports but 0.09% of
  burned area → filtered out as label noise)
- **≥10 ha** ("major fire", the main target `big10`; 1,496 in train 2017-22 ·
  137 in test 2023-24)
- Deterministic negatives (seed 42): 150/day random in fire season + 3/day in
  winter (global ratio ≈ 38:1) + **hard negatives** (NEAR-SPACE: 1-3 km ring of
  same-day fires; NEAR-TIME: the cell 8-21 days earlier, no look-ahead), with
  propagation exclusion (fires <1 km/±7 d).
  Alternative sampling schemes (fuel stratification, high-danger sites) were
  refuted — see [EXPERIMENTS.md](EXPERIMENTS.md).
