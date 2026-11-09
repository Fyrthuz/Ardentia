# Modelo de datos y organización del lake

Todo el lake vive en `data/` (excluido de git salvo los `manifest.json` y `data/catalog/`).
Principio: **raw inmutable · interim derivable (borrable) · processed reconstructible**.

```
data/
├── raw/                                # EXACTAMENTE tal cual se descargó
│   ├── egif/manual/2026-10-29/         # Xml_*.zip y Xlsx_*.zip del Buscador EGIF (descarga manual — no hay API)
│   ├── effis/year={2017..2025}/perimeters.parquet   # WFS per-token (descarga programática)
│   ├── dem_glo30/{lat}_{lon}.tif       # 14 tiles de 1° (Copernicus, AWS open data sin auth)
│   ├── osm/galicia-2026-10-28.osm.pbf  # Geofabrik snapshot denominado con fecha
│   ├── mfe25/mfe_galicia.rar           # ZIP en realidad (ed. Dic2021) + unpacked/
│   ├── mdt25/manual/*.tif              # 2 hojas (respaldo; la base del MVP es GLO-30)
│   ├── s2/s2_summer_{year}.tif         # NDMI/NDVI/NBR @100m (STAC local AWS, mosaico 10 tiles)
│   ├── aemet/
│   │   ├── stations.parquet            # inventario 58 estaciones Galicia (coords decimales)
│   │   ├── obs_hourly/date={día}/obs.parquet      # cron horario de operación (paso 2)
│   │   └── missing_backfill/estacion={idema}.parquet  # backfill paced AEMET de las 19 que OpenMeteo no bajó
│   ├── openmeteo/
│   │   ├── per_station/estacion={idema}.parquet    # checkpoint idempotente backfill
│   │   └── daily_2017_today.parquet    # serie consolidada 2017→hoy (tmax/tmin/prec/wind/RH)
│   └── era5land/{var}_{year}.nc        # netCDF (SWC l1/l2, t2m, td2m → VPD) 32/32 completos
├── interim/                            # derivados reconstructibles
│   ├── egif/parts_normalized.parquet + qa_egif.json
│   ├── dem/{dem,slope,aspect}_galicia.tif
│   ├── osm/roads_driveable.parquet     # 482.881 vías con clase normalizada
│   └── mfe25/fuel_modcom_30m.tif
├── processed/
│   ├── h3_cells/res9_static.parquet    # TABLA MAESTRA ESTÁTICA res9 (245.172 celdas)
│   ├── h3_cells/res8_static.parquet    # aggregada a res8 (38.013 celdas)
│   ├── features/v1/dataset.parquet         # dataset entrenamiento res9
│   ├── features/v1_res8/dataset.parquet    # dataset entrenamiento res8
│   └── labels/v1/ignitions.parquet     # etiquetas
└── catalog/availability_matrix.md
```

## Convenciones generales

- CRS: WGS84/4326 para almacenar; ETRS89/UTM29 (EPSG:25829) para distancias métricas.
- Resoluciones H3 probadas: **res9** (~105 m, 245k celdas) y **res8** (~740 m, 38k celdas).
- `manifest.json` por fuente con sha256 y cobertura; versionado en git aunque los datos no.

## Bloques de features del modelo

### Estáticas (por celda, de `res{9,8}_static.parquet`)
| Feature | Descripción | Fuente |
|---|---|---|
| `forest_frac` | fracción forestal de la celda (muestreo 60m; en res8 media de las hijas) | MFE25 |
| `fuel_model` | modelo de combustible Rothermel mayoritario (1–13) | MFE25 `n_MODCOM` |
| `elev_m`, `slope_deg` | cota y pendiente | Copernicus GLO-30 |
| `aspect_sin`, `aspect_cos` | orientación descompuesta (circular → 2 componentes) | GLO-30 |
| `dist_to_road`, `dist_to_road_log` | distancia a la vía más cercana (m y log1p) | OSM |
| `road_class` | clase de esa vía (autovia…pista_forestal/camino_rural) | OSM |
| `road_density_1km`, `road_density_log` | km de vía por km² en buffer 1 km (y log1p) | OSM |
| `road_fuel_interface` | binaria: la vía más cercana tiene combustible denso en arcén | MFE25 ∩ OSM |

### Dinámicas (por celda-día, en `dataset.parquet`)
| Grupo | Features | Fuente |
|---|---|---|
| Met diaria (IDW 3 estaciones más cercanas) | `tmax_c, rh_mean, wind_max_kmh, prec_mm` | AEMET 58 est (openmeteo + backfill paced) |
| FWI (port cffdrs verbatim) | `fwi, ffmc, dmc, dc, isi, bui` | calculado de la met diaria |
| Tendencias | `ffmc_7d, fwi_7d, fwi_prev` | rolling 7d sobre las series FWI |
| Sequía | `spei30, spei90` (PET Hargreaves) · `ffmc_anom, dmc_anom, dc_anom` (anomalía vs clima del día calendario) | AEMET/openmeteo raw |
| Humedad de suelo / VPD | `swc_l1_mean, swc_l2_mean, vpd_max, vpd_mean` (3-horario → diario) | ERA5-Land 32/32 |
| Satélite (SIN look-ahead) | `ndmi_summer, ndvi_summer, nbr_summer, ndmi_anom` — composite del verano del AÑO ANTERIOR | Sentinel-2 vía STAC local |
| Recencia (memoria de fuel quemado) | `dias_ultimo_fuego_cercano` (días desde el último fuego <5 km), `dist_ultimo_fuego_m`, `ardio_1km_1an` | EGIF histórico |
| Calendario antrópico | `es_finde`, `dens_x_weekend` (interacción viaria×finde) | fecha |

## Etiquetas

- **Cualquier ignición ≥0.1 ha** (los micro-fogatas <0.1 ha = 40% de los partes pero el 0.09% del área quemada → filtrados como ruido de etiqueta)
- **≥10 ha** ("fuego importante", el target principal `big10`; 1.496 en train 2017-22 · 137 en test 2023-24)
- Negativos deterministas (seed 42): 150/día random en temporada + 3/día en invierno (ratio global ~38:1) + **hard negatives** (NEAR-SPACE: anillo 1-3 km de fuegos del mismo día; NEAR-TIME: la celda 8-21 días antes, sin look-ahead) con exclusión de propagación (fuegos <1 km/±7 d).
  Los muestreos alternativos (estratificación por combustible, high-danger-site) quedaron refutados — ver [EXPERIMENTS.md](EXPERIMENTS.md).
