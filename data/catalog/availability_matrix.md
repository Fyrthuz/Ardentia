# Matriz de disponibilidad de fuentes (Fase 0.3) — Piloto Galicia

Actualizada: 2026-10-29 por `bootstrap_data.py` (idempotente, re-visible en cada ejecución)

| Fuente | Status | Período[^1] | Granularidad | Ticket/manifest | % listo |
|---|---|---|---|---|---|
| EGIF (MITECO) | 🟢 parseada | 2004–2023 (consolidada) | punto de ignición + causa | `data/interim/egif/parts_normalized.parquet` | 100% |
| EFFIS RDII | 🟢 descargada (WFS) | 2017–2025 | perímetro ≥30 ha | `data/raw/effis/year=*/perimeters.parquet` | 100% |
| DEM Copernicus GLO-30 | 🟢 descargada | static | 30 m (mosaico+slope+aspect) | `data/interim/dem/dem_galicia.tif` | 100% |
| OSM vias (Geofabrik) | 🟢 extraída | snapshot 2026-10-28 | lineas + clase | `data/interim/osm/roads_driveable.parquet` | 100% |
| MFE25 (MITECO) | 🟢 extraída | ed. Dic2021 | polígono con ModeloComb 1-13 (Rothermel) | `data/raw/mfe25/unpacked/MFE_11.shp` | 100% |
| MDT25 IGN | 🟡 respaldo | ed. 2021 | 25 m (solo hojas 0126/0157) | `data/raw/mdt25/manual/*.tif` | 20% (detalle, no bloquea) |
| AEMET obs horaria | 🟡 pendiente ejecución | última 24h | estación + [ta,hr,vv,prec] | requiere `AEMET_API_KEY` en `.env` | script listo |
| AEMET clima diaria | 🟡 pendiente ejecución | 2017→hoy | estación/día (para FWI/DC/DMC) | requiere key | script listo |
| ERA5-Land (CDS) | 🟡 pendiente | 2017–2024 | 0.1° SWC/T2m/Td2m | requiere `CDS_API_KEY` o ~/.cdsapirc | script listo |
| GEE Sentinel-2/AlphaEarth | 🟡 pendiente | 2017–2024 | H3 res9 pentadual/annual | requiere `GEE_PROJECT` + init GEE | script TODO (F1) |

[^1]: versión/train window consolidada para el MVP: **train 2017–2022, test 2023–2024**.

## QA que se cierra con el boletín de parseo EGIF (`data/interim/egif/qa_egif.json`)

- 164.620 partes totales, **83.2% con coords decimales dentro del bbox de Galicia**
- 100% con fecha de `deteccion`; **100% con causa (`idcausa`) informada**
- Ventana de entrenamiento coherente: disponemos de **≥ 23k igniciones localizadas 2017–2023**

## Criterio para pasar a Fase 1 (features)

Solo quedan AEMET (key), CDS (key) y el script GEE; ninguno bloquea la tabla maestra de
estáticas + etiquetas de entrenamiento porque:
- la meteorología histórica de entrenamiento se cubre con el backfill de climatologías
  diarias de AEMET (2017–2023, serie larga por estación);
- los embeddings/NDMI (pendientes de GEE) se integran como `missing` en la Fase 2 y se
  completan al finalizar el batch en GEE.
