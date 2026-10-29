# Fuentes de datos — acceso, credenciales, límites y quirks

> Checklist actualizada al 2026-10-29 en `data/catalog/availability_matrix.md`.

## Resumen de acceso

| Fuente | Auth | Límite crítico | Uso en el sistema |
|---|---|---|---|
| EGIF (Buscador MITECO) | manual (sin API pública) | export por tramos; XML→partes | etiquetas con causa (idcausa/motivación/causante), fecha `deteccion`, coords |
| EFFIS RDII WFS | **ninguna** | WFS 1.0.0 (1.1.0 invierte lon/lat); latencias de 2 min por año | perímetros ≥30 ha 2017–2025 |
| AEMET OpenData | API key (caduca 90 d) | **rate limit agresivo** (~15k req congelan la cuenta; ver nota) | tiempo real: obs horaria /observacion/convencional/todas |
| Open-Meteo Archive | **ninguna** | 429 por minuto y **por HORARIO** con payloads grandes | backfill histórico diario (58 estaciones; 1 req = 3.589 días × 5 var) |
| Copernicus CDS (ERA5-Land) | cuenta + **aceptación expresa de licencia del dataset** en el portal | cola de ejecución | SWC l1/l2, T2m, Td2m (VPD) 2017–2024 |
| **STAC earth-search + AWS sentinel-cogs** | **ninguna** (open data) | COGs con overviews; lecturas @100m con GDAL cache capped | Sentinel-2 L2A: NDMI/NDVI/NBR composites verano 2017–2024 (10 tiles MGRS; 30TTM en zone 30 → reproyectar) |
| ~~GEE / AlphaEarth~~ | ~~OAuth~~ | API key `AIza…` **no soportada** por EE (401 CREDENTIALS_MISSING) | **CANCELADO** (decisión 2026-10-29); módulo `gee_export.py` parked |
| ~~Prithvi / Clay local~~ | pesos HF | GPU RTX 5080 disponible pero no necesario | **CANCELADO** — la ablation F2 no usa embeddings |
| OSM Geofabrik | ninguna | — | red viaria (482.881 segmentos Galicia incl. 114k pistas forestales) |
| MFE25 (MITECO) | descarga manual | ZIP (no rar) | ModeloComb Rothermel 1–13 por polígono, FCC, especies |
| IGN/CNIG | descarga manual | hojas individuales | MDT25 (respaldo), BCN200 futura comparación viaria |
| Copernicus GLO-30 | **ninguna** (AWS open data) | — | DEM/pendiente/orientación (base del MVP) |

## Quirks descubiertos (para no tropezar dos veces)

1. **AEMET codificação e bytes mixtos**: el inventario de estaciones se sirve en latin-1
   (0xD3 = "Ó"). Patrón 2 pasos: el primer JSON devuelve la URL real (`datos`);
   el cuerpo puede volver **vacío** en tramos sin datos → tratar como vacío, no error.
2. **AEMET climatologías diarias**: fechas en formato `AAAA-MM-DDTHH:MM:SSUTC`
   (no vale ISO fecha simple). Datos con decimales por coma (`6,1`).
   **Límite documentado (FAQ oficial §3.5): 40 conexiones/min por usuario**; el código
   429 significa "sobrepasa los límites (caudal por minuto)". El backfill usa 2.5 s/req
   (~24/min) y ante 429 repite el MISMO tramo tras cooldown (no descarta datos).
   Chunk máximo probado: 31 días (30 con margen). El ACK y el body `datos` a veces
   llegan mal codificados → decodificar tolerante (latin-1 fallback) y descartar tramo.
3. **AEMET rate limit**: la API congela tras ~14–15k peticiones en una sesión
   incl. backfill por 30 días×2 llamadas×58 estaciones ≈ 14k → **usar Open-Meteo
   para histórico** y AEMET solo para el cron horario del tiempo real.
4. **Open-Meteo free tier**: 429 "Minutely" esperable; **"Hourly" también existe** —
   el módulo guarda checkpoint por estación (`per_station/`) y es reanudable.
5. **EFFIS WFS**: usar `WFS 1.0.0` (ejes lon,lat correctos); filtrar `COUNTRY=ES`
   y provincia en `{A Coruña, Lugo, Ourense, Pontevedra}` tras pedir por bbox
   (la bbox arrauga Portugal).
6. **GLO-30 es EPSG:4326** (grados): la pendiente requiere convertir el gradiente por
   píxel a metros (`a_deg·111_320·cos(lat)`) — primera versión con la unidad en grados
   daba pendientes ~90° (ya corregida: p95=28° para Galicia).
7. **MFE25**: el "rar" oficial es un **ZIP** (verificar `PK` antes de unrar);
   camada única para toda Galicia, CRS EPSG:25830 → reproyectar al grid GLO-30 4326.
8. **STAC S2 (earth-search)**: `grid:code` por tile MGRS, `eo:cloud_cover`. Bandas por
   nombre STAC (`nir`, `nir08`/B8A, `swir16`/B11, `swir22`/B12, `scl`). El COG trae
   `raster:bands` scale/offset (0.0001 / -0.1) que hay que aplicar ANTES de los
   normalizedDifference. Máscara SCL: filtrar 3/8/9/10/11. GDAL_CACHEMAX capado a
   512 MB por proceso de lectura para no reventar la RAM con decimaciones COG.
9. **EGIF**: `numeroparte` es la clave de negocio; `idpif` para joins internos.
   La `hora` en `pif_condiciones` a veces vale 2025 → ignorar como hora del parte.
9. **H3 res9 con un solo tipo de geometría**: para distancias viarias usar el
   **polígono de la celda** (`h3.cell_to_boundary`), no el centroide — los centroides
   solos cuelgan a ~50 m del arco y falsean `dist_to_road=9999`.
