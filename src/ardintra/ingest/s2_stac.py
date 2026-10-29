"""Composites Sentinel-2 L2A LOCALES vía STAC earth-search (AWS, sin auth).

Sustituye al bloque GEE (Fase 1.8 v1) por una vía 100% reproducible local:
- STAC search por tile MGRS y año (Jun 1 - Oct 1, nubes < 40%)
- lectura de bandas COG a 100 m con rasterio (resampling en la ventana HTTP)
- escala/offset del producto L2A aplicados (raster:bands)
- máscara SCL (3,8,9,10,11) y mediana temporal por píxel
- mosaico por año → data/raw/s2/s2_summer_{year}.tif (ndmi, ndvi, nbr @100m, EPSG:25829)

Las celdas H9 se muestrean después localmente (mismo sampler que DEM/MFE25).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import gc
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as merge_rasters
from rasterio.warp import reproject
from rasterio.warp import Resampling as RWarp

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BBOX = (-9.65, 41.75, -6.60, 43.95)
TILES = ["29TMG", "29TMH", "29TMJ", "29TNG", "29TNH", "29TNJ", "29TPG", "29TPH", "29TPJ", "30TTM"]
OUT_DIR = Path("data/raw/s2")
TMP_DIR = Path("data/interim/s2_tiles")
UTM29 = "EPSG:25829"
SCALE = 100.0
GDAL_ENV = {"GDAL_CACHEMAX": 512, "GDAL_NUM_THREADS": "1"}
SCL_BAD = (3, 8, 9, 10, 11)
YEARS = list(range(2017, 2025))


def _stac_search(tile: str, year: int, cloud_lt: int = 40, limit: int = 15) -> list[dict]:
    r = httpx.post(
        STAC_URL,
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": BBOX,
            "datetime": f"{year}-06-01T00:00:00Z/{year}-10-01T23:59:59Z",
            "query": {"grid:code": {"eq": f"MGRS-{tile}"}, "eo:cloud_cover": {"lt": cloud_lt}},
            "limit": limit,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("features", [])


def _scaled(read_arr: np.ndarray, src: rasterio.DatasetReader) -> np.ndarray:
    """Aplica scale/offset del COG y nodata 0 → NaN."""
    arr = read_arr.astype("float32")
    scale = src.scales[0] if src.scales else 1.0
    offset = src.offsets[0] if src.offsets else 0.0
    arr = arr * scale + offset
    arr[read_arr == 0] = np.nan
    return arr


def _value_arrays(scene: dict) -> tuple[dict[str, np.ndarray], rasterio.crs.CRS, object]:
    """Por escena: bandas → 100m, SCL mask, NDMI/NDVI/NBR normalizadas. Disco/GDAL cache capped."""
    assets = scene["assets"]

    def band(key: str, resamp: Resampling = Resampling.nearest, raw: bool = False):
        with rasterio.Env(**GDAL_ENV), rasterio.open(assets[key]["href"]) as src:
            ratio = src.res[0] / SCALE
            shape = (int(round(src.height * ratio)), int(round(src.width * ratio)))
            arr = src.read(1, out_shape=shape, resampling=resamp)
            # el transform debe escalar al tamaño decimado (10m→100m ×0.1): sin esto cada
            # tile "ocupa" 22 km en vez de 110 km y el mosaico queda 95% NaN
            tr_dec = src.transform * rasterio.Affine.scale(
                src.width / shape[1], src.height / shape[0]
            )
            if raw:
                return arr, src.crs, tr_dec
            return _scaled(arr, src), src.crs, tr_dec


    scl_raw, crs, trf = band("scl", raw=True)   # uint8 crudo: nada de cast NaN
    scl = scl_raw
    h, w = scl.shape

    def to_scl_shape(a: np.ndarray) -> np.ndarray:
        if a.shape == (h, w):
            return a
        vh = max(a.shape[0] // h, 1)
        vw = max(a.shape[1] // w, 1)
        return a[::vh, ::vw][:h, :w]

    red = to_scl_shape(band("red")[0])       # B04, 10m
    nir = to_scl_shape(band("nir")[0])       # B08, 10m
    nir8 = to_scl_shape(band("nir08")[0])    # B8A, 20m (mismo grid que scl)
    sw16 = to_scl_shape(band("swir16")[0])   # B11
    sw22 = to_scl_shape(band("swir22")[0])   # B12

    mask = ~np.isin(scl_raw.astype("uint8"), SCL_BAD)
    ndmi = (nir8 - sw16) / (nir8 + sw16)
    ndvi = (nir - red) / (nir + red)
    nbr = (nir8 - sw22) / (nir8 + sw22)
    for arr in (ndmi, ndvi, nbr):
        arr[~mask] = np.nan
    return {"ndmi": ndmi, "ndvi": ndvi, "nbr": nbr}, crs, trf


def _tile_year_median(tile: str, year: int, max_scenes: int = 8) -> dict | None:
    scenes = _stac_search(tile, year)
    if not scenes:
        print(f"  {tile} {year}: sin escenas")
        return None
    stacks = {"ndmi": [], "ndvi": [], "nbr": []}
    ref_crs, ref_tr, ref_shape = None, None, None
    used = 0
    for sc in scenes[:max_scenes]:
        try:
            out, crs, trf = _value_arrays(sc)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {sc['id'][:55]}: {type(e).__name__}: {str(e)[:60]}")
            continue
        if ref_crs is None:
            ref_crs, ref_tr, ref_shape = crs, trf, out["ndmi"].shape
        if out["ndmi"].shape != ref_shape:
            continue
        for k in ("ndmi", "ndvi", "nbr"):
            stacks[k].append(out[k])
        used += 1
    if used < 2:
        print(f"  {tile} {year}: solo {used} escenas válidas")
        return None
    med = {k: np.nanmedian(np.stack(v), axis=0).astype("float32") for k, v in stacks.items()}
    del stacks
    gc.collect()
    return {"bands": med, "crs": ref_crs, "transform": ref_tr, "shape": ref_shape}


def _to_utm29(med: dict) -> tuple[np.ndarray, rasterio.crs.CRS, object]:
    arr3 = np.stack([med["bands"]["ndmi"], med["bands"]["ndvi"], med["bands"]["nbr"]])
    crs, tr = med["crs"], med["transform"]
    if crs.to_epsg() == 25829:
        return arr3, crs, tr
    from pyproj import Transformer  # noqa: PLC0415

    tr4 = Transformer.from_crs(crs, UTM29, always_xy=True)
    h, w = arr3.shape[1:]
    xs, ys = tr4.transform(
        [tr.c, tr.c + tr.a * w],          # x: izquierda/derecha
        [tr.f, tr.f + tr.e * h],          # y: arriba/abajo
    )
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    out_tr = rasterio.Affine(SCALE, 0, x_min, 0, -SCALE, y_max)
    out_shape = (3, int(np.ceil((y_max - y_min) / SCALE)), int(np.ceil((x_max - x_min) / SCALE)))
    dst = np.full(out_shape, np.nan, dtype="float32")
    for b in range(3):
        reproject(
            arr3[b], dst[b],
            src_transform=tr, src_crs=crs,
            dst_transform=out_tr, dst_crs=UTM29,
            resampling=RWarp.average,
        )
    return dst, rasterio.crs.CRS.from_string(UTM29), out_tr


def _merge_tiles(items: list[tuple[np.ndarray, object, object]], year: int) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_files = []
    for i, (arr, crs, tr) in enumerate(items):
        p = TMP_DIR / f"tmp_{month_no(year)}_{i:02d}.tif".replace("{Y}", "")
        # nombre determinista por año
        # PID cuyo: dos instancias vivas no colisionan en los mismos ficheros tmp
        p = TMP_DIR / f"tmp_s2_{year}_{i:02d}_{os.getpid()}.tif"
        with rasterio.open(
            p, "w", driver="GTiff",
            height=arr.shape[1], width=arr.shape[2], count=3,
            dtype="float32", crs=crs, transform=tr, nodata=np.nan,
        ) as dst:
            dst.write(arr)
        tmp_files.append(p)
    srcs = [rasterio.open(p) for p in tmp_files]
    mosaic, tr2 = merge_rasters(srcs)
    for s in srcs:
        s.close()
    for p in tmp_files:
        p.unlink(missing_ok=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"s2_summer_{year}.tif"
    with rasterio.open(
        out, "w", driver="GTiff",
        height=mosaic.shape[1], width=mosaic.shape[2], count=3,
        dtype="float32", crs=UTM29, transform=tr2, nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(mosaic)
        dst.set_band_description(1, "ndmi")
        dst.set_band_description(2, "ndvi")
        dst.set_band_description(3, "nbr")
    return out


def month_no(year: int) -> int:  # helper pegado para nombres tmp únicos
    return year % 100


def run_year(year: int) -> Path:
    out = OUT_DIR / f"s2_summer_{year}.tif"
    if out.exists():
        print(f"skip {out.name} (existe)")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    def _one(tile):
        try:
            m = _tile_year_median(tile, year)
            return (tile, _to_utm29(m)) if m else (tile, None)
        except Exception:  # noqa: BLE001
            return (tile, None)

    items = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for tile, m in ex.map(_one, TILES):
            if m is not None:
                items.append(m)
    if not items:
        raise RuntimeError(f"{year}: ningún tile con datos")
    path = _merge_tiles(items, year)
    print(f"-> {path}")
    return path


def run_all(years: list[int] | None = None) -> list[Path]:
    return [run_year(y) for y in (years or YEARS)]


if __name__ == "__main__":
    import sys

    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    run_all(args or None)
