"""Exportaciones Google Earth Engine para las features satelitales (Fase 1.8).

Estrategia "raster-first": en vez de reducRegions sobre 245k celdas (servidor
colapsado), exportamos rasters Galicia a escala 100 m (coincidente con res9) y
el muestreo por celda se hace LOCAL con rasterio (como DEM/MFE25).

Jobs:
  A) S2 verano por año 2017-2024: ndmi/ndvi/nbr media Jun-Sep (3 bandas/año)
  B) AlphaEarth embeddings (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL):
     2023 y 2024 (64 bandas cada uno) + media 2018-2022 (64 bandas)

Requiere (una vez): `earthengine authenticate` (browser) y GEE_PROJECT en .env.
Uso:
  uv run python -m ardintra.ingest.gee.gee_export --probe     # comprueba acceso
  uv run python -m ardintra.ingest.gee.gee_export --launch   # lanza tasks
  uv run python -m ardintra.ingest.gee.gee_export --status   # task states
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import ee

from ardintra.ingest.common.galicia import BBOX_GALICIA, CRS_ETRS89_UTM29

DRIVE_FOLDER = "ardintra_gee"

S2 = "COPERNICUS/S2_SR_HARMONIZED"
EMBED = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

YEARS = list(range(2017, 2025))
FIRE_SEASON = [ee.DateRange(f"{y}-06-01", f"{y}-10-01") for y in YEARS]


def _env(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        for line in Path(".env").read_text().splitlines():
            if line.startswith(key):
                v = line.split("=", 1)[1].strip()
    if not v:
        raise RuntimeError(f"Falta {key} en .env")
    return v


def init_ee() -> None:
    project = _env("GEE_PROJECT")
    ee.Initialize(project=project, opt_url="https://earthengine.googleapis.com")


def _bbox_gee() -> ee.Geometry:
    lon0, lat0, lon1, lat1 = BBOX_GALICIA
    return ee.Geometry.Rectangle([lon0, lat0, lon1, lat1], proj="EPSG:4326", evenOdd=True)


def s2_summer_composite(year: int) -> ee.Image:
    """NDMI/NDVI/NBR medios del verano (Jun 1 - Oct 1) con máscara SCL."""
    start, end = ee.Date(f"{year}-06-01"), ee.Date(f"{year}-10-01")
    col = (
        ee.ImageCollection(S2)
        .filterBounds(_bbox_gee())
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(_mask_scl)
    )
    ndmi = col.map(lambda i: i.normalizedDifference(["B8A", "B11"]).rename("ndmi"))
    ndvi = col.map(lambda i: i.normalizedDifference(["B8", "B4"]).rename("ndvi"))
    nbr = col.map(lambda i: i.normalizedDifference(["B8A", "B12"]).rename("nbr"))
    return (
        ndmi.mean().addBands(ndvi.mean()).addBands(nbr.mean())
        .rename([f"ndmi_{year}", f"ndvi_{year}", f"nbr_{year}"])
        .unmask(-9999)
        .clip(_bbox_gee())
    )


def _mask_scl(img: ee.Image) -> ee.Image:
    qa = img.select("SCL")
    # 3 nube sombra, 8 nube media-prob, 9 nube alta-prob 10 cirrus, 11 nieve
    mask = qa.neq(3).And(qa.neq(8)).And(qa.neq(9)).And(qa.neq(10)).And(qa.neq(11))
    return img.updateMask(mask).divide(10000).copyProperties(img, img.propertyNames())


def embedding_image(years: range) -> ee.Image:
    """Embedding AlphaEarth MEDIA de un rango de años: 64 bandas embed_01..64."""
    col = ee.ImageCollection(EMBED).filterBounds(_bbox_gee()).filter(
        ee.Filter.inList("system:index", [f"{y}" for y in years])
    )
    bands = ee.List(col.first().bandNames())
    mean_img = col.mean()  # media inter-anual elemento a elemento

    def rename_band(_, old):
        n = ee.Number(old).add(1)
        return mean_img.select(old).rename(ee.String("embed_").cat(n.format("%02d")))

    seq = ee.List.sequence(0, 63)
    renamed = ee.ImageCollection.fromImages(seq.map(lambda i: mean_img.select(ee.Number(i)).rename(ee.String("embed_").cat(ee.Number(i).add(1).format("%02d")))))
    return renamed.toBands().unmask(-9999).clip(_bbox_gee())


def launch_all() -> list[dict]:
    init_ee()
    tasks: list[dict] = []
    region_m = ee.Geometry.Rectangle(
        [
            ee.Point(_bbox_gee().coordinates().get(0)).proj?? no usamos:
        ],
        proj=CRS_ETRS89_UTM29,
    )  # placeholder: la región en 25829 se calcula con .transform en el helper
    region = _bbox_gee().transform(CRS_ETRS89_UTM29, 100)

    def _export(img: ee.Image, desc: str, scale: int = 100) -> dict:
        t = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder=DRIVE_FOLDER,
            region=region,
            scale=scale,
            crs=CRS_ETRS89_UTM29,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        t.start()
        return {"task": desc, "id": t.id, "status": "STARTED"}

    for y in YEARS:
        tasks.append(_export(s2_summer_composite(y), f"s2_summer_{y}"))
    tasks.append(_export(embedding_image(range(2023, 2025)), "alphaearth_2023_2024", scale=250))
    tasks.append(_export(embedding_image(range(2018, 2023)), "alphaearth_2018_2022", scale=250))
    for t in tasks:
        print(t["task"], t["id"])
    return tasks


def status() -> list[dict]:
    init_ee()
    return [
        {"id": t.id, "desc": t.status().get("description"), "state": t.status().get("state")}
        for t in ee.batch.Task.list()[:20]
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.probe:
        init_ee()
        img = s2_summer_composite(2020)
        val = img.rename("ndmi_2020").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=_bbox_gee().centroid(1),
            scale=100,
            crs=CRS_ETRS89_UTM29,
        ).getInfo()
        print("probe ndmi_2020 en el centro de Galicia:", val)
        sys.exit(0)
    if args.launch:
        launch_all()
    if args.status:
        for row in status():
            print(row["state"].rjust(10), row["desc"] or row["id"])
