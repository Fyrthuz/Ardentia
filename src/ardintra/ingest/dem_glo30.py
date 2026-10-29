"""Copernicus DEM GLO-30 (30m) — tiles de Galicia vía AWS open data (sin auth).

Descarga las celdas de 1x1 grados que cubren Galicia y genera un mosaico +
pendiente/orientación en data/interim/dem/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge as merge_rasters

from ardintra.ingest.common.galicia import CRS_WGS84
from ardintra.ingest.common.manifest import build_manifest

S3_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{folder}/{folder}.tif"

# Bins de 1 grado que cubren BBOX Galicia: lat 41_0??—43, lon W006—W010
LATS = ("N41", "N42", "N43")
LONS = ("W006", "W007", "W008", "W009", "W010")


def _tile_path(lat: str, lon: str) -> str:
    folder = f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"
    return S3_URL.format(folder=folder)


def download_tiles(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    with requests.Session() as s:
        for lat in LATS:
            for lon in LONS:
                dest = out_dir / f"{lat}_{lon}.tif"
                if dest.exists() and dest.stat().st_size > 1_000:
                    files.append(dest)
                    print(f"skip (ya existe) {dest.name}")
                    continue
                # fallback: bis de lat 41 no es 41.0 sino N41 -> folder N41_00 ok
                r = s.get(_tile_path(lat, lon), timeout=120)
                if r.status_code == 404:
                    continue  # tile costa/océano
                r.raise_for_status()
                dest.write_bytes(r.content)
                files.append(dest)
                print(f"ok {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return files


def build_mosaic(tiles: list[Path], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge_rasters(
        [r for r in srcs if r.width > 0], nodata=np.nan
    )
    profile = srcs[0].profile.copy()
    for s in srcs:
        s.close()
    mosaic = mosaic.astype("float32")
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    )
    out_dem = out_dir / "dem_galicia.tif"
    with rasterio.open(out_dem, "w", **profile) as dst:
        dst.write(mosaic)
    outputs["dem"] = out_dem

    # Pendiente (grados) y orientación (azimut 0..360, 0=N, +90=E)
    # GLO-30 en EPSG:4326: convertir deg->m por fila: X: a·111320·cos(lat), Y: e·110540
    elev = mosaic[0]
    nat = elev.astype("float64")
    dz_dy, dz_dx = np.gradient(nat)
    n_rows = nat.shape[0]
    lat_deg = (transform.f + transform.e * np.arange(n_rows)).astype("float64")
    scale_x = abs(transform.a) * 111_320.0 * np.cos(np.deg2rad(lat_deg))[:, None]  # m/px por fila
    scale_y = abs(transform.e) * 110_540.0  # m/px constante
    with np.errstate(invalid="ignore", divide="ignore"):
        gx = dz_dx / scale_x
        gy = dz_dy / scale_y
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    aspect = np.degrees(np.arctan2(gx, gy))  # dirección de descenso
    aspect = (180.0 + aspect) % 360.0        # orientación hacia fuera de la ladera
    slope = np.where(np.isnan(elev), np.nan, slope)
    aspect = np.where(np.isnan(elev), np.nan, aspect)
    profile.update(count=2, nodata=np.nan)
    out_slope = out_dir / "slope_galicia.tif"
    with rasterio.open(out_slope, "w", **{**profile, "count": 1}) as dst:
        dst.write(slope.astype("float32"), 1)
    out_aspect = out_dir / "aspect_galicia.tif"
    with rasterio.open(out_aspect, "w", **{**profile, "count": 1}) as dst:
        dst.write(np.where(np.isnan(elev), np.nan, (aspect + 360) % 360).astype("float32"), 1)
    outputs["slope"] = out_slope
    outputs["aspect"] = out_aspect
    return outputs


def run(base: Path | None = None) -> dict[str, Path]:
    base = base or Path("data/raw/dem_glo30")
    interim = Path("data/interim/dem")
    tiles = download_tiles(base)
    if not tiles:
        raise RuntimeError("No hay tiles del DEM en Galicia — check bucket.")
    o = build_mosaic(tiles, interim)
    build_manifest(
        "copernicus_glo30",
        base,
        coverage={"lat_bins": LATS, "lon_bins": LONS, "tiles_descargados": len(tiles)},
        license="Copernicus DEM — open license (ESA/Reg. 105/2013+2021/410)",
        description=f"DEM 30m Galicia; outputs: {[str(v) for v in o.values()]}",
    )
    # registro de outputs (not en raw) como catálogo
    (base / "outputs_catalog.json").write_text(
        json.dumps({k: str(v) for k, v in o.items()}, indent=2), encoding="utf-8"
    )
    return o


if __name__ == "__main__":
    run()
