"""Features humanas WUI desde OSM (Gelabert 2025: distancia a WUI = driver
humano principal tras la anomalía de humedad). Proxy sin datos de población:
  - bldg_density_1km: nº de edificios OSM en radio 1 km de la celda
  - dist_settlement_m: distancia al edificio más cercano (WUI = borde
    vivienda-combustible; log para la escala)

Uso: uv run python -m ardintra.features.wui_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PBF = Path("data/raw/osm/galicia-2026-10-28.osm.pbf")
CELLS = Path("data/processed/h3_cells/res9_static.parquet")
OUT = Path("data/processed/h3_cells/res9_wui.parquet")


def main():
    import pyrosm

    print("extrayendo edificios del PBF (puede tardar varios minutos)...", flush=True)
    osm = pyrosm.OSM(str(PBF))
    b = osm.get_buildings()
    print(f"edificios: {len(b):,}", flush=True)
    # centroides de polígonos (proyección métrica para precisión razonable)
    b = b.to_crs(25829)  # ETRS89 / UTM29N (Galicia)
    cen = np.c_[b.geometry.centroid.x.to_numpy(), b.geometry.centroid.y.to_numpy()]
    np.save("/tmp/opencode/osm_bldg_centroids_25829.npy", cen)

    cells = pd.read_parquet(CELLS)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25829", always_xy=True)
    cx, cy = tr.transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
    tree = cKDTree(cen)
    # densidad 1 km
    cnt1 = np.array([len(h) for h in tree.query_ball_point(np.c_[cx, cy], 1000.0, workers=-1)])
    # distancia al edificio más cercano
    dist, _ = tree.query(np.c_[cx, cy], k=1, workers=-1)
    out = pd.DataFrame({
        "h9": cells["h9"],
        "bldg_density_1km": cnt1,
        "dist_settlement_m": dist.round(1),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} · densidad p50={np.median(cnt1):.0f} p95={np.quantile(cnt1, .95):.0f} · "
          f"dist p50={np.median(dist):.0f} m")


if __name__ == "__main__":
    main()
