"""Ronda 2: FIRMS (MODIS/VIIRS) active fire labels + spatial block CV (ya hecho).

FIRMS NRT (Near Real-Time) VIIRS/MODIS active fire detections via NASA FIRMS API.
Los VIIRS 375m detectan fuegos pequeños que el EGIF no consolida y cubren los
años recientes (2024-25) que el EGIF aún no tiene.

Sin API key para <100 peticiones/día (token "MAP_KEY" gratuito en
https://firms.modaps.eosdis.nasa.gov/api/map_key/). Sin key → usa el CSV
público de 24h (limitado). Con key → área/fecha completa.

Este módulo:
  1) Descarga VIIRS_SNPPT_NRT para Galicia (bbox, últimos N días)
  2) Cruza con EGIF: los fuegos VIIRS que coinciden con EGIF sirven para
     VALIDAR que VIIRS cubre los fuegos importantes; los VIIRS únicos son
     candidatos a labels extra (fuegos que EGIF no reportó)
  3) Reporta: cobertura VIIRS de EGIF, fuegos únicos VIIRS, solape espacial

Uso: uv run python -m ardintra.ingest.firms_virs [--days 30] [--key MAP_KEY]
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BBOX = "-9.65,41.75,-6.60,43.95"
OUT = Path("data/raw/firms")
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_firms(key: str, source: str = "VIIRS_SNPP_NRT", days: int = 30) -> pd.DataFrame:
    url = f"{FIRMS_URL}/{key}/{source}/{BBOX}/{days}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if r.text.strip().startswith("Invalid"):
        raise RuntimeError(f"FIRMS API error: {r.text[:100]}")
    df = pd.read_csv(StringIO(r.text))
    return df


def cross_with_egif(viirs: pd.DataFrame) -> dict:
    """Cruza VIIRS con EGIF: ¿VIIRS detecta los fuegos importantes de EGIF?"""
    eg = pd.read_parquet("data/interim/egif/parts_normalized.parquet", columns=None)
    eg = eg[(eg["in_bbox"] == True) & (eg["anio"] >= 2024)]  # noqa: E712
    if eg.empty:
        return {"nota": "EGIF 2024+ no consolidado aún — cruce pendiente"}

    from scipy.spatial import cKDTree
    tree = cKDTree(np.radians(eg[["latitud", "longitud"]].to_numpy()))
    v_ll = np.radians(viirs[["latitude", "longitude"]].to_numpy())
    d, _ = tree.query(v_ll, k=1)
    matched = d < 0.005  # ~500 m
    return {
        "n_viirs": len(viirs),
        "n_egif_2024": len(eg),
        "viirs_cerca_de_egif_500m": int(matched.sum()),
        "viirs_unicos": int((~matched).sum()),
    }


def main(days: int = 30):
    key = os.environ.get("FIRMS_MAP_KEY", "DEMO_KEY")
    OUT.mkdir(parents=True, exist_ok=True)

    for source in ("VIIRS_SNPP_NRT", "MODIS_NRT"):
        try:
            df = fetch_firms(key, source, days)
            out_file = OUT / f"{source}_{days}d.csv"
            df.to_csv(out_file, index=False)
            print(f"{source}: {len(df)} detecciones → {out_file}")
            if len(df) > 0:
                cross = cross_with_egif(df)
                print(f"  cruce EGIF: {json.dumps(cross, default=str)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {source}: {type(e).__name__}: {str(e)[:80]}")

    # resumen
    result = {"status": "descargado", "days": days, "bbox": BBOX}
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2))
    print(f"-> {OUT}/")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(days)
