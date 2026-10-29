"""MFE25 (Mapa Forestal 1:25.000): extracción del rar y resumen de capas.

El rar oficial trae SHP (o GDB) por provincia. Se extrae a data/raw/mfe25/unpacked
y se consolida un parquet de polígonos por clase de uso forestal para
rasterizar después a la máscara de combustible 10m.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ardintra.ingest.common.manifest import build_manifest

RAW = Path("data/raw/mfe25")
UNPACKED = RAW / "unpacked"
SHP_CANDIDATES = ("*.shp",)


def unpack() -> list[Path]:
    UNPACKED.mkdir(parents=True, exist_ok=True)
    rar_file = next(RAW.glob("*.rar"), None)
    if rar_file is None:
        raise FileNotFoundError("No hay *.rar en data/raw/mfe25")
    if not any(UNPACKED.iterdir()):
        # El fichero lo llaman .rar pero suele ser ZIP (verificado Dic2021)
        if rar_file.read_bytes()[:2] == b"PK":
            import zipfile

            with zipfile.ZipFile(rar_file) as zf:
                zf.extractall(UNPACKED)
        else:
            subprocess.run(["unrar", "x", "-y", str(rar_file), str(UNPACKED)], check=True)
    return [p for pat in SHP_CANDIDATES for p in UNPACKED.rglob(pat)]


def summarize(shps: list[Path]) -> pd.DataFrame:
    rows = []
    for shp in shps:
        try:
            gdf = gpd.read_file(shp, include_fields=None)  # schema completo
        except Exception as e:  # noqa: BLE001
            print(f"  ! {shp}: {e}")
            continue
        rows.append({
            "shapefile": shp.name,
            "crs": str(gdf.crs.to_string() if gdf.crs else None),
            "n_features": len(gdf),
            "columns": [c.lower() for c in gdf.columns][:40],
        })
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    shps = unpack()
    print(f"shapes: {[s.name for s in shps]}")
    df = summarize(shps)
    out = UNPACKED / "mfe25_summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(max_colwidth=60))
    build_manifest(
        "mfe25",
        RAW,
        coverage={"n_shapefiles": len(shps), "summary": str(out)},
        license="MITECO — uso público con atribución",
        description="MFE25 Galicia extraído del rar oficial",
    )
    return {"n_shapes": len(shps), "summary": str(out)}


if __name__ == "__main__":
    run()
