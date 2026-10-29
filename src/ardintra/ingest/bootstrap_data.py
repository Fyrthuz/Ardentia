"""Orquestador de ingesta Fase 0/1 (idempotente: cada fuente aprovecha su manifest).

Uso:
  uv run python -m ardintra.ingest.bootstrap_data --sources egif,effis,dem,osm,mfe25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ardintra.ingest.common.manifest import build_manifest, load_manifest

PYTHON = sys.executable

PIPELINES: dict[str, tuple[str, str]] = {
    "egif": ("ardintra.ingest.egif.parse_egif", "EGIF XML → parts_normalized + QA"),
    "effis": ("ardintra.ingest.effis_wfs", "EFFIS RDII WFS 2017–2025"),
    "dem": ("ardintra.ingest.dem_glo30", "DEM GLO-30 mosaic/slope/aspect"),
    "osm": ("ardintra.ingest.osm_roads", "OSM pbf → roads parquet"),
    "mfe25": ("ardintra.ingest.mfe25", "MFE25 rar → SHP + fuel parquet"),
    "aemet": ("ardintra.ingest.aemet.aemet_obs", "AEMET obs + backfill (requiere AEMET_API_KEY)"),
    "era5": ("ardintra.ingest.era5.era5land_cds", "ERA5-Land netCDF (requiere CDS_API_KEY)"),
}

MANUAL_RAW: dict[str, dict[str, str]] = {
    "data/raw/egif/manual": {
        "license": "EGIF MITECO — público con atribución",
        "description": "Export manual del Buscador EGIF (zips Xml_/Xlsx_), snapshot 2026-10-29",
        "coverage": "partes de incendio EGIF, export manual usuario",
    },
    "data/raw/mdt25/manual": {
        "license": "IGN/CNIG — licencia abierta con atribución",
        "description": "Hojas MDT25 ETRS89-H29 seleccionadas manualmente",
        "coverage": "2 hojas (precisión de respaldo; la base del MVP es GLO-30)",
    },
    "data/raw/mfe25": {
        "license": "MITECO — uso público con atribución",
        "description": "Mapa Forestal de España 1:25.000 (Galicia), rar",
        "coverage": "Galicia",
    },
    "data/raw/osm": {
        "license": "ODbL © OpenStreetMap contributors",
        "description": "Extract Geofabrik Galicia, snapshot 2026-10-28",
        "coverage": "Galicia",
    },
}


def ensure_manual_manifests() -> None:
    for base_s, meta in MANUAL_RAW.items():
        base = Path(base_s)
        if not base.exists() or not any(p.is_file() for p in base.rglob("*")):
            continue
        if load_manifest(base) is None:
            build_manifest(
                source=base.name,
                out_dir=base,
                coverage={"nota": meta["coverage"]},
                license=meta["license"],
                description=meta["description"],
            )


def run(sources: list[str]) -> dict[str, str]:
    ensure_manual_manifests()
    status: dict[str, str] = {}
    for src in sources:
        if src not in PIPELINES:
            status[src] = "fuente desconocida"
            continue
        mod, desc = PIPELINES[src]
        print(f"\n=== {src}: {desc}")
        try:
            subprocess.run([PYTHON, "-m", mod], check=True, timeout=7200)
            status[src] = "OK"
        except subprocess.TimeoutExpired:
            status[src] = "TIMEOUT (7200s)"
        except subprocess.CalledProcessError as e:
            status[src] = f"FAIL (exit {e.returncode})"
    print("\n== RESUMEN ==")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources", type=str, default="egif,effis,dem,osm,mfe25",
        help="lista separada por comas: egif,effis,dem,osm,mfe25,aemet,era5",
    )
    args = ap.parse_args()
    run([s.strip() for s in args.sources.split(",") if s.strip()])
