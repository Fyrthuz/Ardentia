"""Etiquetas v1: igniciones EGIF (punto+fecha+causa) y perímetros EFFIS -> celdas H3 res9.

Salidas (data/processed/labels/v1/):
  ignitions.parquet      — una fila por celda×día con fuente(s) de ignición
  burned_cells.parquet   — una fila por celda×día+fuente EFFIS (perímeous >=30 ha)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd

from ardintra.ingest.common.galicia import CRS_WGS84

LABELS_DIR = Path("data/processed/labels/v1")
EGIF = Path("data/interim/egif/parts_normalized.parquet")
EFFIS = Path("data/raw/effis")

GALICIA_PROVINCES = {"A CORUÑA", "LUGO", "OURENSE", "PONTEVEDRA"}


import os

MIN_HA = float(os.environ.get("ARDENTIA_MIN_HA", "0.0"))


def egif_ignitions() -> pd.DataFrame:
    df = pd.read_parquet(EGIF)
    df = df[df["in_bbox"] == True].copy()  # noqa: E712
    if MIN_HA > 0:
        # igniciones "reales": quema superficie medible (los micro-fogatas <0.1 ha solo
        # representan el 0.09% del área quemada y son ruido de etiqueta)
        df = df[df["superficie_quemada_ha"] >= MIN_HA]
        print(f'filtro superficie ≥{MIN_HA} ha: {len(df):,} partes')
    df = df[(df["anio"] >= 2017) & df["deteccion"].notna()]
    df = df[df["latitud"].notna() & df["longitud"].notna()]
    df["h9"] = [
        h3.latlng_to_cell(la, lo, 9)
        for la, lo in zip(df["latitud"], df["longitud"], strict=True)
    ]
    out = pd.DataFrame({
        "h9": df["h9"],
        "fecha": df["deteccion"].dt.normalize(),
        "fire_id": df["numeroparte"],
        "source": "egif",
        "surface_ha": df["superficie_quemada_ha"],
        "cause": df["idcausa"].astype("Int16"),
        "motivation": df["idmotivacion"].astype("Int16"),
        "provincia": df["provincia"],
    })
    # colisiones celda×día: superficie sumada, causa/es seleccionamos la mayoritaria
    agg = (out.groupby(["h9", "fecha"], as_index=False).agg({
        "fire_id": list,
        "source": "first",
        "surface_ha": "sum",
        "cause": lambda s: s.mode().iat[0] if len(s.mode()) else pd.NA,
        "motivation": lambda s: s.mode().iat[0] if len(s.mode()) else pd.NA,
        "provincia": "first",
    }))
    return agg


def effis_burned() -> pd.DataFrame:
    parts = []
    for pq in sorted(EFFIS.glob("year=*/perimeters.parquet")):
        gdf = gpd.read_parquet(pq).to_crs(CRS_WGS84)
        gdf = gdf[gdf["COUNTRY"].eq("ES") & gdf["PROVINCE"].str.upper().isin(GALICIA_PROVINCES)]
        gdf["AREA_HA"] = pd.to_numeric(gdf["AREA_HA"], errors="coerce")
        gdf["fecha"] = pd.to_datetime(
            gdf["FIREDATE"].astype(str), format="mixed", errors="coerce"
        ).dt.normalize()
        for row in gdf.itertuples(index=False):
            geoms = getattr(row.geometry, "geoms", [row.geometry])
            cells: set[str] = set()
            for g in geoms:
                if g is None or g.is_empty:
                    continue
                try:
                    ring = list(g.exterior.coords)          # [(lon, lat), ...]
                    poly = h3.LatLngPoly([(lat, lon) for lon, lat in ring])
                    cells |= h3.polygon_to_cells(poly, 9)
                except Exception:  # noqa: BLE001
                    continue
            if not cells and row.AREA_HA >= 5:
                # perímetro pequeño (<ré un cell res9) pero relevante: celda del centroide
                rp = row.geometry.representative_point()
                cells = {h3.latlng_to_cell(rp.y, rp.x, 9)}
            for c in cells:
                parts.append({
                    "h9": c,
                    "fecha": row.fecha,
                    "fire_id": row.id,
                    "source": "effis",
                    "surface_ha": row.AREA_HA,
                    "provincia": row.PROVINCE,
                })
    return pd.DataFrame(parts) if parts else pd.DataFrame(
        columns=["h9", "fecha", "fire_id", "source", "surface_ha", "provincia"]
    )


def run() -> dict[str, int]:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    eg = egif_ignitions()
    ef = effis_burned()
    eg.to_parquet(LABELS_DIR / "ignitions.parquet", index=False)
    ef.to_parquet(LABELS_DIR / "burned_cells.parquet", index=False)
    qa = {
        "egif_cells": len(eg),
        "egif_days": eg["fecha"].nunique() if len(eg) else 0,
        "effis_burn_cells": len(ef),
        "effis_days": ef["fecha"].nunique() if len(ef) else 0,
    }
    print(qa)
    return qa


if __name__ == "__main__":
    run()
