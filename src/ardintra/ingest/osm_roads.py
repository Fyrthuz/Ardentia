"""Extracción de la red viaria de OSM (.osm.pbf) → roads_driveable.parquet.

Filtra las vías con tag highway de clases "driveable" (desde motorway hasta
track/path según configuración) y las proyecta a ETRS89/UTM29 (EPSG:25829).
Salida: data/interim/osm/roads_driveable.parquet (lineas + clase).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString

from ardintra.ingest.common.galicia import CRS_ETRS89_UTM29, CRS_WGS84
from ardintra.ingest.common.manifest import build_manifest

# Jerarquía de vía (clases de la var R del modelo). 'resta' minor: recorrido pista.
ROAD_CLASSES = {
    "motorway": "autovia",
    "motorway_link": "autovia",
    "trunk": "nacional",
    "trunk_link": "nacional",
    "primary": "nacional",
    "primary_link": "nacional",
    "secondary": "comarcal",
    "secondary_link": "comarcal",
    "tertiary": "comarcal",
    "tertiary_link": "comarcal",
    "unclassified": "pista",
    "residential": "local",
    "service": "local",
    "track": "pista_forestal",
    "path": "camino_rural",
    "footway": "camino_rural",
    "bridleway": "camino_rural",
}


class RoadHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, list[tuple[float, float]]]] = []

    def way(self, w: osmium.Way) -> None:
        highway = w.tags.get("highway")
        if highway not in ROAD_CLASSES:
            return
        coords: list[tuple[float, float]] = []
        for nd in w.nodes:
            if not nd.location.valid():
                continue
            coords.append((nd.location.lon, nd.location.lat))
        if len(coords) < 2:
            return
        self.rows.append((ROAD_CLASSES[highway], coords))


def run(pbf: Path | None = None, out: Path | None = None) -> dict[str, int]:
    pbf = pbf or Path("data/raw/osm/galicia-2026-10-28.osm.pbf")
    out_dir = out or Path("data/interim/osm")
    out_dir.mkdir(parents=True, exist_ok=True)

    handler = RoadHandler()
    handler.apply_file(str(pbf), locations=True)

    records = [
        {"road_class": cls, "geometry": LineString(coords)}
        for cls, coords in handler.rows
        if cls is not None
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_WGS84)
    gdf = gdf.to_crs(CRS_ETRS89_UTM29)
    gdf["length_m"] = gdf.length.round(1)
    out_parquet = out_dir / "roads_driveable.parquet"
    gdf.to_parquet(out_parquet, index=False)

    counts = gdf["road_class"].value_counts(dropna=False).to_dict()
    print(
        json.dumps({"n_vias": len(gdf), "por_clase": counts}, ensure_ascii=False, indent=2)
    )

    build_manifest(
        "osm_geofabrik",
        out_dir,
        coverage={"source_pbf": str(pbf), "n_vias": len(gdf), "por_clase": counts},
        license="ODbL © OpenStreetMap contributors",
        description="Red viaria OSM extraída a parquet con clases normalizadas.",
    )
    return counts


if __name__ == "__main__":
    run()
