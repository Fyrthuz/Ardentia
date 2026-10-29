"""Tabla maestra estática: celdas H3 res9 de Galicia con suelo forestal.

Por celda: lat/lon, elev/slope/aspect (GLO-30 4326), fracción forestal y modelo
Rothermel mayoritario (MFE25 n_MODCOM), dist_to_road, road_class,
road_density_1km y road_fuel_interface (vía cuyo arcén toca combustible).

Salida: data/processed/h3_cells/res9_static.parquet
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import STRtree, distance as shp_distance, intersection, length as shp_length
from shapely.geometry import box

from ardintra.ingest.common.galicia import BBOX_GALICIA, CRS_ETRS89_UTM29, CRS_WGS84

DEM = Path("data/interim/dem/dem_galicia.tif")
SLOPE = Path("data/interim/dem/slope_galicia.tif")
ASPECT = Path("data/interim/dem/aspect_galicia.tif")
MFE = Path("data/raw/mfe25/unpacked/MFE_11.shp")
FUEL = Path("data/interim/mfe25/fuel_modcom_30m.tif")
ROADS = Path("data/interim/osm/roads_driveable.parquet")
OUT = Path("data/processed/h3_cells/res9_static.parquet")

STRIDE = 2          # muestreo del grid a ~60 m para zonal por celda
CHUNK_CELLS = 20_000


def build_fuel_raster() -> Path:
    """Rasteriza n_MODCOM del MFE25 (1-13) al grid del DEM (CRS 4326)."""
    if FUEL.exists():
        return FUEL
    FUEL.parent.mkdir(parents=True, exist_ok=True)
    mfe = gpd.read_file(MFE, columns=["n_MODCOM", "geometry"])
    with rasterio.open(DEM) as dem:
        transform, height, width = dem.transform, dem.height, dem.width
        crs = dem.crs
    mfe = mfe.to_crs(crs)
    mfe = mfe[mfe["n_MODCOM"].fillna(0) > 0]
    fuel = rasterize(
        ((geom, int(mod)) for geom, mod in zip(mfe.geometry, mfe["n_MODCOM"], strict=True)),
        out_shape=(height, width),
        transform=transform,
        fill=255,
        dtype="uint8",
    )
    with rasterio.open(FUEL, "w", driver="GTiff", height=height, width=width, count=1,
                       transform=transform, crs=crs, dtype="uint8", nodata=255,
                       compress="deflate") as dst:
        dst.write(fuel, 1)
    print(f"fuel raster {FUEL}: {np.mean(fuel != 255):.1%} píxeles con modelo")
    return FUEL


def h3_cells_forest() -> pd.DataFrame:
    """Celdas res9 con suelo; fracción forestal y modelo mayoritario vía muestreo 60 m."""
    with rasterio.open(DEM) as dem:
        elev = dem.read(1)
        land = np.isfinite(elev)
        tr, crs, (H, W) = dem.transform, dem.crs, (dem.height, dem.width)
    with rasterio.open(FUEL) as fsrc:
        fuel = fsrc.read(1)

    rows_land, cols_land = np.where(land)
    rows_s, cols_s = rows_land[::STRIDE], cols_land[::STRIDE]
    print(f"píxeles de tierra muestreados ({STRIDE}x stride): {rows_s.size:,}")

    xs = tr.c + (cols_s + 0.5) * tr.a
    ys = tr.f + (rows_s + 0.5) * tr.e  # tr.e < 0
    fvals = fuel[rows_s, cols_s]

    idxs = [h3.latlng_to_cell(y, x, 9) for y, x in zip(ys, xs, strict=True)]
    df_s = pd.DataFrame({
        "h9": idxs,
        "is_fuel": np.asarray(fvals != 255, dtype=np.int8),
        "fuel_val": np.where(fvals != 255, np.asarray(fvals, dtype=np.int16), np.nan),
    })
    del idxs, xs, ys, fvals, rows_s, cols_s, rows_land, cols_land

    grp = df_s.groupby("h9", sort=True).agg(
        land_px=("is_fuel", "size"),
        fuel_px=("is_fuel", "sum"),
        fuel_model=("fuel_val", "median"),
    ).reset_index()
    del df_s
    grp = grp[grp["land_px"] >= 5]  # descarte de ruido en bordes/Océano
    grp["forest_frac"] = (grp["fuel_px"] / grp["land_px"]).round(4)
    # solo celdas con presencia real de combustible forestal (dominio del modelo)
    grp = grp[(grp["fuel_px"] >= 3) | (grp["forest_frac"] >= 0.05)].reset_index(drop=True)
    grp["fuel_model"] = grp["fuel_model"].round().astype("Int16")

    pts = [h3.cell_to_latLng(i) if hasattr(h3, "cell_to_latLng") else h3.cell_to_latlng(i)
           for i in grp["h9"]]
    grp["lat"] = [p[0] for p in pts]
    grp["lon"] = [p[1] for p in pts]
    return grp[["h9", "lat", "lon", "forest_frac", "fuel_model", "fuel_px", "land_px"]]


def _sample_raster(path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.crs != CRS_WGS84:
            raise RuntimeError(f"{path} debe estar en 4326 (es {src.crs})")
        pts = [(lo, la) for la, lo in zip(lats, lons, strict=True)]
        vals = np.asarray([s[0] for s in src.sample(pts)], dtype="float32")
    return vals


def _cell_polygons(h9: pd.Series) -> gpd.GeoSeries:
    """Polígono WGS84 de cada celda H3 (para distancia/intersección correctamente)."""
    import shapely  # noqa: PLC0415

    from shapely.geometry import Polygon  # noqa: PLC0415
    polys = []
    for idx in h9:
        ring = h3.cell_to_boundary(idx)  # [(lat, lng), ...] orientado CCW
        polys.append(Polygon([(lng, lat) for lat, lng in ring]))
    return gpd.GeoSeries(polys, crs=CRS_WGS84, index=h9.index)


def _add_road_feats(cells: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """dist_to_road, road_class, road_fuel_interface y road_density_1km."""
    # road_fuel por vía: arcén (buffer 20 m) toca combustible -> muestreo del midpoint
    road_fuel = np.zeros(len(roads), dtype=np.int8)
    mids = roads.geometry.interpolate(0.5, normalized=True)
    with rasterio.open(FUEL) as fsrc:
        for b in range(0, len(roads), 100_000):
            mid = mids.iloc[b : b + 100_000]
            vals = np.asarray([s[0] for s in fsrc.sample([(pt.x, pt.y) for pt in mid])])
            road_fuel[b : b + 100_000] = (vals != 255).astype(np.int8)
    roads = roads.assign(road_fuel=road_fuel)
    print(f"vías con combustible en arcén: {road_fuel.mean():.1%}")

    road_geoms = roads.geometry.to_numpy()
    rc = roads["road_class"].to_numpy().astype(object)
    rfu = roads["road_fuel"].to_numpy()
    tree = STRtree(road_geoms)
    cell_geoms = cells.geometry.to_numpy()

    n = len(cells)
    dist = np.empty(n, dtype=np.float32)
    klass = np.empty(n, dtype=object)
    iface = np.zeros(n, dtype=np.int8)
    dens = np.zeros(n, dtype=np.float32)

    for i0 in range(0, n, CHUNK_CELLS):
        cg = cell_geoms[i0 : i0 + CHUNK_CELLS]
        # vía más cercana (vectorizado a nivel de chunk)
        nearest_idx = tree.nearest(cg)
        dist[i0 : i0 + CHUNK_CELLS] = shp_distance(road_geoms[nearest_idx], cg)
        klass[i0 : i0 + CHUNK_CELLS] = rc[nearest_idx]
        iface[i0 : i0 + CHUNK_CELLS] = rfu[nearest_idx]

        # densidad: buffer 1 km alrededor del centróide de la celda
        centroids = cells.geometry[i0 : i0 + CHUNK_CELLS].centroid
        buffers = centroids.buffer(1000.0).to_numpy()
        r = tree.query(buffers, predicate="intersects")  # 2 x K
        if r.size:
            clipped = intersection(road_geoms[r[1]], buffers[r[0]])
            lens = shp_length(clipped)
            acc = np.zeros(len(buffers), dtype=np.float64)
            np.add.at(acc, r[0], lens)
            # km lineales por km² del buffer circular (área π·1km²)
            dens[i0 : i0 + CHUNK_CELLS] = acc / 1000.0 / np.pi
        print(f"   roads {min(i0 + CHUNK_CELLS, n):,}/{n:,}")
    cells["dist_to_road"] = dist
    cells["road_class"] = klass
    cells["road_fuel_interface"] = iface
    cells["road_density_1km"] = dens.round(4)
    return cells


def run() -> pd.DataFrame:
    build_fuel_raster()
    cells = h3_cells_forest()
    print(f"celdas res9: {len(cells):,}")

    cells["elev_m"] = _sample_raster(DEM, cells["lat"].to_numpy(), cells["lon"].to_numpy())
    cells["slope_deg"] = _sample_raster(SLOPE, cells["lat"].to_numpy(), cells["lon"].to_numpy())
    cells["aspect_deg"] = _sample_raster(ASPECT, cells["lat"].to_numpy(), cells["lon"].to_numpy())

    # polígono real de la celda H3 (no el centroide) para joins viarios correctos
    gdf = gpd.GeoDataFrame(cells, geometry=_cell_polygons(cells["h9"]), crs=CRS_WGS84).to_crs(
        CRS_ETRS89_UTM29
    )

    roads = gpd.read_parquet(ROADS).to_crs(CRS_ETRS89_UTM29)
    gdf = _add_road_feats(gdf, roads)

    out_df = gdf.drop(columns="geometry").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out_df):,} celdas)")
    print(out_df[["dist_to_road", "road_density_1km", "slope_deg", "forest_frac"]].describe().to_string())
    return out_df


if __name__ == "__main__":
    run()
