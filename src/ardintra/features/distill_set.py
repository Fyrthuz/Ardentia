"""Set de destilación: filas full-grid muestreadas del período de train.

A diferencia del dataset v1 (15 celdas/día), aquí muestreamos ~1.000 celdas
aleatorias por día en ~420 días de 2017-2022 → el student aprende la
DISTRIBUCIÓN REAL del grid completo (fix estructural de la calibración).

Features: las mismas 48 del pipeline (para que el student sea drop-in).

Uso: uv run python -m ardintra.features.distill_set
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CELLS = Path("data/processed/h3_cells/res9_static.parquet")
OUT = Path("data/processed/features/v1/distill_set.parquet")
DAYS_PER_YEAR = 70
CELLS_PER_DAY = 1000
SEED = 42

DYN_STATION = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
               "prec_mm", "spei30", "spei90", "ffmc_anom", "dmc_anom", "dc_anom",
               "fwi_7d", "fwi_prev", "ffmc_7d"]
ERA5_FILL = ["swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"]


def main():
    from ardintra.features.build_dataset import load_met_stations

    rng = np.random.default_rng(SEED)
    cells = pd.read_parquet(CELLS)
    cells["aspect_sin"] = np.sin(np.radians(cells["aspect_deg"]))
    cells["aspect_cos"] = np.cos(np.radians(cells["aspect_deg"]))
    cells["dist_to_road_log"] = np.log1p(cells["dist_to_road"].clip(0, 5000))
    cells["road_density_log"] = np.log1p(cells["road_density_1km"])
    cell_ll = np.radians(cells[["lat", "lon"]].to_numpy())

    met = load_met_stations()
    met["fecha_n"] = pd.to_datetime(met["fecha"]).dt.normalize()
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    st_tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    dd, ii = st_tree.query(cell_ll, k=3)
    w_idw = 1.0 / np.maximum(dd, 1e-5) ** 2
    w_idw = w_idw / w_idw.sum(axis=1, keepdims=True)
    st_ids = st_meta["idema"].to_numpy()

    piv = {}
    for var in DYN_STATION:
        piv[var] = met.pivot_table(index="fecha_n", columns="idema", values=var, aggfunc="first")

    # medianas ERA5 (mismo fill que fullgrid_eval)
    ds = pd.read_parquet("data/processed/features/v1/dataset.parquet", columns=ERA5_FILL)
    era5_med = {c: float(ds[c].median()) for c in ERA5_FILL}

    # KD tree de igniciones para recencia/lags vectorizados
    ig = pd.read_parquet("data/processed/labels/v1/ignitions.parquet")
    ig = ig.merge(cells[["h9", "lat", "lon"]], on="h9", how="left").dropna(subset=["lat", "lon"])
    lag_tree = cKDTree(np.radians(ig[["lat", "lon"]].to_numpy()))
    lag_dates = pd.to_datetime(ig["fecha"]).to_numpy().astype("datetime64[D]")
    ang5, ang25 = 5000.0 / 6_371_000.0, 2500.0 / 6_371_000.0

    # S2 cache por año
    s2_cache = {}

    # días muestreados
    days = []
    for y in range(2017, 2023):
        all_days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        days += list(rng.choice(all_days, size=DAYS_PER_YEAR, replace=False))
    print(f"días muestreados: {len(days)} × {CELLS_PER_DAY} celdas = {len(days) * CELLS_PER_DAY:,} filas", flush=True)

    rows = []
    for n_day, day in enumerate(days):
        cidx = rng.choice(len(cells), size=CELLS_PER_DAY, replace=False)
        d = cells.iloc[cidx].copy()
        day_ts = pd.Timestamp(day)
        # met IDW
        for var in DYN_STATION:
            p = piv[var]
            if day_ts not in p.index:
                d[var] = np.nan
                continue
            vals = p.loc[day_ts].reindex(st_ids).to_numpy()
            vv = vals[ii[cidx]]
            valid = ~np.isnan(vv)
            d[var] = np.nansum(np.where(valid, vv, 0) * np.where(valid, w_idw[cidx], 0), axis=1) / \
                np.maximum(np.where(valid, w_idw[cidx], 0).sum(axis=1), 1e-9)
        for c in ERA5_FILL:
            d[c] = era5_med[c]
        # S2 prev-summer
        year = day_ts.year
        if year - 1 not in s2_cache:
            import rasterio
            from pyproj import Transformer
            tif = Path(f"data/raw/s2/s2_summer_{year - 1}.tif")
            if tif.exists():
                with rasterio.open(tif) as src:
                    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    xs, ys = tr.transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
                    s2_cache[year - 1] = np.array(
                        [x for x in src.sample(list(zip(xs, ys)))], dtype="float32")
            else:
                s2_cache[year - 1] = np.full((len(cells), 3), np.nan, dtype="float32")
        s2 = s2_cache[year - 1][cidx]
        d["ndmi_summer"], d["ndvi_summer"], d["nbr_summer"] = s2[:, 0], s2[:, 1], s2[:, 2]
        d["ndmi_anom"] = s2[:, 0] - np.nanmean(s2_cache[year - 1][:, 0])
        # calendario
        d["es_finde"] = int(day_ts.day_of_week >= 5)
        d["dens_x_weekend"] = d["road_density_1km"].fillna(0) * d["es_finde"]
        doy = day_ts.day_of_year
        d["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        d["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        d["lat_f"] = d["lat"].to_numpy()
        d["lon_f"] = d["lon"].to_numpy()
        # recencia + lags vectorizados
        day64 = np.datetime64(day_ts.date())
        rec = np.full(len(cidx), 1825.0)
        n7 = np.zeros(len(cidx))
        n2 = np.zeros(len(cidx))
        for ang, out_cnt, win in ((ang5, n7, 7), (ang25, n2, 2)):
            hits = lag_tree.query_ball_point(cell_ll[cidx], ang, workers=-1)
            lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
            if not lens.any():
                continue
            cell_rep = np.repeat(np.arange(len(cidx)), lens)
            fire_flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])
            dts = (day64 - lag_dates[fire_flat]) / np.timedelta64(1, "D")
            ok = (dts > 0) & (dts <= 1825)
            cv, dv = cell_rep[ok], dts[ok]
            if out_cnt is n7:
                np.minimum.at(rec, cv, dv)
            mwin = dv <= win
            np.add.at(out_cnt, cv[mwin], 1.0)
        np.clip(rec, 0, 1825, out=rec)
        d["dias_ultimo_fuego_cercano"] = rec
        d["dist_ultimo_fuego_m"] = np.where(rec < 1825, 500.0, 5000.0)
        d["ardio_1km_1an"] = (rec <= 365).astype(int)
        d["n_ign_near_7d"] = n7
        d["n_ign_near_2d"] = n2
        d["n_ign_cell_3d"] = n2
        d["fecha"] = day_ts
        rows.append(d)
        if (n_day + 1) % 60 == 0:
            print(f"  {n_day + 1}/{len(days)} días", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} filas)")


if __name__ == "__main__":
    main()
