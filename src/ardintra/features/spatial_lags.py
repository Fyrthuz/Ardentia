"""Features espaciales y de momentum de igniciones (Ronda 1 de la literatura):

B) LAGS AUTOREGRESIVOS ESPACIO-TEMPORALES de igniciones (Martínez et al., Int.
   J. Wildland Fire — paper sobre GALICIA con modelos PAR(P): las igniciones de
   los últimos 3 días en la celda y de los 2 días en el vecindario son de los
   predictores más significativos; -15% RMSE):
     - n_ign_cell_3d:  nº de igniciones en la MISMA celda últimos 3 días
     - n_ign_near_2d:  nº de igniciones en radio 2.5 km últimos 2 días
     - n_ign_near_7d:  nº de igniciones en radio 5 km últimos 7 días (momentum zonal)
   Siempre con fecha ESTRUCTAMENTE anterior a la predicha (sin look-ahead).

A) CONTEXTO ESPACIAL de dinámicas (CNN-2D de Zhang 2021 / Next-Day-Spread de
   Google en versión tabular): media de las dinámicas de las 8 celdas más
   cercanas del MISMO día ("la zona se seca aunque la celda no").

C) GEOGRAFÍA + ESTACIONALIDAD (LOAN 2023): lat, lon, sin/cos del día del año.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DYN_CTX = ["fwi", "ffmc", "dc", "tmax_c", "rh_mean", "wind_max_kmh", "swc_l1_mean",
           "vpd_max", "ndmi_summer", "spei30"]


def fire_lag_features(df: pd.DataFrame, ig: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Conteos de igniciones previas en la celda y su entorno (radios 2.5/5 km).

    Eficiente: KDTree de fuegos con query_ball_point vectorizado; el conteo de la
    MISMA celda vía arrays ordenados por celda + searchsorted (sin bucles O(N²)).
    """
    ll = dict(zip(cells["h9"], zip(cells["lat"].to_numpy(), cells["lon"].to_numpy())))
    fpts = ig.dropna(subset=["h9"]).merge(cells[["h9", "lat", "lon"]], on="h9", how="inner").dropna(subset=["lat", "lon"])
    tree = cKDTree(np.radians(fpts[["lat", "lon"]].to_numpy()))
    ig_dates = pd.to_datetime(fpts["fecha"]).to_numpy().astype("datetime64[D]")

    # historial por celda (para el conteo same-cell en O(log n) por fila)
    per_cell_dates = {}
    for h, g in fpts.groupby("h9"):
        per_cell_dates[h] = np.sort(pd.to_datetime(g["fecha"]).to_numpy().astype("datetime64[D]"))

    pair_ll = np.array([ll.get(h, (np.nan, np.nan)) for h in df["h9"]])
    ok_ll = ~np.isnan(pair_ll[:, 0])
    d0 = pd.to_datetime(df["fecha"]).to_numpy().astype("datetime64[D]")

    out = pd.DataFrame({
        "n_ign_cell_3d": np.zeros(len(df), dtype="float32"),
        "n_ign_near_2d": np.zeros(len(df), dtype="float32"),
        "n_ign_near_7d": np.zeros(len(df), dtype="float32"),
    })

    hits_25 = tree.query_ball_point(np.radians(pair_ll[ok_ll]), 2500 / 6_371_000.0, workers=-1)
    hits_50 = tree.query_ball_point(np.radians(pair_ll[ok_ll]), 5000 / 6_371_000.0, workers=-1)

    for n, i in enumerate(np.where(ok_ll)[0]):
        di = d0[i]
        # misma celda: fechas del fuego de ESTA celda en (d0-3, d0) estricto
        h_i = df["h9"].iloc[i]
        cell_days = per_cell_dates.get(h_i)
        if cell_days is not None and len(cell_days):
            k_past = np.searchsorted(cell_days, di, side="left")  # fechas < di
            past3 = cell_days[max(0, k_past - 3):k_past]
            n3 = int(((di - past3) / np.timedelta64(1, "D") >= 1).sum())  # (d0-3, d0) estricto
            out.loc[i, "n_ign_cell_3d"] = n3
        # entorno
        near50 = hits_50[n]
        if len(near50):
            dts = (di - ig_dates[near50]) / np.timedelta64(1, "D")
            past7 = (dts > 0) & (dts <= 7)
            out.loc[i, "n_ign_near_7d"] = float(past7.sum())
            out.loc[i, "n_ign_near_2d"] = float(((dts > 0) & (dts <= 2)).sum())
    return out


def spatial_context_features(df: pd.DataFrame, cells: pd.DataFrame, k_neighbors: int = 9) -> pd.DataFrame:
    """Media de las dinámicas de las k celdas más cercanas del MISMO día.

    KDTree por día sobre las filas de ese día (muestreo ~150-300/día) → k vecinos
    más próximos geográficamente → media de sus dinámicas (excluye la propia fila).
    """
    DYN = [c for c in DYN_CTX if c in df.columns]
    out = pd.DataFrame(index=df.index, columns=[f"ctx_{c}" for c in DYN], dtype="float32")
    lonlat = df[["h9"]].merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    lonlat = lonlat.fillna({"lat": 42.8, "lon": -8.1})

    for day, g_idx in df.groupby("fecha").indices.items():
        idx = np.asarray(g_idx)
        pts = np.radians(lonlat.loc[idx, ["lat", "lon"]].to_numpy())
        if len(idx) <= k_neighbors:
            day_means = df.loc[idx, DYN].mean()
            for c in DYN:
                out.loc[idx, f"ctx_{c}"] = day_means[c]
            continue
        tree = cKDTree(pts)
        _, ii = tree.query(pts, k=k_neighbors + 1)  # el +1 incluye la propia fila
        flat = idx[ii[:, 1:]].ravel()   # (n_rows*k,) — loc no admite claves 2D
        nb_vals = df.loc[flat, DYN].to_numpy().reshape(len(idx), k_neighbors, len(DYN))
        ctx = np.nanmean(nb_vals, axis=1)
        for j, c in enumerate(DYN):
            out.loc[idx, f"ctx_{c}"] = ctx[:, j]
    return out


def geo_features(df: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    ll = df[["h9"]].merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    doy = df["fecha"].dt.dayofyear.to_numpy()
    return pd.DataFrame({
        "lat_f": ll["lat"].to_numpy(),
        "lon_f": ll["lon"].to_numpy(),
        "doy_sin": np.sin(2 * np.pi * doy / 365),
        "doy_cos": np.cos(2 * np.pi * doy / 365),
    }, index=df.index)
