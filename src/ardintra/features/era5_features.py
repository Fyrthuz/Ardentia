"""Features ERA5-Land por (celda, día): SWC l1/l2 y VPD diario (de t2m/td2m).

Lee los netCDF Galicia (data/raw/era5land/{var}_{year}.nc — 0.1°, 8 pasos
horarios, ~3 MB c/u) y produce, para las filas (h9, fecha) del dataset:
  swc_l1_mean / swc_l2_mean (humedad volumétrica) · vpd_max / vpd_mean (kPa).
Nearest-neighbor del centro de celda al grid; todo acotado en RAM por fichero.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ERA5_DIR = Path("data/raw/era5land")
COLS = ["swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"]


def _ensure_extracted() -> None:
    """CDS entrega Zips con data_0.nc dentro; extrae si hace falta (idempotente)."""
    import zipfile

    for f in sorted(ERA5_DIR.glob("*.nc")):
        if f.read_bytes()[:4] != b"PK\x03\x04":
            continue
        # el zip pasa a .zip_raw y el data_0.nc interior ocupa el nombre real
        tmp = f.with_suffix(".zip_raw_tmp")
        f.rename(tmp)
        with zipfile.ZipFile(tmp) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".nc"))
            zf.extract(inner, ERA5_DIR)
        (ERA5_DIR / inner).rename(f)  # data_0.nc -> var_year.nc
        tmp.unlink()
        print(f"extraído {f.name}")


def _esat_kpa(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def era5_features(pairs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean de slices sin dato (mar/borde nodata)
        return _era5_inner(pairs, cells)


def _era5_inner(pairs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    _ensure_extracted()
    data = {c: np.full(len(pairs), np.nan) for c in COLS}
    ll = pairs.merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    years = sorted(set(pairs.fecha.dt.year))

    iCache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for y in years:
        f_l1 = ERA5_DIR / f"soil_moisture_l1_{y}.nc"
        if f_l1.exists():
            with xr.open_dataset(f_l1) as ds:
                lon_g = ds["longitude"].to_numpy()
                lat_g = ds["latitude"].to_numpy()
            sub = ll.drop_duplicates("h9")
            iCache[y] = (
                np.clip(np.round((sub["lon"] - lon_g[0]) / (lon_g[-1] - lon_g[0]) * (len(lon_g) - 1)).astype(int), 0, len(lon_g) - 1),
                np.clip(np.round((sub["lat"] - lat_g[0]) / (lat_g[-1] - lat_g[0]) * (len(lat_g) - 1)).astype(int), 0, len(lat_g) - 1),
            )
            h9_order = sub["h9"].to_numpy()

        mask_y = (pairs.fecha.dt.year == y).to_numpy()
        if not mask_y.any() or y not in iCache:
            continue
        ii_all, jj_all = iCache[y]
        pos_h9 = {h: k for k, h in enumerate(h9_order)}
        i_by_h9 = dict(zip(h9_order, ii_all))
        j_by_h9 = dict(zip(h9_order, jj_all))
        dates = pd.to_datetime(pairs.fecha[mask_y]).dt.normalize().to_numpy()

        for var, col in (("soil_moisture_l1", "swc_l1_mean"), ("soil_moisture_l2", "swc_l2_mean")):
            f = ERA5_DIR / f"{var}_{y}.nc"
            if not f.exists():
                continue
            with xr.open_dataset(f) as ds:
                for extra in ("number", "expver"):
                    if extra in ds.dims:
                        ds = ds.isel({extra: 0}, drop=True)
                name = "swvl1" if var.endswith("l1") else "swvl2"
                arr = ds[name].to_numpy().astype("float32")  # (T_horas, lat, lon)
                n_steps = 8
                D = arr.shape[0] // n_steps
                days = pd.DatetimeIndex(ds["valid_time"].to_numpy()).normalize()[::n_steps][:D]
            day_pos = {d: k for k, d in enumerate(pd.DatetimeIndex(days).unique())}
            k_local = np.where(mask_y)[0]
            vals = np.full(len(k_local), np.nan)
            for n, (rowsel, day) in enumerate(zip(np.where(mask_y)[0], dates)):
                ip = i_by_h9.get(ll["h9"].iloc[rowsel], -1)
                jp = j_by_h9.get(ll["h9"].iloc[rowsel], -1)
                if ip < 0 or pd.Timestamp(day) not in day_pos:
                    continue
                # arr dims: (valid_time, latitude, longitude) → arr[día, j_lat, i_lon]
                gv = arr[day_pos[pd.Timestamp(day)] * n_steps:(day_pos[pd.Timestamp(day)] + 1) * n_steps, jp, ip]
                vals[n] = np.nanmean(gv)
            data[col][mask_y] = vals

        f_t, f_d = ERA5_DIR / f"t2m_{y}.nc", ERA5_DIR / f"td2m_{y}.nc"
        if f_t.exists() and f_d.exists():
            with xr.open_dataset(f_t) as ds_t, xr.open_dataset(f_d) as ds_d:
                t = ds_t["t2m"].to_numpy() - 273.15
                td = ds_d["d2m"].to_numpy() - 273.15
                days = pd.DatetimeIndex(ds_t["valid_time"].to_numpy()).normalize()
            day_pos = {d: k for k, d in enumerate(pd.DatetimeIndex(days).unique())}
            vmax, vmean = np.full(len(np.where(mask_y)[0]), np.nan), np.full(len(np.where(mask_y)[0]), np.nan)
            vp_hr = _esat_kpa(t) - _esat_kpa(td)  # (T_horas, lat, lon)
            D = vp_hr.shape[0] // n_steps
            vp_d = vp_hr[: D * n_steps].reshape(D, n_steps, *vp_hr.shape[1:])
            vp_max_all = np.nanmax(vp_d, axis=1)
            vp_mean_all = np.nanmean(vp_d, axis=1)
            for n, (rowsel, day) in enumerate(zip(np.where(mask_y)[0], dates)):
                ip = i_by_h9.get(ll["h9"].iloc[rowsel], -1)
                jp = j_by_h9.get(ll["h9"].iloc[rowsel], -1)
                if ip < 0 or pd.Timestamp(day) not in day_pos:
                    continue
                vmax[n] = vp_max_all[day_pos[pd.Timestamp(day)], jp, ip]
                vmean[n] = vp_mean_all[day_pos[pd.Timestamp(day)], jp, ip]
            data["vpd_max"][mask_y], data["vpd_mean"][mask_y] = vmax, vmean

    return pd.DataFrame(data, columns=COLS)
