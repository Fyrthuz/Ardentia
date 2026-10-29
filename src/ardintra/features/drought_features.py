"""Features de sequía por estación (todo desde nuestro raw, sin descargas):

- SPEI-30 / SPEI-90: precip acumulada vs PET (Hargreaves desde tmax/tmin+lat)
  estandarizado por día del calendario (media/σ multi-año). El paper mexicano
  (ifor3623) y Turco 2019: SPEI≥2 meses bate a precip cruda como predictor.
- ANOMALÍAS de FFMC/DMC/DC: valor del día − media multi-año del mismo día
  calendario (la "DFMC anomaly" fue la variable #1 del estudio europeo NHESS-2025).

Entrada: serie diaria de una estación [fecha, tmax_c, tmin_c, prec_mm, ffmc, dmc, dc, lat]
Salida:  columnas [spei30, spei90, ffmc_anom, dmc_anom, dc_anom]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pet_hargreaves(tmax, tmin, lat_deg, doy):
    """PET Hargreaves-Samani (mm/día) — solo requiere tmax/tmin/lat/doy."""
    lat = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    ra_mj = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws)
    )
    ra_mm = ra_mj / 2.45  # MJ/m² → mm/día eq.
    return 0.0023 * ra_mm * (np.maximum(tmax, 0) - np.maximum(tmin, 0)) ** 0.5 * (np.maximum(tmax, 0) + 17.8) / 2 * 0.408


def _std_rolling(x: pd.Series, win: int, doy_ref: pd.Series) -> np.ndarray:
    """SPEI simplificado: acumulado `win` días menos la media multi-año del mismo
    día calendario, dividido por σ multi-año (aprox normal; suficiente MVP)."""
    acc = x.rolling(win, min_periods=max(win // 2, 10)).sum()
    doy = doy_ref.astype(int)
    df = pd.DataFrame({"acc": acc, "doy": doy})
    mu = df.groupby("doy")["acc"].transform(lambda s: s.rolling(9, center=True, min_periods=3).mean().bfill().ffill())
    sd = df.groupby("doy")["acc"].transform(lambda s: s.rolling(9, center=True, min_periods=3).std().bfill().ffill())
    return ((df["acc"] - mu) / sd.replace(0, np.nan)).to_numpy()


def drought_features(df_st: pd.DataFrame) -> pd.DataFrame:
    """df: [fecha, tmax_c, tmin_c, prec_mm, ffmc, dmc, dc, lat] ordenado por fecha."""
    d = df_st.copy().sort_values("fecha")
    d["fecha"] = pd.to_datetime(d["fecha"])
    doy = d["fecha"].dt.dayofyear
    lat = float(d["lat"].iloc[0]) if d["lat"].notna().any() else 42.8

    pet = _pet_hargreaves(d["tmax_c"].to_numpy(), d["tmin_c"].to_numpy(), lat, doy.to_numpy())
    balance = d["prec_mm"].to_numpy() - pet  # P − ET0 (déficit si <0)

    out = pd.DataFrame(index=d.index)
    b = pd.Series(balance, index=d.index)
    out["spei30"] = _std_rolling(b, 30, doy)
    out["spei90"] = _std_rolling(b, 90, doy)

    # anomalías de códigos vs media multi-año del mismo día calendario (±7 d)
    for code in ("ffmc", "dmc", "dc"):
        s = d[code].astype(float)
        clim = s.groupby(doy).transform(lambda s_: s_.rolling(15, center=True, min_periods=5).mean().bfill().ffill())
        sd = s.groupby(doy).transform(lambda s_: s_.rolling(15, center=True, min_periods=5).std().bfill().ffill())
        out[f"{code}_anom"] = ((s - clim) / sd.replace(0, np.nan)).clip(-5, 5).to_numpy()
    return out[["spei30", "spei90", "ffmc_anom", "dmc_anom", "dc_anom"]]
