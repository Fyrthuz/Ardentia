"""Construye met_fc.parquet: forecast diario por estación + cadena FWI + lags,
con corrección de sesgo NGR-lite (fit obs-vs-forecast 2023, aplica a todo).

Uso: uv run python -m ardintra.features.build_met_fc
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FC_DAILY = Path("data/raw/openmeteo/forecast_daily.parquet")
OUT = Path("data/raw/openmeteo/met_fc.parquet")
WX = ["tmax_c", "rh_mean", "wind_max_kmh", "prec_mm"]


def main():
    from ardintra.features.build_dataset import load_met_stations
    from ardintra.features.fwi import fwi_series

    fc = pd.read_parquet(FC_DAILY)
    fc["fecha"] = pd.to_datetime(fc["fecha"]).dt.normalize()
    print("cobertura NaN:", fc.groupby(fc["fecha"].dt.year)[WX].apply(lambda g: g.isna().mean().round(3).to_dict()).to_dict())

    # cadena FWI sobre el forecast crudo
    out = []
    for st, g in fc.groupby("idema"):
        g = g.sort_values("fecha").dropna(subset=WX)
        if g.empty:
            continue
        f = fwi_series(g[["fecha", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm", "lat"]])
        f["idema"] = st
        f["provincia"] = g["provincia"].iloc[0]
        f["lat"], f["lon"] = g["lat"].iloc[0], g["lon"].iloc[0]
        out.append(f)
    fc2 = pd.concat(out, ignore_index=True)
    fc2 = fc2.merge(fc[["idema", "fecha"] + WX], on=["idema", "fecha"], how="left")
    print(f"FWI forecast: {len(fc2):,} filas · {fc2['fecha'].min().date()} → {fc2['fecha'].max().date()}")

    # NGR-lite: corrección lineal por estación (fit 2023)
    obs = load_met_stations()
    obs["fecha"] = pd.to_datetime(obs["fecha"]).dt.normalize()
    fc2["fecha"] = pd.to_datetime(fc2["fecha"]).dt.normalize()
    pairs = fc2.merge(obs[["idema", "fecha"] + WX], on=["idema", "fecha"],
                      suffixes=("_f", "_o")).dropna(subset=["tmax_c_o", "tmax_c_f"])
    p23 = pairs[pairs["fecha"].dt.year == 2023]
    print(f"pares obs-fcst 2023: {len(p23):,}")
    corr = {}
    for st, g in p23.groupby("idema"):
        c = {}
        for v in ("tmax_c", "rh_mean", "wind_max_kmh"):
            b, a = np.polyfit(g[v + "_f"].to_numpy(), g[v + "_o"].to_numpy(), 1)
            c[v] = (float(a), float(b))
        pf = g["prec_mm_o"].sum() / max(g["prec_mm_f"].sum(), 1e-6)
        c["prec_mm"] = (0.0, float(np.clip(pf, 0.5, 2.0)))
        corr[st] = c
    print(f"estaciones con corrección: {len(corr)}")

    # aplicar corrección y recalcular la cadena FWI (consistencia componentes)
    out2 = []
    for st, g in fc2.groupby("idema"):
        g = g.sort_values("fecha").copy()
        if st in corr:
            for v, (a, b) in corr[st].items():
                g[v] = np.maximum(a + b * g[v], 0) if v == "prec_mm" else a + b * g[v]
        g2 = g.dropna(subset=WX)
        if g2.empty:
            continue
        f = fwi_series(g2[["fecha", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm", "lat"]])
        f = f.merge(g2[["fecha"] + WX], on="fecha", how="left")   # ← weather corregido incluido
        f["idema"] = st
        f["provincia"] = g2["provincia"].iloc[0]
        f["lat"], f["lon"] = g2["lat"].iloc[0], g2["lon"].iloc[0]
        out2.append(f)
    fc3 = pd.concat(out2, ignore_index=True)
    for st, g in fc3.groupby("idema"):
        g = g.sort_values("fecha")
        fc3.loc[g.index, "ffmc_7d"] = g["ffmc"].rolling(7, min_periods=4).mean().values
        fc3.loc[g.index, "fwi_7d"] = g["fwi"].rolling(7, min_periods=4).mean().values
        fc3.loc[g.index, "fwi_prev"] = g["fwi"].shift(1).values
    fc3.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(fc3):,} filas) · cols: {fc3.columns.tolist()}")

    # verificación del sesgo residual (holdout 2024)
    p24 = fc3.merge(obs[["idema", "fecha", "fwi"]], on=["idema", "fecha"],
                    suffixes=("_c", "_o")).dropna(subset=["fwi_o", "fwi_c"])
    p24 = p24[p24["fecha"].dt.year == 2024]
    print(f"FWI 2024 corregido: bias={np.mean(p24.fwi_c - p24.fwi_o):+.2f} · "
          f"RMSE={np.sqrt(np.mean((p24.fwi_c - p24.fwi_o) ** 2)):.2f}")
    p24b = fc2.merge(obs[["idema", "fecha", "fwi"]], on=["idema", "fecha"],
                     suffixes=("_f", "_o")).dropna(subset=["fwi_o", "fwi_f"])
    p24b = p24b[p24b["fecha"].dt.year == 2024]
    print(f"FWI 2024 sin corregir: bias={np.mean(p24b.fwi_f - p24b.fwi_o):+.2f} · "
          f"RMSE={np.sqrt(np.mean((p24b.fwi_f - p24b.fwi_o) ** 2)):.2f}")


if __name__ == "__main__":
    main()
