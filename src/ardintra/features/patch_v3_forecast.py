"""Dataset v3: base v1 (muestreo original, tras refutar el estratificado)
+ columnas de causa (de EGIF, valor analítico)
+ features met del día D sustituidas por FORECAST D-1 en filas de test (2023-24).

Train conserva meteo observada (mejor histórico disponible); test usa forecast
(régimen operativo real). Spei/anomalías/swc/vpd se mantienen observadas
(índices lentos, poco sensibles al error de forecast).

Uso: uv run python -m ardintra.features.patch_v3_forecast
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

V1 = Path("/tmp/opencode/dataset_v1_backup.parquet")
V2 = Path("data/processed/features/v1/dataset.parquet")
MET_FC = Path("data/raw/openmeteo/met_fc.parquet")
OUT = Path("data/processed/features/v1/dataset.parquet")

FC_VARS = ["tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
           "ffmc", "dmc", "dc", "isi", "bui", "fwi", "ffmc_7d", "fwi_7d", "fwi_prev"]


def main():
    df = pd.read_parquet(V1)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"base v1: {len(df):,} filas")

    # 1) causa desde v2 (positivos: h9+fecha únicos)
    v2 = pd.read_parquet(V2, columns=["h9", "fecha", "cause", "motivation", "es_humana", "es_rayo", "label"])
    v2 = v2[(v2["label"] == 1) & v2["cause"].notna()][["h9", "fecha", "cause", "motivation", "es_humana", "es_rayo"]]
    v2["fecha"] = pd.to_datetime(v2["fecha"]).dt.normalize()
    v2 = v2.drop_duplicates(["h9", "fecha"]).rename(columns={"fecha": "fecha_n"})
    df["fecha_n"] = df["fecha"].dt.normalize()
    df = df.merge(v2, on=["h9", "fecha_n"], how="left", suffixes=("", "_v2"))
    df["cause"] = df["cause_v2"].fillna(df["cause"]) if "cause_v2" in df.columns else df["cause"]
    df = df.drop(columns=[c for c in ("cause_v2", "motivation_v2", "es_humana_v2", "es_rayo_v2", "fecha_n") if c in df.columns])
    print(f"causa propagada: {df['cause'].notna().sum():,} filas con causa")

    # 2) met forecast para filas test
    fc = pd.read_parquet(MET_FC)
    fc["fecha"] = pd.to_datetime(fc["fecha"]).dt.normalize()
    piv = {v: fc.pivot_table(index="fecha", columns="idema", values=v, aggfunc="first")
           for v in FC_VARS}

    from ardintra.features.build_dataset import load_met_stations
    met = load_met_stations()
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    cells = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")[["h9", "lat", "lon"]]
    uniq_cells = np.sort(df["h9"].unique())
    cell_ll = np.radians(cells.set_index("h9").loc[uniq_cells][["lat", "lon"]].to_numpy())
    dd, ii = tree.query(cell_ll, k=3)
    w = 1.0 / np.maximum(dd, 1e-5) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    st_ids = st_meta["idema"].to_numpy()
    cell_pos = {h: i for i, h in enumerate(uniq_cells)}

    m_test = (df["fecha"].dt.year >= 2023).to_numpy()
    print(f"filas test a parchear con forecast: {m_test.sum():,}")
    row_days = df.loc[m_test, "fecha"].dt.normalize().to_numpy().astype("datetime64[D]")
    uniq_days, inv = np.unique(row_days, return_inverse=True)
    row_cell = df.loc[m_test, "h9"].map(cell_pos).to_numpy()
    for var in FC_VARS:
        p = piv[var]
        p.index = pd.DatetimeIndex(p.index).normalize()
        pv = p.reindex(pd.DatetimeIndex(uniq_days)).reindex(columns=st_ids).to_numpy()  # [n_dias, n_st]
        vv = pv[inv]                                    # [n_rows_test, n_st]
        vv3 = np.take_along_axis(vv, ii[row_cell], axis=1)   # 3 estaciones más cercanas
        wv = w[row_cell]
        valid = ~np.isnan(vv3)
        interp = np.nansum(np.where(valid, vv3, 0) * np.where(valid, wv, 0), axis=1) / \
                 np.maximum(np.where(valid, wv, 0).sum(axis=1), 1e-9)
        old = df[var].to_numpy(dtype=float)
        full = np.full(len(df), np.nan)
        full[m_test] = interp
        n_changed = int(np.nansum(np.abs(interp - old[m_test]) > 1e-9))
        df[var] = np.where(m_test & ~np.isnan(full), full, old)
        print(f"  {var}: {n_changed:,} filas actualizadas")

    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} filas)")


if __name__ == "__main__":
    main()
