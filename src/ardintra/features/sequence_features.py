"""Features de secuencia temporal: para cada fila (celda, día) del dataset,
ventana de los T días anteriores de meteo/FWI en esa celda (IDW 3 estaciones).

Fase 3 (Mesogeos track A adaptado): contexto temporal explícito vs features
solo del día D. Salida:
  - X_seq   [n_rows, T, n_vars]  para LSTM
  - X_flat  [n_rows, T*n_vars]   para LGBM (aplanado)
  - X_sum   [n_rows, n_vars*4]   resúmenes (mean/max/min/trend) para LGBM
  - idx     índices de fila del dataset conservados (historia completa)

Uso: uv run python -m ardintra.features.sequence_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("data/processed/features/v1/sequences.npz")

SEQ_VARS = ["tmax_c", "rh_mean", "wind_max_kmh", "prec_mm", "fwi", "ffmc", "dmc", "dc"]
T_WINDOW = 14


def main():
    from ardintra.features.build_dataset import load_met_stations

    df = pd.read_parquet(DATASET, columns=["h9", "fecha", "label", "surface_ha", "split_temporal"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"dataset: {len(df):,} filas")

    met = load_met_stations()
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    st_ids = st_meta["idema"].to_numpy()

    # pivot estación × día para cada variable
    dates = pd.date_range("2016-06-01", "2024-12-31", freq="D")
    date_idx = pd.Series(np.arange(len(dates)), index=dates)
    n_st, n_days = len(st_ids), len(dates)
    mats = {}
    for var in SEQ_VARS:
        p = met.pivot_table(index="fecha", columns="idema", values=var, aggfunc="first")
        p = p.reindex(columns=st_ids)
        m = np.full((n_days, n_st), np.nan)
        pos = date_idx.reindex(p.index).to_numpy()
        ok = ~np.isnan(pos)
        m[pos[ok].astype(int)] = p.to_numpy()[ok]
        mats[var] = m  # [n_days, n_st]
    print(f"pivots: {n_days} días × {n_st} estaciones")

    # 3 estaciones más cercanas por celda (una vez)
    cells = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")[["h9", "lat", "lon"]]
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    uniq_cells = np.sort(df["h9"].unique())
    cell_ll = np.radians(cells.set_index("h9").loc[uniq_cells][["lat", "lon"]].to_numpy())
    dd, ii = tree.query(cell_ll, k=3)
    w = 1.0 / np.maximum(dd, 1e-5) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    cell_pos = {h: i for i, h in enumerate(uniq_cells)}
    row_cell = df["h9"].map(cell_pos).to_numpy()
    row_day = date_idx.reindex(df["fecha"].dt.normalize()).to_numpy().astype(int)

    n = len(df)
    X_seq = np.zeros((n, T_WINDOW, len(SEQ_VARS)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for t in range(T_WINDOW):
        day_t = row_day - t
        inr = (day_t >= 0) & (day_t < n_days)
        for vi, var in enumerate(SEQ_VARS):
            vv = np.full(n, np.nan, dtype=np.float32)
            st_vals = mats[var][day_t[inr]]                    # [n_in, n_st]
            rows_in = np.where(inr)[0]
            st_sel = ii[row_cell[rows_in]]                     # [n_in, 3]
            s3 = np.take_along_axis(st_vals, st_sel, axis=1)   # [n_in, 3]
            w3 = w[row_cell[rows_in]]
            good = ~np.isnan(s3)
            vv[rows_in] = np.nansum(np.where(good, s3, 0) * np.where(good, w3, 0), axis=1) / \
                np.maximum(np.where(good, w3, 0).sum(axis=1), 1e-9)
            X_seq[:, T_WINDOW - 1 - t, vi] = vv                # orden cronológico: t=0 es el más antiguo
        valid |= inr
    # filas con ventana completa (sin NaN en ninguna celda temporal)
    full = ~np.isnan(X_seq).any(axis=(1, 2))
    print(f"filas con ventana completa de {T_WINDOW} días: {full.sum():,} de {n:,}")

    idx = np.where(full)[0]
    X_seq = X_seq[idx]
    # resúmenes para LGBM: media, max, min, tendencia (pendiente simple)
    X_sum = np.concatenate([
        X_seq.mean(axis=1), X_seq.max(axis=1), X_seq.min(axis=1),
        (X_seq[:, -1, :] - X_seq[:, 0, :]),                 # delta últimos 14 días
    ], axis=1).astype(np.float32)
    X_flat = X_seq.reshape(len(idx), -1)
    np.savez_compressed(OUT, idx=idx, X_seq=X_seq, X_sum=X_sum, X_flat=X_flat,
                        labels=df["label"].to_numpy()[idx],
                        big10=((df["label"] == 1) & (pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0) >= 10)).to_numpy()[idx],
                        years=df["fecha"].dt.year.to_numpy()[idx],
                        seq_vars=np.array(SEQ_VARS), t_window=T_WINDOW)
    print(f"-> {OUT} · X_seq {X_seq.shape} · X_sum {X_sum.shape}")


if __name__ == "__main__":
    main()
