"""Teacher: inferencia del ensemble TabPFN sobre el set de destilación.

El teacher puntúa 420k filas full-grid de train (2017-2022) — el student
(LGBM) aprenderá a imitar su ranking a coste de inferencia mínimo.

Uso: uv run python -m ardintra.models.distill_teacher
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path("data/processed/features/v1/dataset.parquet")
DISTILL = Path("data/processed/features/v1/distill_set.parquet")
OUT = Path("data/processed/features/v1/distill_set.parquet")  # añade columna teacher_p

DYN = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
       "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
       "ffmc_anom", "dmc_anom", "dc_anom", "fwi_7d", "fwi_prev", "ffmc_7d",
       "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
       "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
       "es_finde", "dens_x_weekend", "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
       "lat_f", "lon_f", "doy_sin", "doy_cos"]
STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
          "dist_to_road_log", "road_density_log", "road_fuel_interface", "road_class"]
FEATS = STATIC + DYN
N_ENSEMBLE = 10
SUBSIZE = 10_000
CHUNK = 20_000


def main():
    import torch
    from tabpfn import TabPFNClassifier

    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True)
    y = df["big10"].to_numpy()
    tr = df["fecha"].dt.year.to_numpy() <= 2022

    Xtr = df.loc[tr, FEATS].copy().reset_index(drop=True)
    ytr = y[tr]
    cat_maps = {}
    for c in Xtr.columns:
        if not (pd.api.types.is_numeric_dtype(Xtr[c]) or pd.api.types.is_bool_dtype(Xtr[c])):
            cats = pd.Categorical(Xtr[c]).categories
            cat_maps[c] = {v: i for i, v in enumerate(cats)}
            Xtr[c] = pd.Categorical(Xtr[c], categories=cats).codes.astype("float64")
    Xtr = Xtr.astype("float64")

    dist = pd.read_parquet(DISTILL)
    Xd = dist[FEATS].copy()
    for c, mapping in cat_maps.items():
        Xd[c] = Xd[c].map(mapping).fillna(-1)
    Xd = Xd.astype("float64")
    print(f"distill: {len(Xd):,} filas × {Xd.shape[1]} features", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)
    pos_all = np.where(ytr == 1)[0]
    neg_all = np.where(ytr == 0)[0]
    models = []
    for k in range(N_ENSEMBLE):
        neg_k = rng.choice(neg_all, size=SUBSIZE - len(pos_all), replace=False)
        idx_k = np.sort(np.concatenate([pos_all, neg_k]))
        clf = TabPFNClassifier(device=device, random_state=42)
        clf.fit(Xtr.iloc[idx_k], ytr[idx_k])
        models.append(clf)
        print(f"  teacher {k + 1}/{N_ENSEMBLE} ajustado", flush=True)

    n = len(Xd)
    teacher = np.zeros(n)
    for s in range(0, n, CHUNK):
        chunk = Xd.iloc[s:s + CHUNK]
        ps = [m.predict_proba(chunk)[:, 1] for m in models]
        teacher[s:s + CHUNK] = np.mean(ps, axis=0)
        done = s + len(chunk)
        if done % 100_000 < CHUNK:
            print(f"  {done:,}/{n:,} filas inferidas", flush=True)
        torch.cuda.empty_cache()

    dist["teacher_p"] = teacher.round(6)
    dist.to_parquet(OUT, index=False)
    print(f"-> teacher_p añadido a {OUT} · p50={np.median(teacher):.5f} · "
          f"p95={np.quantile(teacher, .95):.4f} · max={teacher.max():.4f}")


if __name__ == "__main__":
    main()
