"""Spatial Block Cross-Validation (California GeoAI 2025 / Kondylatos).

5-fold spatial block CV: KMeans sobre coordenadas → GroupKFold por bloque.
Evita la fuga por autocorrelación espacial que infla el ROC-AUC en splits
aleatorios. Complementa (no sustituye) el split temporal 2023-24.

Evalúa el balanced-ensemble (25×LGBM, target ≥10 ha) con:
  - temporal split (referencia)
  - spatial block CV (5 folds, reporta media±std)
  - transferencia Ourense
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/spatial_block_cv.json")

FEATS = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
    "ffmc_anom", "dmc_anom", "dc_anom",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend", "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
    "lat_f", "lon_f", "doy_sin", "doy_cos",
    "ctx_fwi", "ctx_dc", "ctx_tmax_c", "ctx_rh_mean", "ctx_swc_l1_mean", "ctx_vpd_max",
    "ctx_ndmi_summer", "ctx_spei30", "road_class",
]


def prep(df):
    X = df[[c for c in FEATS if c in df.columns]].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def fit_eval(Xtr, ytr, Xte, yte):
    spw = max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 0.1)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return {
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "brier": round(float(brier_score_loss(yte, p)), 4),
        "n_pos_test": int(yte.sum()),
    }


def main(n_blocks: int = 5):
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    y = df["big10"].to_numpy()
    X = prep(df)

    # temporal split (referencia)
    tr = (df["fecha"].dt.year <= 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()
    temporal = fit_eval(X[tr], y[tr], X[te], y[te])
    print(f"temporal: {temporal}")

    # transferencia Ourense
    our_tr = tr & (df["provincia"] != "ourense").to_numpy()
    our_te = te & (df["provincia"] == "ourense").to_numpy()
    ourense = fit_eval(X[our_tr], y[our_tr], X[our_te], y[our_te]) if our_te.sum() > 100 else None
    print(f"ourense:  {ourense}")

    # spatial block CV: KMeans sobre lat/lon de las celdas train
    statics = pd.read_parquet("data/processed/h3_cells/res9_static.parquet", columns=["h9", "lat", "lon"])
    df = df.merge(statics[["h9", "lat", "lon"]], on="h9", how="left")
    geo_cols = [c for c in ("lat_f", "lon_f") if c in df.columns]
    if not geo_cols:
        df["lat_f"] = df["lat"].fillna(42.8)
        df["lon_f"] = df["lon"].fillna(-8.1)
        geo_cols = ["lat_f", "lon_f"]
    coords = StandardScaler().fit_transform(df[geo_cols].fillna(42.8).to_numpy())
    blocks = KMeans(n_clusters=n_blocks, n_init=5, random_state=42).fit_predict(coords)
    df["block"] = blocks

    block_results = []
    for fold in range(n_blocks):
        tr_b = (df["block"] != fold).to_numpy() & (df["fecha"].dt.year <= 2022).to_numpy()
        te_b = (df["block"] == fold).to_numpy() & (df["fecha"].dt.year >= 2023).to_numpy()
        if te_b.sum() < 100 or y[tr_b].sum() < 10:
            continue
        r = fit_eval(X[tr_b], y[tr_b], X[te_b], y[te_b])
        r["fold"] = fold
        r["n_test"] = int(te_b.sum())
        block_results.append(r)
        print(f"  block {fold}: {r}")

    aps = [r["pr_auc"] for r in block_results]
    rocs = [r["roc_auc"] for r in block_results]
    spatial_summary = {
        "mean_pr_auc": round(float(np.mean(aps)), 4),
        "std_pr_auc": round(float(np.std(aps)), 4),
        "mean_roc_auc": round(float(np.mean(rocs)), 4),
        "std_roc_auc": round(float(np.std(rocs)), 4),
        "n_folds": len(block_results),
    }
    print(f"spatial block CV: {spatial_summary}")

    out = {
        "temporal_split": temporal,
        "transfer_ourense": ourense,
        "spatial_block_cv": {"blocks": block_results, "summary": spatial_summary},
        "n_blocks": n_blocks,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")
    return out


if __name__ == "__main__":
    main()
