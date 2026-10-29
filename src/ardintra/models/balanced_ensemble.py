"""Ensemble de undersamples balanceados (Texas-2025) + métricas AUC-ROC (literatura).

N=25 LGBMs: cada uno entrena con TODOS los positivos + una muestra aleatoria de
negativos del mismo tamaño (balance 1:1). Promedio de predict_proba → métricas
PR-AUC + ROC-AUC + recall@5%/1% en test 2023-24, contra el single-model LGBM
(scale_pos_weight) como baseline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import os as _os
_RES = int(_os.environ.get("ARDENTIA_RES", "9"))
DATASET = Path(f"data/processed/features/v1{'' if _RES == 9 else f'_res{_RES}'}/dataset.parquet")
OUT = Path(f"reports/balanced_ensemble{'' if _RES == 9 else f'_res{_RES}'}.json")

COLS_NUM = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend",
    "spei30", "spei90", "ffmc_anom", "dmc_anom", "dc_anom",
    "fwi_7d", "fwi_prev", "ffmc_7d",
]
COL_CAT = ["road_class"]


def _prep(df):
    X = pd.concat([df[COLS_NUM].copy(), df[COL_CAT].copy().astype("category")], axis=1)
    return X


def _metrics(y_true, y_prob):
    return {
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "recall_top5": round(float(y_true[np.argsort(-y_prob)[: max(int(len(y_prob) * .05), 1)]].sum() / max(y_true.sum(), 1)), 3),
        "recall_top1": round(float(y_true[np.argsort(-y_prob)[: max(int(len(y_prob) * .01), 1)]].sum() / max(y_true.sum(), 1)), 3),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def run(n_models: int = 25):
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    y = np.where((df["label"] == 1) & (surf >= 10), 1, 0)   # target ≥10 ha (el defendible)
    X = _prep(df)
    tr = ((df["fecha"].dt.year <= 2022) & df["fwi"].notna()).to_numpy()
    te = ((df["fecha"].dt.year >= 2023) & df["fwi"].notna()).to_numpy()

    idx_pos = np.where((y == 1) & tr)[0]
    idx_neg = np.where((y == 0) & tr)[0]
    print(f"train: {len(idx_pos):,} positivos · {len(idx_neg):,} negativos | test {te.sum():,}")

    # baseline single-model (scale_pos_weight como hasta ahora)
    spw = len(idx_neg) / max(len(idx_pos), 1)
    m0 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                            min_child_samples=100, scale_pos_weight=spw,
                            random_state=42, verbosity=-1)
    m0.fit(X[tr], y[tr])
    base = _metrics(y[te], m0.predict_proba(X[te])[:, 1])
    print("baseline single-LGBM:", base)

    # ensemble de undersamples balanceados
    probs = np.zeros(int(te.sum()))
    rng = np.random.default_rng(7)
    for i in range(n_models):
        sel_neg = rng.choice(idx_neg, size=len(idx_pos), replace=False)
        sel = np.concatenate([idx_pos, sel_neg])
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, random_state=42, verbosity=-1)
        m.fit(X.iloc[sel], y[sel])
        probs += m.predict_proba(X[te])[:, 1] / n_models
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{n_models} modelos entrenados", flush=True)

    ens = _metrics(y[te], probs)
    ens["n_models"] = n_models
    print("balanced-ensemble:", ens)

    out = {"baseline_single": base, "balanced_ensemble": ens, "target": "big>=10ha"}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")
    return out


if __name__ == "__main__":
    run()
