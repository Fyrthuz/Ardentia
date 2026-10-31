"""TabPFN v2 (transformer preentrenado en datasets tabulares sintéticos) vs LGBM.

Fundamento: TabPFN (Nature 2025) es SOTA en tabular con pocos positivos —
nuestro régimen (1.496 big10 en train). Limites nativos ~10k filas →
ensembling por subsampling (todos los positivos + negativos estratificados),
promedio de probabilidades.

Evaluación: todos los positivos test + 20k negativos aleatorios (el AUC es
insesgado bajo subsampling uniforme de negativos).

Uso: uv run python -m ardintra.models.tabpfn_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/tabpfn_results.json")
N_ENSEMBLE = int(__import__('os').environ.get('TABPFN_ENSEMBLE', '30'))
SUBSIZE = 10_000
N_TEST_NEG = 20_000

DYN = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
       "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
       "ffmc_anom", "dmc_anom", "dc_anom", "fwi_7d", "fwi_prev", "ffmc_7d",
       "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
       "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
       "es_finde", "dens_x_weekend", "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
       "lat_f", "lon_f", "doy_sin", "doy_cos"]
STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
          "dist_to_road_log", "road_density_log", "road_fuel_interface", "road_class"]


def main():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True)
    feats = [c for c in STATIC + DYN if c in df.columns]
    X = df[feats].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    y = df["big10"].to_numpy()
    yr = df["fecha"].dt.year.to_numpy()
    tr, te = yr <= 2022, yr >= 2023
    print(f"train {tr.sum():,} ({y[tr].sum()} big10) · test {te.sum():,} ({y[te].sum()} big10)", flush=True)

    Xtr, ytr = X[tr].reset_index(drop=True), y[tr]
    Xte, yte = X[te].reset_index(drop=True), y[te]

    # TabPFN 2.x requiere numérico: categóricas → códigos enteros (NaN → -1)
    def to_numeric(dfX: pd.DataFrame) -> pd.DataFrame:
        Z = dfX.copy()
        for c in Z.columns:
            if pd.api.types.is_categorical_dtype(Z[c]):
                Z[c] = Z[c].cat.codes.astype("float64")
        return Z.astype("float64")

    Xtr_num, Xte_num = to_numeric(Xtr), to_numeric(Xte)

    # evaluación: todos los positivos + N negativos
    rng = np.random.default_rng(42)
    pos_te = np.where(yte == 1)[0]
    neg_te = np.where(yte == 0)[0]
    neg_sample = rng.choice(neg_te, size=min(N_TEST_NEG, len(neg_te)), replace=False)
    eval_idx = np.sort(np.concatenate([pos_te, neg_sample]))
    Xev, yev = Xte_num.iloc[eval_idx], yte[eval_idx]
    print(f"eval: {len(eval_idx):,} filas ({len(pos_te)} positivos)", flush=True)

    from tabpfn import TabPFNClassifier
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pos_all = np.where(ytr == 1)[0]
    neg_all = np.where(ytr == 0)[0]
    preds = []
    for k in range(N_ENSEMBLE):
        n_neg = SUBSIZE - len(pos_all)
        neg_k = rng.choice(neg_all, size=n_neg, replace=False)
        idx_k = np.sort(np.concatenate([pos_all, neg_k]))
        clf = TabPFNClassifier(device=device, random_state=42)
        clf.fit(Xtr_num.iloc[idx_k], ytr[idx_k])
        p = clf.predict_proba(Xev)[:, 1]
        preds.append(p)
        auc_k = roc_auc_score(yev, p)
        print(f"  ensemble {k + 1}/{N_ENSEMBLE}: AUC={auc_k:.4f}", flush=True)
    p_avg = np.mean(preds, axis=0)
    res = {
        "tabpfn_v2_ensemble": {
            "auc": round(float(roc_auc_score(yev, p_avg)), 4),
            "ap": round(float(average_precision_score(yev, p_avg)), 4),
            "n_ensemble": N_ENSEMBLE, "subsize": SUBSIZE,
        },
        "nota": "AUC sobre todos los positivos test + 20k negativos (insesgado)",
        "ref_lgbm_v3b_full_test": {"auc": 0.8874, "ap": 0.0339},
    }
    # referencia LGBM en el MISMO subconjunto de evaluación (comparación justa)
    import lightgbm as lgb
    spw = max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 0.1)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1).fit(Xtr, ytr)
    pl = m.predict_proba(Xte.iloc[eval_idx])[:, 1]
    res["lgbm_mismo_eval"] = {"auc": round(float(roc_auc_score(yev, pl)), 4),
                              "ap": round(float(average_precision_score(yev, pl)), 4)}
    print(f"\nTabPFN v2 (ens {N_ENSEMBLE}×{SUBSIZE}):", res["tabpfn_v2_ensemble"])
    print(f"LGBM (mismo eval set):        ", res["lgbm_mismo_eval"])
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
