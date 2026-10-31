"""Student: LGBM destilado del teacher TabPFN.

Entrena sobre 420k filas full-grid de train con target = teacher_p (regresión).
Evaluación en el mismo subset que TabPFN (137 positivos + 20k negativos) +
comparación con LGBM-de-labels y blend de ambos.

Uso: uv run python -m ardintra.models.distill_student
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

DATASET = Path("data/processed/features/v1/dataset.parquet")
DISTILL = Path("data/processed/features/v1/distill_set.parquet")
STUDENT = Path("models/student_lgbm.joblib")
OUT = Path("reports/distill_results.json")
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
FEATS = STATIC + DYN


def main():
    dist = pd.read_parquet(DISTILL)
    # mapeo de categóricas desde el dataset de train (consistente en todo el pipeline)
    df0 = pd.read_parquet(DATASET)
    df0 = df0[df0["fwi"].notna()].reset_index(drop=True)
    df0 = df0[df0["fecha"].dt.year <= 2022]
    cat_maps = {}
    for c in FEATS:
        if not (pd.api.types.is_numeric_dtype(df0[c]) or pd.api.types.is_bool_dtype(df0[c])):
            cat_maps[c] = {v: i for i, v in enumerate(pd.Categorical(df0[c]).categories)}
    def apply_codes(D: pd.DataFrame) -> pd.DataFrame:
        Z = D[FEATS].copy()
        for c, mapping in cat_maps.items():
            Z[c] = Z[c].map(mapping).fillna(-1)
        return Z.astype("float64")
    Xd = apply_codes(dist)
    y_teacher = dist["teacher_p"].to_numpy()
    print(f"distill: {len(Xd):,} filas · teacher p50={np.median(y_teacher):.4f}", flush=True)

    # student: regresión sobre la salida del teacher
    stud = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                             min_child_samples=100, random_state=42, verbosity=-1)
    stud.fit(Xd, y_teacher)
    STUDENT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": stud, "feats": FEATS, "cat_maps": cat_maps}, STUDENT)
    print(f"-> {STUDENT}", flush=True)

    # evaluación en el subset estándar
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True)
    X = apply_codes(df)
    y = df["big10"].to_numpy()
    yr = df["fecha"].dt.year.to_numpy()
    tr, te = yr <= 2022, yr >= 2023
    Xte, yte = X[te].reset_index(drop=True), y[te]

    rng = np.random.default_rng(42)
    pos_te = np.where(yte == 1)[0]
    neg_te = np.where(yte == 0)[0]
    eval_idx = np.sort(np.concatenate([pos_te, rng.choice(neg_te, size=N_TEST_NEG, replace=False)]))
    Xev, yev = Xte.iloc[eval_idx], yte[eval_idx]

    p_stud = stud.predict(Xev)
    p_stud = np.clip(p_stud, 0, 1)
    # LGBM de labels (referencia, mismo eval set)
    spw = max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 0.1)
    m_lab = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, scale_pos_weight=spw,
                               random_state=42, verbosity=-1).fit(X[tr], y[tr])
    p_lab = m_lab.predict_proba(Xev)[:, 1]
    p_blend = 0.5 * (p_stud + p_lab)

    res = {
        "teacher_tabpfn": {"auc": 0.9387, "ap": 0.1653},
        "student_distilled": {"auc": round(float(roc_auc_score(yev, p_stud)), 4),
                              "ap": round(float(average_precision_score(yev, p_stud)), 4)},
        "lgbm_labels": {"auc": round(float(roc_auc_score(yev, p_lab)), 4),
                        "ap": round(float(average_precision_score(yev, p_lab)), 4)},
        "blend_50_50": {"auc": round(float(roc_auc_score(yev, p_blend)), 4),
                        "ap": round(float(average_precision_score(yev, p_blend)), 4)},
        "nota": "eval: 137 positivos test + 20k negativos (AUC insesgado)",
    }
    print(json.dumps(res, indent=2))
    OUT.write_text(json.dumps(res, indent=2))
    # guardar también el modelo de labels para el blend operativo
    joblib.dump({"model": m_lab, "feats": FEATS, "cat_maps": cat_maps}, Path("models/lgbm_labels.joblib"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
