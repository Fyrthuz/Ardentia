"""Ensemble Gate final — versión estable con UNA convención de arrays.

REGLA DE ORO: todas las probabilidades (final_probs/val_probs) son arrays
fila-completos (longitud = len(df)). El slicing a test/val ocurre SOLO al
consumirlos. Así no puede haber desalineaciones entre familias.

Familias: fino_lgbm · regional (parent-day) · catboost (Optuna) ·
lgbm_big10_tuned (Optuna) · ft_attention (probs guardadas si el tamaño encaja).
Pesos de fusión: Optuna (rank-avg ponderado) sobre val 2022 con modelos fit ≤2021.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/ensemble_gate.json")
OUT_MD = Path("reports/ensemble_gate.md")

NUM = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend",
]
FEATS = NUM + ["road_class"]


def prep(df, cols=None):
    cols = cols or FEATS
    X = df[cols].copy()
    for c in cols:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def fit_lgb(df, y, mask, min_child=100, params=None):
    base = dict(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=min_child,
        scale_pos_weight=max((y[mask] == 0).sum() / max((y[mask] == 1).sum(), 1), 0.1),
        random_state=42, verbosity=-1,
    )
    if params:
        base.update(params)
    m = lgb.LGBMClassifier(**base)
    m.fit(prep(df)[mask], y[mask])
    return m


def main(n_trials: int = 20):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].copy()
    df["parent8"] = [h3.cell_to_parent(x, 8) for x in df["h9"]]
    df["fecha_n"] = df["fecha"].dt.normalize()
    y = df["big"].to_numpy()
    N = len(df)

    fe_year = df["fecha"].dt.year.to_numpy()
    tr = fe_year <= 2022
    te = fe_year >= 2023
    inner = fe_year <= 2021
    val22 = fe_year == 2022

    final_probs: dict[str, np.ndarray] = {}   # SIEMPRE longitud N
    val_probs: dict[str, np.ndarray] = {}

    # ---------- 1) fino lgbm
    m1 = fit_lgb(df, y, tr)
    final_probs["fino_lgbm"] = m1.predict_proba(prep(df))[:, 1]
    m1v = fit_lgb(df, y, inner)
    val_probs["fino_lgbm"] = m1v.predict_proba(prep(df))[:, 1]

    # ---------- 2) regional parent-day (features = media de hijas por (parent, día))
    agg = df.groupby(["parent8", "fecha_n"]).agg(
        {**{c: "mean" for c in NUM},
         "road_class": lambda s: s.mode().iat[0] if len(s.mode()) else "local",
         "big": "max"}).reset_index()
    y8 = agg["big"].to_numpy()
    tr8 = agg["fecha_n"].dt.year.to_numpy() <= 2022
    m2 = fit_lgb(agg, y8, tr8, min_child=50)
    Xreg = prep(agg)
    pr_all_reg = np.full(len(agg), np.nan)
    pr_all_reg[m2.predict(Xreg) >= 0] = np.nan  # placeholder no necesario; predict directo:
    pr_all_reg = m2.predict_proba(Xreg)[:, 1]   # sobre TODAS las filas agg (train+test)
    reg_map = dict(zip(zip(agg["parent8"], agg["fecha_n"]), pr_all_reg))
    row_reg = np.array([reg_map.get((p, f), np.nan) for p, f in zip(df["parent8"], df["fecha_n"])])
    ok = ~np.isnan(row_reg)
    med = np.median(row_reg[ok]) if ok.any() else 0.0
    final_probs["regional"] = np.where(ok, row_reg, med)
    # val: modelo entrenado solo con ≤2021
    m2v = fit_lgb(agg, y8, agg["fecha_n"].dt.year.to_numpy() <= 2021, min_child=50)
    prv = m2v.predict_proba(Xreg)[:, 1]
    regv_map = dict(zip(zip(agg["parent8"], agg["fecha_n"]), prv))
    row_regv = np.array([regv_map.get((p, f), np.nan) for p, f in zip(df["parent8"], df["fecha_n"])])
    val_probs["regional"] = np.where(ok, row_regv, np.nanmedian(row_regv[ok]) if ok.any() else 0.0)

    # ---------- 3) catboost (Optuna 20, inner split ≤2021 → val 2022)
    def cb(p):
        return CatBoostClassifier(
            iterations=p["iters"], learning_rate=p["lr"], depth=p["depth"],
            l2_leaf_reg=p["l2"], auto_class_weights="Balanced",
            verbose=0, random_seed=42, allow_writing_files=False)

    def gcb(t):
        return {"iters": t.suggest_int("iters", 300, 1100, step=100),
                "lr": t.suggest_float("lr", 0.01, 0.15, log=True),
                "depth": t.suggest_int("depth", 4, 10),
                "l2": t.suggest_float("l2", 1, 30, log=True)}

    Xdf = prep(df)
    def cb_obj(t):
        m = cb(gcb(t)).fit(Xdf[inner], y[inner], cat_features=["road_class"])
        return average_precision_score(y[val22], m.predict_proba(Xdf[val22])[:, 1])

    st_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st_cb.optimize(cb_obj, n_trials=n_trials)
    m3 = cb(st_cb.best_params).fit(Xdf[tr], y[tr], cat_features=["road_class"])
    final_probs["catboost"] = m3.predict_proba(Xdf)[:, 1]
    m3v = cb(st_cb.best_params).fit(Xdf[inner], y[inner], cat_features=["road_class"])
    val_probs["catboost"] = m3v.predict_proba(Xdf)[:, 1]

    # ---------- 4) LGBM big10 específico (Optuna 20)
    def gbig(t):
        return {"n_estimators": t.suggest_int("n_est", 200, 1400, step=100),
                "learning_rate": t.suggest_float("lr", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("leaves", 16, 128),
                "min_child_samples": t.suggest_int("min_child", 20, 250),
                "subsample": t.suggest_float("subsample", 0.6, 1.0), "subsample_freq": 1,
                "colsample_bytree": t.suggest_float("colsample", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_l", 1e-3, 10, log=True),
                "scale_pos_weight": t.suggest_float("spw", 10, 60)}

    def big_obj(t):
        m = fit_lgb(df, y, inner, min_child=100, params=gbig(t))
        return average_precision_score(y[val22], m.predict_proba(prep(df))[val22][:, 1])

    tg = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43))
    tg.optimize(big_obj, n_trials=n_trials)
    m4 = fit_lgb(df, y, tr, min_child=100, params=gbig(tg.best_trial))
    final_probs["lgbm_big10_tuned"] = m4.predict_proba(prep(df))[:, 1]
    m4v = fit_lgb(df, y, inner, min_child=100, params=gbig(tg.best_trial))
    val_probs["lgbm_big10_tuned"] = m4v.predict_proba(prep(df))[:, 1]

    # ---------- 5) FT-Attention guardado (si encaja la longitud)
    ft_file = Path("reports/ft_prob_big10.parquet")
    if ft_file.exists():
        pft = pd.read_parquet(ft_file)["p"].to_numpy()
        if len(pft) == N:
            final_probs["ft_attention"] = pft
            val_probs["ft_attention"] = pft

    # ---------- 6) pesos de fusión por Optuna sobre val 2022
    fams = list(final_probs.keys())
    val_rank = {f: rankdata(val_probs[f])[val22] / max(val22.sum(), 1) for f in fams}

    def fuse_obj(t):
        ws = np.array([t.suggest_float(f"w_{f}", 0.0, 1.0) for f in fams])
        if ws.sum() < 1e-6:
            return 0.0
        s = sum(w * val_rank[f] for w, f in zip(ws, fams)) / ws.sum()
        return average_precision_score(y[val22], s)

    wf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=44))
    wf.optimize(fuse_obj, n_trials=25)
    weights = {f: wf.best_params[f"w_{f}"] for f in fams}

    te_rank = {f: rankdata(final_probs[f])[te] / max(te.sum(), 1) for f in fams}
    fused_te = sum(weights[f] * te_rank[f] for f in fams) / max(sum(weights.values()), 1e-9)
    ap_fused = average_precision_score(y[te], fused_te)

    fam_res = {}
    for f in fams:
        ap = average_precision_score(y[te], te_rank[f])
        o = np.argsort(-te_rank[f]); k = max(int(te.sum() * 0.05), 1)
        fam_res[f] = {
            "test_ap": round(float(ap), 4),
            "recall_top5": round(float(y[te][o[:k]].sum() / max(y[te].sum(), 1)), 3),
            "peso_fusion": round(weights[f], 3),
        }
        print(f"{f:18s} AP={ap:.3f} · recall@5%={fam_res[f]['recall_top5']:.0%} · peso={weights[f]:.2f}")

    o = np.argsort(-fused_te); k = max(int(te.sum() * 0.05), 1)
    out = {
        "test_prevalence": round(float(y[te].mean()), 5),
        "n_test": int(te.sum()),
        "fusion_ap": round(float(ap_fused), 4),
        "fusion_lift": round(float(ap_fused / max(y[te].mean(), 1e-9)), 1),
        "fusion_recall_top5": round(float(y[te][o[:k]].sum() / max(y[te].sum(), 1)), 3),
        "fusion_precision_top5": round(float(y[te][o[:k]].mean()), 4),
        "familias": fam_res,
        "optuna": {"catboost": st_cb.best_params, "lgbm_big10": tg.best_params},
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    md = ["# Ensemble Gate (≥10 ha, test 2023-24)", "",
          f"prev test {out['test_prevalence']:.2%} · N {out['n_test']:,}", "",
          "| Familia | PR-AUC | recall@5% | peso |", "|---|---|---|---|"]
    for f, r in fam_res.items():
        md.append(f"| {f} | {r['test_ap']} | {r['recall_top5']:.0%} | {r['peso_fusion']} |")
    md += ["", f"**FUSIÓN: PR-AUC {out['fusion_ap']} ({out['fusion_lift']}× azar) · "
               f"recall@5% {out['fusion_recall_top5']:.0%} · precision@5% {out['fusion_precision_top5']:.1%}**"]
    OUT_MD.write_text("\n".join(md))
    print("\n", json.dumps({k: v for k, v in out.items() if k != "familias"}, indent=2))
    return out


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
