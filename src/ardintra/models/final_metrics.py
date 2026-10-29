"""Tabla final de métricas: jerárquico fusionado con TODAS las features nuevas
(S2 8/8 + sequía SPEI/anomalías + AEMET 18/18 + recencia + calendario).

Modelos evaluados (target ≥10 ha Y cualquier ignición):
  fino global · regional parent-day · fusión rank · (fusión producto)
Métricas: PR-AUC · ROC-AUC · precision/recall@5%/1% · Brier.

Uso: uv run python -m ardintra.models.final_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import os as _os
_RES = int(_os.environ.get("ARDENTIA_RES", "9"))
_SUF = "" if _RES == 9 else f"_res{_RES}"
DATASET = Path(f"data/processed/features/v1{_SUF}/dataset.parquet")
OUT = Path(f"reports/final_metrics{_SUF}.json")

FEATS = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
    "ffmc_anom", "dmc_anom", "dc_anom",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend", "road_class",
]


def prep(df, cols=None):
    cols = cols or FEATS
    X = df[[c for c in cols if c in df.columns]].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def metrics_row(y_true, y_score, k_pcts=(0.05, 0.01)):
    y_score = np.asarray(y_score, dtype=float)
    if np.nanmax(y_score) > 1.0:  # score crudo (p.ej. FWI 0-50): AP solo depende del orden
        y_score = rankdata(y_score) / len(y_score)
    row = {
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
    }
    o = np.argsort(-y_score)
    for pct in k_pcts:
        k = max(int(len(y_score) * pct), 1)
        row[f"recall@{pct:.0%}"] = round(float(y_true[o[:k]].sum() / max(y_true.sum(), 1)), 3)
        row[f"precision@{pct:.0%}"] = round(float(y_true[o[:k]].mean()), 4)
    return row


def run():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].copy()
    df["parent8"] = [h3.cell_to_parent(x, 8) for x in df["h9"]]

    NUM = [c for c in FEATS if pd.api.types.is_numeric_dtype(df[c]) or c == "road_class"]
    yr = df["fecha"].dt.normalize().rename("fecha")

    y_all = df["label"].to_numpy()
    y_big = df["big10"].to_numpy()
    tr = df["fecha"].dt.year.to_numpy() <= 2022
    te = df["fecha"].dt.year.to_numpy() >= 2023

    X9 = prep(df)
    out = {"targets": {}, "n_test": int(te.sum()), "dataset": str(DATASET)}

    for tgt, yvec, tgt_col in (("cualquier_ignicion", y_all, "label"), ("big10", y_big, "big10")):
        agg = df.groupby(["parent8", yr]).agg(
            {**{c: "mean" for c in NUM if c != "road_class"},
             "road_class": lambda s: s.mode().iat[0] if len(s.mode()) else "local",
             tgt_col: "max"}).reset_index()
        y8v = agg[tgt_col].to_numpy()
        tr8 = agg["fecha"].dt.year.to_numpy() <= 2022

        spw = max((yvec[tr] == 0).sum() / max((yvec[tr] == 1).sum(), 1), 0.1)
        m9 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                min_child_samples=100, scale_pos_weight=spw,
                                random_state=42, verbosity=-1)
        m9.fit(X9[tr], yvec[tr])

        m8 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                min_child_samples=50,
                                scale_pos_weight=max((y8v[tr8] == 0).sum() / max((y8v[tr8] == 1).sum(), 1), 0.1),
                                random_state=42, verbosity=-1)
        m8.fit(prep(agg)[tr8], y8v[tr8])

        # probs del regional mapeadas a filas df por (parent, fecha)
        pr_agg = m8.predict_proba(prep(agg))[:, 1]
        keymap = dict(zip(zip(agg["parent8"], agg["fecha"]), pr_agg))
        dte = df[te].copy()
        p_reg = np.array([keymap.get((p, f), np.nan) for p, f in
                          zip(dte["parent8"], dte["fecha"].dt.normalize())])
        med = np.nanmedian(p_reg) if np.isfinite(np.nanmedian(p_reg)) else 0.0
        p_reg = np.where(np.isnan(p_reg), med, p_reg)

        yte = yvec[te]
        p9 = m9.predict_proba(prep(dte))[:, 1]
        fused_rank = (rankdata(p9) + rankdata(p_reg)) / (2 * len(p9))  # normaliza a [0,1]
        p_fwi = dte["fwi"].to_numpy()

        out["targets"][tgt] = {
            "prev": round(float(yte.mean()), 4),
            "fino_solo": metrics_row(yte, p9),
            "regional_solo": metrics_row(yte, p_reg),
            "fusion_rank": metrics_row(yte, fused_rank),
            "fwi_solo": metrics_row(yte, p_fwi),
        }
        print(f"{tgt}: fino {out['targets'][tgt]['fino_solo']['pr_auc']} · "
              f"fusión {out['targets'][tgt]['fusion_rank']['pr_auc']}")

    OUT.write_text(json.dumps(out, indent=2))
    md = ["# Métricas finales (jerárquico + features completos, test 2023-24)", ""]
    for tgt, blocks in out["targets"].items():
        md.append(f"## Target: {tgt} (prev {blocks['prev']:.2%})")
        md.append("| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |")
        md.append("|---|---|---|---|---|---|---|")
        for name, m in blocks.items():
            if name == "prev":
                continue
            md.append(f"| {name} | {m['pr_auc']} | {m['roc_auc']} | {m.get('recall@5%')} | "
                      f"{m.get('precision@5%')} | {m.get('recall@1%')} | {m['brier']} |")
        md.append("")
    Path(f"reports/final_metrics{_SUF if (SUF := _SUF) else ''}_x.md".replace("_x", "")).write_text("\n".join(md))
    print(f"-> {OUT}")
    return out


if __name__ == "__main__":
    run()
