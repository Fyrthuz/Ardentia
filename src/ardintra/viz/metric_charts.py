"""Gráficas de métricas del ciclo completo (por resolución + ablación + curvas).

Genera reports/figures/*.png desde los json de balanced_ensemble y el dataset
(LGBM para las curvas ROC/PR por resolución).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

FIGS = Path("reports/figures")
AREA = {9: 0.105, 8: 0.737, 7: 5.16, 6: 36, 5: 253, 4: 1770}
CORP = {"verde": "#36a852", "amarillo": "#ebd23c", "naranja": "#f58c28", "rojo": "#e83c28", "granate": "#960f2d"}

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


def _prob_big10(res: int):
    suf = "" if res == 9 else f"_res{res}"
    df = pd.read_parquet(f"data/processed/features/v1{suf}/dataset.parquet")
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    ok = df["fwi"].notna().to_numpy()
    y = ((df["label"] == 1) & (surf >= 10)).astype(int).to_numpy()[ok]
    df = df[ok].reset_index(drop=True)
    X = df[[c for c in FEATS if c in df.columns]].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    tr = (df["fecha"].dt.year <= 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()
    spw = max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 0.1)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1)
    m.fit(X[tr], y[tr])
    return y[te], m.predict_proba(X[te])[:, 1]


def charts_resoluciones():
    """Barras: ROC-AUC · lift · recall@5% por resolución."""
    rows = []
    for res in (9, 8, 7, 6, 5, 4):
        suf = "" if res == 9 else f"_res{res}"
        f = Path(f"reports/balanced_ensemble{suf}.json")
        if not f.exists():
            continue
        be = json.load(open(f))["balanced_ensemble"]
        df = pd.read_parquet(f"data/processed/features/v1{suf}/dataset.parquet",
                             columns=["label", "split_temporal", "fwi", "surface_ha"])
        surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
        big = ((df["label"] == 1) & (surf >= 10)).astype(int)
        te = (df["split_temporal"] == "test") & df["fwi"].notna()
        prev_big = big[te].mean()
        rows.append({"res": res, "roc": be["roc_auc"], "lift": be["pr_auc"] / prev_big,
                     "rec5": be["recall_top5"], "ap": be["pr_auc"]})
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = [f"res{r}" for r in rdf["res"]]
    colors = [CORP["verde"], CORP["amarillo"], CORP["naranja"], CORP["rojo"], CORP["granate"], "#5b2a86"]

    ax = axes[0]
    b = ax.bar(x, rdf["roc"], color=colors)
    ax.axhline(0.898, ls="--", c="gray", label="RF Kondylatos (lit.)")
    ax.axhline(0.926, ls=":", c="black", label="ConvLSTM (lit.)")
    ax.set_title("ROC-AUC por resolución (≥10 ha)"); ax.legend(fontsize=8); ax.set_ylim(0.6, 0.95)
    for rect, v in zip(b, rdf["roc"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    b = ax.bar(x, rdf["lift"], color=colors)
    ax.set_title("Lift sobre azar (PR-AUC / prevalencia test)")
    for rect, v in zip(b, rdf["lift"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.3, f"{v:.1f}×", ha="center", fontsize=9)

    ax = axes[2]
    b = ax.bar(x, rdf["rec5"], color=colors)
    ax.set_title("recall@5% diario")
    ax.axhline(rdf["rec5"].max(), ls="--", c="gray", alpha=0.5)
    for rect, v in zip(b, rdf["rec5"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "resoluciones.png", dpi=120)
    plt.close(fig)
    print("-> figures/resoluciones.png")


def charts_curvas():
    """Curvas ROC + PR por resolución (target ≥10 ha)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {9: CORP["rojo"], 8: CORP["naranja"], 7: CORP["amarillo"], 6: CORP["verde"],
              5: "#5b2a86", 4: "gray"}
    for res in (9, 8, 7, 6, 5, 4):
        suf = "" if res == 9 else f"_res{res}"
        if not Path(f"reports/balanced_ensemble{suf}.json").exists():
            continue
        yte, p = _prob_big10(res)
        fpr, tpr, _ = roc_curve(yte, p)
        prec, rec, _ = precision_recall_curve(yte, p)
        ax1.plot(fpr, tpr, label=f"res{res} (AUC {roc_auc_score(yte, p):.3f})", color=colors[res])
        ax2.plot(rec, prec, label=f"res{res} (AP {average_precision_score(yte, p):.3f})", color=colors[res])
    ax1.plot([0, 1], [0, 1], ls="--", c="gray", alpha=0.5)
    ax1.set_title("Curvas ROC (target ≥10 ha)"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend(fontsize=9)
    ax2.set_title("Curvas Precision-Recall"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGS / "curvas_roc_pr.png", dpi=120)
    plt.close(fig)
    print("-> figures/curvas_roc_pr.png")


def chart_ablacion():
    """Escalera de la ablación M0→M4b (res9, cualquier ignición)."""
    rep = json.load(open("reports/phase2_ablation.json"))
    rows = [(r["modelo"], r["temporal"]["pr_auc"], r["temporal"].get("recall_top5pct"))
            for r in rep["results"] if "temporal" in r]
    names = [r[0].replace("_", " ") for r in rows]
    aps = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = [CORP["verde"], CORP["amarillo"], CORP["naranja"], CORP["rojo"], CORP["granate"], "#5b2a86"][:len(rows)]
    b = ax.bar(names, aps, color=cols)
    for rect, v in zip(b, aps):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title("Ablación M0→M4b (cualquier ignición ≥0.1 ha, res9)")
    ax.set_ylabel("PR-AUC")
    plt.xticks(rotation=18, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGS / "ablacion.png", dpi=120)
    plt.close(fig)
    print("-> figures/ablacion.png")


def main():
    charts_resoluciones()
    charts_curvas()
    chart_ablacion()


if __name__ == "__main__":
    main()
