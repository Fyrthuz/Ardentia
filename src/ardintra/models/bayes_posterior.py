"""Predicción BAYESIANA: prior climático de igniciones × evidencia del modelo → posterior.

Formulación (la que pide el usuario):
  PRIOR:   P_clim(celda) — tasa histórica de igniciones de la celda (suavizada
           espacialmente con anillos H3 y ajustada por mes), estimada de EGIF 2017-2022.
  EVIDENCIA: el score del modelo (LGBM/ensemble) como razón de verosimilitud
           LR = P_modelo / P_clim  (cuánto multiplican las condiciones de HOY al prior).
  POSTERIOR: logit(p_post) = logit(p_clim) + β·log(LR)   [β=1 → Bayes puro]
           equivale a p_post = p_clim·LR / (1 + p_clim·(LR−1)).

Variante "modelo con prior": el logit(prior) entra como FEATURE al LGBM y deja
que aprenda cuándo fiarse de la climatología (shrinkage empírico aprendido).

Evalúa las tres variantes (raw / bayesiana / prior-como-feature) en ambos targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/bayes_posterior.json")

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_prior(df: pd.DataFrame, cells: pd.DataFrame, k_smooth: int = 2,
                prior_years: tuple = (2017, 2022)) -> pd.DataFrame:
    """Prior climático por celda: tasa diaria de igniciones 2017-22 suavizada por
    anillos + componente mensual (tasa del mes relativa a la media anual)."""
    days_per_year = 365.0
    hist = df[df["fecha"].dt.year.between(*prior_years)]
    fire_cells = hist[hist["label"] == 1].groupby("h9").size()

    cells = cells.copy()
    cells["n_fire"] = cells["h9"].map(fire_cells).fillna(0)
    n_years = prior_years[1] - prior_years[0] + 1

    # suavizado por anillos (empirical-Bayes shrinkage espacial)
    idx = {h: i for i, h in enumerate(cells["h9"])}
    n_fire = cells["n_fire"].to_numpy()
    smoothed = np.zeros(len(cells))
    for i, h in enumerate(cells["h9"]):
        ring = h3.grid_disk(h, k_smooth)
        vals = [n_fire[idx[n]] for n in ring if n in idx]
        # shrinkage: media local ponderada por el tamaño del anillo
        smoothed[i] = np.mean(vals) if vals else n_fire[i]
    cells["prior_rate"] = (smoothed + 0.5) / (n_years * days_per_year + 1.0)  # Laplace

    # componente mensual: multiplicador de la tasa del mes (global, no por celda)
    fires_m = hist[hist["label"] == 1].groupby(hist["fecha"].dt.month).size() / n_years
    fires_m = (fires_m / fires_m.mean()).reindex(range(1, 13)).fillna(1.0)
    cells["prior_month_mult"] = cells["month"] if "month" in cells else np.nan
    cells["prior_rate_seasonal"] = cells["prior_rate"] * cells["month"].map(fires_m) if "month" in cells else cells["prior_rate"]
    return cells[["h9", "prior_rate", "prior_rate_seasonal"]]


def main():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].copy()
    df["month"] = df["fecha"].dt.month

    statics = pd.read_parquet("data/processed/h3_cells/res9_static.parquet", columns=["h9", "lat", "lon"])
    prior = build_prior(df, statics)
    df = df.merge(prior, on="h9", how="left")
    df["prior_rate"] = df["prior_rate"].fillna(df["prior_rate"].median())
    df["prior_rate_seasonal"] = df["prior_rate_seasonal"].fillna(df["prior_rate_seasonal"].median())

    FEATS = [c for c in [
        "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
        "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
        "ffmc_anom", "dmc_anom", "dc_anom",
        "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
        "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
        "dist_to_road", "road_density_1km", "road_fuel_interface",
        "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
        "es_finde", "dens_x_weekend", "road_class"] if c in df.columns]

    tr = (df["fecha"].dt.year <= 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()
    val22 = (df["fecha"].dt.year == 2022).to_numpy()

    out = {"targets": {}}
    for tgt, ycol in (("cualquier_ignicion", "label"), ("big10", "big10")):
        y = df[ycol].to_numpy()
        spw = max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 0.1)
        X = df[FEATS].copy()
        for c in FEATS:
            if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
                X[c] = X[c].astype("category")

        # modelo RAW (sin prior)
        m_raw = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=100, scale_pos_weight=spw,
                                   random_state=42, verbosity=-1)
        m_raw.fit(X[tr], y[tr])
        p_raw_all = m_raw.predict_proba(X)[:, 1]

        # BAYES: posterior = prior ⊕ evidencia — SIN isotónica (aplanan el ranking y
        # degradan el AP; el log-odds funciona con el score crudo directamente)
        p_clim = df["prior_rate_seasonal"].to_numpy()
        lr = np.clip(p_raw_all, EPS, 1) / np.clip(p_clim, EPS, 1)
        p_post = 1 / (1 + np.exp(-(logit(p_clim) + np.log(np.clip(lr, 1e-9, 1e9)))))

        # VARIANTE FEATURE: logit(prior) como feature extra reentrenando
        X2 = X.copy()
        X2["logit_prior"] = logit(df["prior_rate_seasonal"].to_numpy())
        m2 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                min_child_samples=100, scale_pos_weight=spw,
                                random_state=42, verbosity=-1)
        m2.fit(X2[tr], y[tr])
        p_feat = m2.predict_proba(X2)[:, 1]

        def met(p):
            ap = average_precision_score(y[te], np.asarray(p)[te])
            return {"pr_auc": round(float(ap), 4),
                    "lift": round(float(ap / max(y[te].mean(), 1e-9)), 1),
                    "roc": round(float(roc_auc_score(y[te], p[te])), 4),
                    "brier": round(float(brier_score_loss(y[te], p[te])), 4)}

        out["targets"][tgt] = {
            "prev_test": round(float(y[te].mean()), 4),
            "modelo_raw_sin_calibrar": met(p_raw_all),
            "bayes_posterior": met(p_post),
            "prior_como_feature": met(p_feat),
            "fwi_solo": {"pr_auc": round(float(average_precision_score(y[te], df.loc[te, "fwi"].rank() / len(df))), 4)},
        }
        print(tgt, "->", json.dumps(out["targets"][tgt], default=str)[:180])

    OUT.write_text(json.dumps(out, indent=2))
    md = ["# Bayesiano: prior climático × evidencia del modelo", ""]
    for tgt, blocks in out["targets"].items():
        md.append(f"## {tgt} (prev {blocks['prev_test']:.2%})")
        md.append("| Variante | PR-AUC | Lift | ROC | Brier |")
        md.append("|---|---|---|---|---|")
        for k, v in blocks.items():
            if k == "prev_test":
                continue
            md.append(f"| {k} | {v.get('pr_auc')} | {v.get('lift')}× | {v.get('roc')} | {v.get('brier')} |")
        md.append("")
    Path("reports/bayes_posterior.md").write_text("\n".join(md))
    return out


if __name__ == "__main__":
    main()
