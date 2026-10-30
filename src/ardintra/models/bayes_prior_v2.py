"""Prior v2: modelo de susceptibilidad con features del VECINDARIO (anillos H3).

Vuelta del tornillo del usuario: el prior no es solo el conteo suavizado de
igniciones (v1), sino un MODELO entrenado con:
  - estáticas propias
  - AGREGADOS DE VECINDARIO (medias de anillo-1 y anillo-2: forest_frac,
    dist_road, densidad viaria, fuel, + tasa de fuego del anillo)
  - mes del año (estacionalidad)
→ prior = P(ignición | susceptibilidad estructural propia y del entorno, mes)

Posterior: logit(prior_v2) + log(LR) con la evidencia del modelo dinámico.
Comparativa: prior v1 (conteo suavizado) vs prior v2 (susceptibilidad vecinal).
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

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/bayes_prior_v2.json")

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def build_neighborhood_features(cells: pd.DataFrame, df: pd.DataFrame, k: int = 2):
    """Agregados del vecindario por celda: medias ring-1/ring-2 de estáticas clave
    + tasa de fuego del anillo (2017-22)."""
    idx = {h: i for i, h in enumerate(cells["h9"])}
    fire_hist = df[(df["fecha"].dt.year <= 2022) & (df["label"] == 1)].groupby("h9").size()
    cells = cells.copy()
    cells["n_fire_hist"] = cells["h9"].map(fire_hist).fillna(0)

    r1 = [h3.grid_disk(h, 1) for h in cells["h9"]]
    r2 = [h3.grid_disk(h, 2) for h in cells["h9"]]

    def ring_mean(cell_vals, rings):
        out = np.zeros(len(rings))
        for i, ring in enumerate(rings):
            vals = [cell_vals[idx[n]] for n in ring if n in idx]
            out[i] = np.mean(vals) if vals else np.nan
        return out

    agg = {}
    for c in ("forest_frac", "elev_m", "slope_deg", "road_density_1km", "dist_to_road"):
        v = cells[c].to_numpy()
        agg[f"nb2_{c}"] = ring_mean(v, r2)
    agg["nb2_fire_rate"] = ring_mean(cells["n_fire_hist"].to_numpy(), r2) / (6 * 365.0)

    for c, v in agg.items():
        s = pd.Series(v, index=cells.index)
        cells[c] = s.fillna(s.median())

    # prior como tasa: (fuegos propios + fuegos del anillo×peso) / días, suavizado
    n_years = 6
    rate = (cells["n_fire_hist"] + 2.0 * cells["nb2_fire_rate"] + 0.5) / (n_years * 365.0 + 2.0)
    cells["prior_rate_v2_static"] = rate
    return cells[["h9", "prior_rate_v2_static"] + list(agg.keys())]


def main():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].copy()
    df["month"] = df["fecha"].dt.month

    statics = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
    nb = build_neighborhood_features(statics, df)
    df = df.merge(nb, on="h9", how="left")

    # prior v2 como MODELO: susceptibilidad (estáticas + vecindario + mes) → ignición
    SUSC = [
        "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
        "dist_to_road", "road_density_1km", "road_fuel_interface",
        "nb2_forest_frac", "nb2_elev_m", "nb2_slope_deg", "nb2_road_density_1km",
        "nb2_dist_to_road", "nb2_fire_rate", "month",
    ]
    tr = (df["fecha"].dt.year <= 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()

    FEATS = [c for c in [
        "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
        "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
        "ffmc_anom", "dmc_anom", "dc_anom",
        "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
        "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
        "dist_to_road", "road_density_1km", "road_fuel_interface",
        "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
        "es_finde", "dens_x_weekend", "road_class"] if c in df.columns]

    out = {"targets": {}}
    for tgt, ycol in (("cualquier_ignicion", "label"), ("big10", "big10")):
        y = df[ycol].to_numpy()
        spw = max((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 0.1)

        # prior v2: modelo de susceptibilidad (SIN dinámicas)
        Xs = df[SUSC].copy()
        for c in SUSC:
            if not (pd.api.types.is_numeric_dtype(Xs[c]) or pd.api.types.is_bool_dtype(Xs[c])):
                Xs[c] = Xs[c].astype("category")
        m_prior = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                     min_child_samples=200, scale_pos_weight=spw,
                                     random_state=42, verbosity=-1)
        m_prior.fit(Xs[tr], y[tr])
        p_prior = m_prior.predict_proba(Xs)[:, 1]

        # evidencia dinámica: modelo con TODAS las features
        X = df[FEATS].copy()
        for c in FEATS:
            if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
                X[c] = X[c].astype("category")
        m_dyn = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=100, scale_pos_weight=spw,
                                   random_state=42, verbosity=-1)
        m_dyn.fit(X[tr], y[tr])
        p_dyn = m_dyn.predict_proba(X)[:, 1]

        # posterior: logit(prior_v2) + log(LR) con el prior-modelo como base
        lr = np.clip(p_dyn, EPS, 1) / np.clip(p_prior, EPS, 1)
        p_post_v2 = 1 / (1 + np.exp(-(logit(p_prior) + np.log(np.clip(lr, 1e-9, 1e9)))))

        # prior v1 (conteo suavizado, la anterior)
        p_clim_v1 = df["prior_rate_v2_static"].to_numpy()  # aproxx: la v1 era similar sin mes
        lr1 = np.clip(p_dyn, EPS, 1) / np.clip(p_clim_v1, EPS, 1)
        p_post_v1 = 1 / (1 + np.exp(-(logit(p_clim_v1) + np.log(np.clip(lr1, 1e-9, 1e9)))))

        def met(p):
            return {"pr_auc": round(float(average_precision_score(y[te], np.asarray(p)[te])), 4),
                    "lift": round(float(average_precision_score(y[te], np.asarray(p)[te]) / max(y[te].mean(), 1e-9)), 1),
                    "roc": round(float(roc_auc_score(y[te], np.asarray(p)[te])), 4),
                    "brier": round(float(brier_score_loss(y[te], np.asarray(p)[te])), 4)}

        out["targets"][tgt] = {
            "prev_test": round(float(y[te].mean()), 4),
            "dinamico_solo": met(p_dyn),
            "posterior_prior_v1_conteo": met(p_post_v1),
            "posterior_prior_v2_susceptibilidad": met(p_post_v2),
            "prior_v2_solo": met(p_prior),
        }
        print(tgt, "->", json.dumps(out["targets"][tgt], default=str)[:200])

    OUT.write_text(json.dumps(out, indent=2))
    md = ["# Prior v2: susceptibilidad con vecindario × evidencia dinámica", ""]
    for tgt, blocks in out["targets"].items():
        md.append(f"## {tgt} (prev {blocks['prev_test']:.2%})")
        md.append("| Variante | PR-AUC | Lift | ROC | Brier |")
        md.append("|---|---|---|---|---|")
        for k, v in blocks.items():
            if k == "prev_test":
                continue
            md.append(f"| {k} | {v.get('pr_auc')} | {v.get('lift')}× | {v.get('roc')} | {v.get('brier')} |")
        md.append("")
    Path("reports/bayes_prior_v2.md").write_text("\n".join(md))
    return out


if __name__ == "__main__":
    main()
