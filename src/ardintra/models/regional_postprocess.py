"""Post-proceso regional: agrupa celdas en regiones y predice a nivel región.

La idea del usuario: el LGBM da P(fire) por celda, pero lo que importa operativamente
es "¿va a haber un fuego EN ESTA ZONA?". Agrupamos celdas en regiones (H3 res7 parent)
y para cada región-día computamos:
  - max(p_cell) en la región
  - mean(p_cell)
  - top-3 mean
  - coherencia espacial: fracción de celdas de la región que son "altas"
  - nº de celdas sobre umbral
→ la etiqueta región = "¿algún fuego ≥10 ha en la región ese día?"

Esto es un CLASIFICADOR REGIONAL que usa las probabilidades del modelo fino como
features + coherencia espacial. Complementa (no sustituye) la predicción por celda.
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT = Path("reports/regional_postprocess.json")


def main():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    df["parent7"] = [h3.cell_to_parent(x, 7) for x in df["h9"]]

    # P(fire) por celda del balanced-ensemble simplificado (1 LGBM, rápido)
    FEATS = [c for c in [
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
        "ctx_ndmi_summer", "ctx_spei30", "road_class"] if c in df.columns]
    X = df[FEATS].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    y_cell = df["big10"].to_numpy()
    tr = (df["fecha"].dt.year <= 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()
    spw = max((y_cell[tr] == 0).sum() / max((y_cell[tr] == 1).sum(), 1), 0.1)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1)
    m.fit(X[tr], y_cell[tr])
    df["p_cell"] = m.predict_proba(X)[:, 1]

    # etiqueta regional: ¿algún fuego ≥10 ha en la región ese día?
    df["fire_regional"] = df.groupby(["parent7", df["fecha"].dt.normalize().rename("d")])["big10"].transform("max")

    # features regionales por (región, día)
    def agg_region(g):
        p = g["p_cell"].to_numpy()
        high = (p > np.nanquantile(p, 0.90)).astype(float) if len(p) > 1 else np.array([0.0])
        return pd.Series({
            "p_max": np.nanmax(p) if len(p) else 0,
            "p_mean": np.nanmean(p) if len(p) else 0,
            "p_top3": np.nanmean(np.sort(p)[-3:]) if len(p) >= 3 else np.nanmean(p),
            "p_std": np.nanstd(p) if len(p) > 1 else 0,
            "n_cells_high": (p > 0.05).sum(),
            "frac_high": (p > 0.05).mean() if len(p) else 0,
            "spatial_coherence": high.mean() if len(p) else 0,
            "n_cells": len(p),
        })

    reg = df.groupby(["parent7", df["fecha"].dt.normalize().rename("fecha")]).apply(agg_region).reset_index()
    reg.columns = ["parent7", "fecha"] + [c for c in reg.columns if c not in ("parent7", "fecha")]
    # etiqueta regional
    reg_lab = df.groupby(["parent7", df["fecha"].dt.normalize().rename("fecha")])["fire_regional"].max().reset_index()
    reg_lab.columns = ["parent7", "fecha", "fire_regional"]
    reg = reg.merge(reg_lab, on=["parent7", "fecha"], how="left")
    reg["fire_regional"] = reg["fire_regional"].fillna(0).astype(int)

    tr_r = (pd.to_datetime(reg["fecha"]).dt.year <= 2022).to_numpy()
    te_r = (pd.to_datetime(reg["fecha"]).dt.year >= 2023).to_numpy()
    y_reg = reg["fire_regional"].to_numpy()

    REG_FEATS = ["p_max", "p_mean", "p_top3", "p_std", "n_cells_high", "frac_high", "spatial_coherence"]
    Xr = reg[REG_FEATS].to_numpy()

    # modelo regional (LGBM simple)
    spw_r = max((y_reg[tr_r] == 0).sum() / max((y_reg[tr_r] == 1).sum(), 1), 0.1)
    m_reg = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               min_child_samples=30, scale_pos_weight=spw_r,
                               random_state=42, verbosity=-1)
    m_reg.fit(Xr[tr_r], y_reg[tr_r])
    p_reg = m_reg.predict_proba(Xr[te_r])[:, 1]
    y_reg_te = y_reg[te_r]

    # también: solo max(p_cell) como baseline regional (sin modelo)
    p_max_only = reg.loc[te_r, "p_max"].to_numpy()

    def met(p, tag):
        ap = average_precision_score(y_reg_te, p)
        roc = roc_auc_score(y_reg_te, p)
        br = brier_score_loss(y_reg_te, p)
        o = np.argsort(-p); k5 = max(int(len(p) * .05), 1)
        r5 = y_reg_te[o[:k5]].sum() / max(y_reg_te.sum(), 1)
        print(f"{tag:40s} AP={ap:.4f} ({ap/max(y_reg_te.mean()+1e-9,1e-9):.0f}× azar) · "
              f"ROC={roc:.4f} · rec@5%={r5:.0%}")
        return {"pr_auc": round(ap, 4), "lift": round(ap / max(y_reg_te.mean() + 1e-9, 1e-9), 1),
                "roc_auc": round(roc, 4), "recall_top5": round(r5, 3), "brier": round(br, 4)}

    prev_reg = y_reg_te.mean()
    print(f"\n=== REGIONAL (res7 parent, {reg.loc[te_r, 'parent7'].nunique()} regiones · "
          f"prev {prev_reg:.1%}) ===")
    r_max = met(p_max_only, "baseline: max(p_cell) en la región")
    r_model = met(p_reg, "modelo regional (LGBM sobre features regionales)")

    out = {
        "prev_regional_test": round(float(prev_reg), 4),
        "n_regions_test": int(reg.loc[te_r, "parent7"].nunique()),
        "baseline_max_p": r_max,
        "modelo_regional": r_model,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")
    return out


if __name__ == "__main__":
    main()
