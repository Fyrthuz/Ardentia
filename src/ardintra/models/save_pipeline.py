"""Guarda los artefactos del pipeline de producción para la webapp:
  - models/lgbm_labels.joblib  (ya existe tras distill_student)
  - models/iso_calibrator.joblib (aux ≤2021 + isotónica ajustada en 2022)

Uso: uv run python -m ardintra.models.save_pipeline
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

DATA = Path("data/processed/features/v1/dataset.parquet")
MODELS = Path("models")

DYN = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
       "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
       "ffmc_anom", "dmc_anom", "dc_anom", "fwi_7d", "fwi_prev", "ffmc_7d",
       "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
       "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an", "es_finde",
       "dens_x_weekend", "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
       "lat_f", "lon_f", "doy_sin", "doy_cos"]
STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
          "dist_to_road_log", "road_density_log", "road_fuel_interface", "road_class"]
FEATS = STATIC + DYN


def main():
    df = pd.read_parquet(DATA)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True)
    y = df["big10"].to_numpy()
    years = df["fecha"].dt.year.to_numpy()

    Z = df[FEATS].copy()
    cat_maps = {}
    for c in Z.columns:
        if not (pd.api.types.is_numeric_dtype(Z[c]) or pd.api.types.is_bool_dtype(Z[c])):
            cat_maps[c] = {v: i for i, v in enumerate(pd.Categorical(Z[c]).categories)}
            Z[c] = pd.Categorical(Z[c], categories=list(cat_maps[c])).codes.astype("float64")
    Z = Z.astype("float64")

    m21, m22 = years <= 2021, years == 2022
    spw = max((y[m21] == 0).sum() / max((y[m21] == 1).sum(), 1), 0.1)
    aux = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                             min_child_samples=100, scale_pos_weight=spw,
                             random_state=42, verbosity=-1)
    aux.fit(Z[m21], y[m21])
    raw22 = aux.predict_proba(Z[m22])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw22, y[m22])
    brier_raw = float(np.mean((raw22 - y[m22]) ** 2))
    brier_cal = float(np.mean((iso.predict(raw22) - y[m22]) ** 2))
    print(f"Brier 2022: crudo {brier_raw:.6f} → calibrado {brier_cal:.6f}")

    MODELS.mkdir(exist_ok=True)
    if not (MODELS / "lgbm_labels.joblib").exists():
        pack = {"model": aux, "feats": FEATS, "cat_maps": cat_maps}
        joblib.dump(pack, MODELS / "lgbm_labels.joblib")
        print("models/lgbm_labels.joblib creado a partir del modelo auxiliar")
    joblib.dump(iso, MODELS / "iso_calibrator.joblib")
    print("-> models/iso_calibrator.joblib")


if __name__ == "__main__":
    main()
