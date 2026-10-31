"""Capa de predicción probabilística operativa:

1. Calibración isotónica: score LGBM -> probabilidad real de ignición ≥10 ha.
2. Agregación regional (Poisson): por hexágono padre res6,
   lambda = sum(p_i)  =>  P(>=1 fuego) = 1 - exp(-lambda).
3. Niveles de riesgo (verde/amarillo/naranja/rojo) por percentiles históricos.
4. Reliability diagram para verificar la calibración.

Uso: uv run python -m ardintra.models.calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT_DIR = Path("reports")
OUT_CAL = OUT_DIR / "calibration.json"
OUT_RELIABILITY = OUT_DIR / "reliability_diagram.csv"

FEATS_STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
                "dist_to_road_log", "road_density_log", "road_fuel_interface", "road_class"]
FEATS_DYN = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
             "prec_mm", "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
             "ffmc_anom", "dmc_anom", "dc_anom", "fwi_7d", "fwi_prev", "ffmc_7d",
             "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
             "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
             "es_finde", "dens_x_weekend",
             "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
             "lat_f", "lon_f", "doy_sin", "doy_cos"]


def load_dataset(target_ha: float = 10.0) -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big"] = ((df["label"] == 1) & (surf >= target_ha)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True)
    df["aspect_sin"] = np.sin(np.radians(df["aspect_deg"]))
    df["aspect_cos"] = np.cos(np.radians(df["aspect_deg"]))
    df["dist_to_road_log"] = np.log1p(df["dist_to_road"].clip(0, 5000))
    df["road_density_log"] = np.log1p(df["road_density_1km"])
    df["lat_f"] = df["lat"]
    df["lon_f"] = df["lon"]
    df["doy_sin"] = np.sin(2 * np.pi * df["fecha"].dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["fecha"].dt.dayofyear / 365)
    df["es_finde"] = (df["fecha"].dt.dayofweek >= 5).astype(int)
    df["dens_x_weekend"] = df["road_density_log"] * df["es_finde"]
    return df


def train_model(df: pd.DataFrame, year_max: int):
    feats = [c for c in FEATS_STATIC + FEATS_DYN if c in df.columns]
    X = df[feats].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    y = df["big"].to_numpy()
    mask = df["fecha"].dt.year.to_numpy() <= year_max
    spw = max((y[mask] == 0).sum() / max((y[mask] == 1).sum(), 1), 0.1)
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, scale_pos_weight=spw,
                               random_state=42, verbosity=-1)
    model.fit(X[mask], y[mask])
    return model, feats, X, y, mask


def fit_isotonic(df: pd.DataFrame, model, feats: str, cal_year: int = 2022):
    """Entrena en <=cal_year-1, calibra en cal_year."""
    X = df[feats].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    y = df["big"].to_numpy()
    years = df["fecha"].dt.year.to_numpy()
    m_tr, m_cal = years <= cal_year - 1, years == cal_year
    spw = max((y[m_tr] == 0).sum() / max((y[m_tr] == 1).sum(), 1), 0.1)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1)
    m.fit(X[m_tr], y[m_tr])
    raw_cal = m.predict_proba(X[m_cal])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-9, y_max=0.999)
    iso.fit(raw_cal, y[m_cal])
    return iso, raw_cal, y[m_cal]


def reliability_table(raw: np.ndarray, y_true: np.ndarray, iso, n_bins: int = 10) -> pd.DataFrame:
    p_cal = iso.predict(raw)
    bins = np.quantile(raw, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    rows = []
    for i in range(n_bins):
        m = (raw >= bins[i]) & (raw < bins[i + 1])
        if m.sum() < 50:
            continue
        rows.append({
            "bin": i,
            "n": int(m.sum()),
            "n_fuegos": int(y_true[m].sum()),
            "p_cruda": float(raw[m].mean()),
            "p_calibrada": float(p_cal[m].mean()),
            "frecuencia_real": float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)


def aggregate_regional(p_cell: np.ndarray, h9: np.ndarray, res: int = 6) -> pd.DataFrame:
    """Agrega probabilidades de celda a regiones (hexágonos padre H3).

    lambda = sum(p_i)  (esperanza de fuegos en la región)
    P(>=1) = 1 - exp(-lambda)  (aprox. Poisson)
    """
    parents = np.array([h3.cell_to_parent(c, res) for c in h9])
    df = pd.DataFrame({"region": parents, "p": p_cell})
    g = df.groupby("region")["p"].agg(lam="sum", n_celdas="count", p_max="max").reset_index()
    g["p_alguno"] = 1 - np.exp(-g["lam"])
    return g


def risk_levels(p_alguno: pd.Series, cal_p: np.ndarray) -> pd.Series:
    """Categorías por percentiles de la distribución histórica de P(>=1)."""
    q50, q90, q99 = np.quantile(cal_p, [0.50, 0.90, 0.99])
    def lvl(p):
        if p >= q99: return "rojo"
        if p >= q90: return "naranja"
        if p >= q50: return "amarillo"
        return "verde"
    return p_alguno.apply(lvl)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("cargando dataset...", flush=True)
    df = load_dataset()

    print("entrenando modelo de calibración (<=2021) y ajustando isotónica en 2022...", flush=True)
    iso, raw_cal, y_cal = fit_isotonic(df, None, FEATS_STATIC + FEATS_DYN, cal_year=2022)

    rel = reliability_table(raw_cal, y_cal, iso)
    rel.to_csv(OUT_RELIABILITY, index=False)
    print("\n=== RELIABILITY (2022) ===")
    print(rel.to_string(index=False))

    # Brier antes/después de calibrar
    brier_raw = brier_score_loss(y_cal, raw_cal)
    brier_cal = brier_score_loss(y_cal, iso.predict(raw_cal))
    print(f"\nBrier crudo: {brier_raw:.6f}  |  Brier calibrado: {brier_cal:.6f}")

    # Percentiles de P(>=1) regional en 2022 para los niveles de riesgo
    cal_df = df[df["fecha"].dt.year == 2022]
    p_cal_regional = iso.predict(raw_cal)
    lam_2022 = cal_df.assign(p=p_cal_regional).groupby("h9")["p"].sum()  # placeholder por celda
    # distribución regional de lambda por día en 2022 (aprox: media diaria por celda agregada)
    print(f"\nlambda medio por celda-año 2022: {lam_2022.mean():.4f}")

    meta = {
        "brier_raw": round(float(brier_raw), 6),
        "brier_calibrado": round(float(brier_cal), 6),
        "n_cal": int(len(y_cal)),
        "n_fuegos_cal": int(y_cal.sum()),
        "nota": "isotónica ajustada con modelo <=2021 sobre año 2022",
    }
    OUT_CAL.write_text(json.dumps(meta, indent=2))
    print(f"-> {OUT_CAL}")


if __name__ == "__main__":
    main()
