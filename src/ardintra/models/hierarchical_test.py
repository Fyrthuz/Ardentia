"""Test jerárquico: fino res9 + regional res8 (parent-day agregado desde res9).

Variante A) fino SOLO · B) regional SOLO · C) fusión rank-avg · D) fusión producto.
El modelo regional usa parent-day features = media de las celdas res9 hijas ese día
(cobertura COMPLETA por día: sin sesgo de muestreo).

Uso: uv run python -m ardintra.models.hierarchical_test
"""

from __future__ import annotations

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

FEATS_NUM = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend",
]
FEATS = FEATS_NUM + ["road_class"]


def prep(df: pd.DataFrame):
    X = df[FEATS].copy()
    for c in FEATS:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def fit_lgbm(df: pd.DataFrame, y: np.ndarray, tr: np.ndarray, min_child=100) -> lgb.LGBMClassifier:
    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    m = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=min_child,
        scale_pos_weight=max(spw, 0.1), random_state=42, verbosity=-1,
    )
    m.fit(prep(df)[tr], y[tr])
    return m


def main() -> None:
    d9 = pd.read_parquet("data/processed/features/v1/dataset.parquet")
    surf = pd.to_numeric(d9["surface_ha"], errors="coerce").fillna(0)
    d9["big"] = (d9["label"] & (surf >= 10)).astype(int)
    d9 = d9[d9["fwi"].notna()].copy()
    d9["parent8"] = [h3.cell_to_parent(x, 8) for x in d9["h9"]]

    # ---- modelo regional (parent-day) desde las filas res9 exactas
    fe = d9["fecha"].dt.normalize()
    agg = (
        d9.groupby(["parent8", fe.rename("fecha")])
        .agg({**{c: "mean" for c in FEATS_NUM},
              "road_class": lambda s: s.mode().iat[0] if len(s.mode()) else "local",
              "big": "max"})
        .reset_index()
    )
    print(f"parent-day dataset: {len(agg):,} filas · prev big {agg['big'].mean():.2%}")

    tr8 = (pd.to_datetime(agg["fecha"]).dt.year <= 2022).to_numpy()
    te8 = (pd.to_datetime(agg["fecha"]).dt.year >= 2023).to_numpy()
    y8 = agg["big"].to_numpy()
    m8 = fit_lgbm(agg, y8, tr8, min_child=50)
    agg["p8"] = np.nan
    agg.loc[te8, "p8"] = m8.predict_proba(prep(agg)[te8])[:, 1]
    print(f"regional (parent-day): test AP={average_precision_score(y8[te8], m8.predict_proba(prep(agg)[te8])[:, 1]):.3f}")

    # ---- modelo fino (res9)
    tr9 = (d9["fecha"].dt.year <= 2022).to_numpy()
    y9 = d9["big"].to_numpy()
    m9 = fit_lgbm(d9, y9, tr9, min_child=100)

    d9te = d9[d9["fecha"].dt.year >= 2023].copy()
    d9te["fecha_n"] = d9te["fecha"].dt.normalize()
    d9te["p9"] = m9.predict_proba(prep(d9te))[:, 1]

    key = agg.set_index(["parent8", "fecha"])["p8"]
    d9te["p8"] = [key.get((p, f), np.nan) for p, f in zip(d9te["parent8"], d9te["fecha_n"])]

    y = d9te["big"].to_numpy()
    def rep(tag, sc):
        sc = np.asarray(sc, float)
        ap = average_precision_score(y, sc)
        o = np.argsort(-sc)
        k = max(int(len(sc) * 0.05), 1)
        r5 = y[o[:k]].sum() / max(y.sum(), 1)
        print(f"{tag:38s} AP={ap:.3f} ({ap / y.mean():5.1f}× azar) · recall@5%={r5:.0%}")

    print("\n=== RESULTADOS JERÁRQUICOS (target ≥10 ha, test 2023-24) ===")
    rep("A) fino res9 SOLO", d9te["p9"].to_numpy())
    rep("B) regional res8 SOLO", d9te["p8"].to_numpy())
    rep("C) FUSIÓN rank-avg", (rankdata(d9te["p9"]) + rankdata(d9te["p8"])) / 2)
    rep("D) FUSIÓN producto p9×p8", d9te["p9"] * d9te["p8"])
    print(f"\nbase azar: {y.mean():.2%} · n test {len(y):,}")


if __name__ == "__main__":
    main()
