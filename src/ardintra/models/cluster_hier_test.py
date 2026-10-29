"""Clusters de región (estáticas) + JERÁRQUICO regional→fino: experimento combinado.

Pipeline del usuario: 1º clustering de celdas por estáticas (terreno/fuel/viaria),
2º modelo local POR REGIÓN, 3º fusionado con el nivel regional (parent-day) de la
prueba jerárquica. Target: ≥10 ha. Splits: train ≤2021 · val 2022 (no usado) · test 23-24.

Uso: uv run python -m ardintra.models.cluster_hier_test [k=6]
"""

from __future__ import annotations

import sys

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

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
STATIC_FOR_CLUSTER = [
    "forest_frac", "fuel_model", "elev_m", "slope_deg", "aspect_sin", "aspect_cos",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
]
MIN_CHILD = 100


def prep(df: pd.DataFrame):
    X = df[FEATS].copy()
    for c in FEATS:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def fit_lgbm(d, y, tr, min_child=MIN_CHILD):
    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    m = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=min_child,
        scale_pos_weight=max(spw, 0.1), random_state=42, verbosity=-1,
    )
    m.fit(prep(d)[tr], y[tr])
    return m


def main(k: int = 6):
    df = pd.read_parquet("data/processed/features/v1/dataset.parquet")
    statics = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
    import numpy as np

    if "aspect_sin" not in statics.columns:
        statics["aspect_sin"] = np.sin(np.radians(statics["aspect_deg"]))
        statics["aspect_cos"] = np.cos(np.radians(statics["aspect_deg"]))

    xs = StandardScaler().fit_transform(
        statics[STATIC_FOR_CLUSTER].fillna(statics[STATIC_FOR_CLUSTER].median()))
    statics["region"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(xs)
    df = df.merge(statics[["h9", "region"]], on="h9", how="left")
    df["region"] = df["region"].fillna(-1).astype(int)

    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].copy()

    tr = (df["fecha"].dt.year <= 2022).to_numpy()
    te = (df["fecha"].dt.year >= 2023).to_numpy()
    y = df["big"].to_numpy()

    print(f"dataset {len(df):,} filas · regiones KMeans k={k}")

    # ---- nivel fino GLOBAL (baseline jerárquico)
    m_fine = fit_lgbm(df, y, tr)
    df["p_fine"] = m_fine.predict_proba(prep(df))[:, 1]

    # ---- regional parent-day (del jerárquico)
    df["parent8"] = [h3.cell_to_parent(x, 8) for x in df["h9"]]
    fe = df["fecha"].dt.normalize()
    d9 = d = df  # alias
    agg = d.groupby(["parent8", fe.rename("fecha")]).agg(
        {**{c: "mean" for c in FEATS_NUM},
         "road_class": lambda s: s.mode().iat[0] if len(s.mode()) else "local",
         "big": "max"}).reset_index()
    tr8 = (pd.to_datetime(agg["fecha"]).dt.year <= 2022).to_numpy()
    y8 = agg["big"].to_numpy()
    m_reg = fit_lgbm(agg, y8, tr8, min_child=50)
    main_agg = agg.copy()
    df["p_regional"] = np.nan
    key_map = None

    # ---- modelos POR REGIÓN (fino local)
    df["p_local"] = np.nan
    for k_i in range(k):
        m_tr = tr & (df["region"].to_numpy() == k_i)
        if m_tr.sum() < 300 or y[m_tr].sum() < 3:
            continue
        m_k = fit_lgbm(df, y, m_tr, min_child=min(100, max(10, int(m_tr.sum() * 0.002))))
        m_te = (df["region"].to_numpy() == k_i)
        df.loc[m_te, "p_local"] = m_k.predict_proba(prep(df)[m_te])[:, 1]
        print(f"  región {k_i}: train {m_tr.sum():,} · entrenada ✓")

    dte = df[te].copy()
    yte = dte["big"].to_numpy()

    def rep(tag, sc, valid=None):
        sc = np.asarray(sc, float)
        if valid is not None:
            sc_, y_ = sc[valid], yte[valid]
        else:
            sc_, y_ = sc, yte
        ap = average_precision_score(y_, sc_)
        o = np.argsort(-sc_); kk = max(int(len(sc_) * .05), 1)
        r5 = y_[o[:kk]].sum() / max(y_.sum(), 1)
        print(f"{tag:44s} AP={ap:.3f} ({ap / y_.mean():5.1f}× azar) · recall@5%={r5:.0%}")

    print(f"\n=== RESULTADOS (target ≥10 ha · test {len(dte):,}) ===")
    rep("A) fino global SOLO", dte["p_fine"].to_numpy())
    dv = dte["p_local"].notna().to_numpy()
    rep("B) cluster-local SOLO (regiones con modelo)", dte["p_local"].to_numpy(), dv)
    # regional: parent-day predicciones sobre las gradeadas d9test: mapeo índice agg→df
    # el regional HA sido entrenado sobre la agg completa; aplica a cada fila test por parent/fecha
    key = agg.set_index(["parent8", "fecha"])["p_regional"] if "p_regional" in agg else None

    # regional: predict directo sobre el dataset de test remapeado a features parent ya medias
    dte_agg = dte.groupby(["parent8", dte["fecha"].dt.normalize()]).agg(
        {**{c: "mean" for c in FEATS_NUM},
         "road_class": lambda s: s.mode().iat[0] if len(s.mode()) else "local"}).reset_index()
    dte_agg.columns = ["parent8", "fecha"] + FEATS_NUM + ["road_class"]
    p_reg_bykey = m_reg.predict_proba(prep(dte_agg))[:, 1]
    dte_agg["p_regional"] = p_reg_bykey
    reg_map = dte_agg.set_index(["parent8", "fecha"])["p_regional"]
    dte_reg = dte.copy()
    dte_reg["fecha_n"] = dte_reg["fecha"].dt.normalize()
    dte_reg["p_regional"] = [reg_map.get((p, f), np.nan) for p, f in
                             zip(dte_reg["parent8"], dte_reg["fecha_n"])]

    rep("C) regional SOLO (parent-day completo)", dte_reg["p_regional"].to_numpy())

    # D) fusion fino local + regional (rank-avg, NaN→mediana de rank para no expulsar)
    pl = dte_reg["p_local"].fillna(dte_reg["p_fine"]).to_numpy()
    pr = dte_reg["p_regional"].fillna(np.nanmedian(dte_reg["p_regional"])).to_numpy()
    rep("D) FUSIÓN cluster-local + regional", (rankdata(pl) + rankdata(pr)) / 2)
    # E) con el fino global como referencia (cluster só añade si supera a A)
    rep("E) FUSIÓN fino_global + regional (jerárq. previa)", (rankdata(dte_reg["p_fine"]) + rankdata(pr)) / 2)


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(k)
