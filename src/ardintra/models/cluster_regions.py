"""Clusters de región por features estáticas + modelo por región.

Idea del usuario: 1º agrupar celdas por similitud de terreno/combustible/viaria
(KMeans sobre estáticas), luego entrenar un modelo POR CLÚSTER (dinámicas+estáticas
locales) y predecir con más detalle dentro de las regiones que saltan.

Comparativa:
  GLOBAL (1 modelo) vs PER-CLUSTER (k modelos) sobre test 2023-24 (target ≥10 ha).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

STATIC = "data/processed/h3_cells/res9_static.parquet"
DATASET = "data/processed/features/v1/dataset.parquet"

STATIC_FEATURES = [
    "forest_frac", "fuel_model", "elev_m", "slope_deg", "aspect_sin", "aspect_cos",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
]
FEATS = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend", "road_class",
]


def main(k_clusters: int = 6):
    df = pd.read_parquet(DATASET)
    import numpy as np
    statics = pd.read_parquet(STATIC)
    if "aspect_sin" not in statics.columns:
        statics["aspect_sin"] = np.sin(np.radians(statics["aspect_deg"]))
        statics["aspect_cos"] = np.cos(np.radians(statics["aspect_deg"]))
    xs = StandardScaler().fit_transform(
        statics[STATIC_FEATURES].fillna(statics[STATIC_FEATURES].median())
    )
    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    statics["region"] = km.fit_predict(xs)
    df = df.merge(statics[["h9", "region"]], on="h9", how="left")
    df["region"] = df["region"].fillna(-1).astype(int)

    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    d = df[df["fwi"].notna()].copy()

    tr = (d["fecha"].dt.year <= 2022).to_numpy()
    te = (d["fecha"].dt.year >= 2023).to_numpy()
    y = d["big"].to_numpy()

    def _prep(X):
        X = X.copy()
        for c in FEATS:
            if c not in X.columns:
                continue
            if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
                X[c] = X[c].astype("category")
        return X

    Xp = _prep(d[FEATS])

    def fit_score(indices_tr, indices_te, tag, return_model=False) -> dict:
        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=100,
            scale_pos_weight=max(((y[indices_tr] == 0).sum() / max((y[indices_tr] == 1).sum(), 1)), 0.2),
            random_state=42, verbosity=-1,
        )
        m.fit(Xp[indices_tr], y[indices_tr])
        p = m.predict_proba(Xp[indices_te])[:, 1]
        yt = y[indices_te]
        ap = average_precision_score(yt, p)
        o = np.argsort(-p)
        kk = max(int(len(o) * 0.05), 1)
        r5 = yt[o[:kk]].sum() / max(yt.sum(), 1)
        if return_model:
            return {"tag": tag, "n_test": int(len(o)), "pr_auc": round(ap, 4),
                    "recall_top5": round(r5, 3)}, m
        return {"tag": tag, "n_test": int(len(o)), "pr_auc": round(ap, 4),
                "recall_top5": round(r5, 3)}

    # global
    res_global = fit_score(tr, te, "GLOBAL (1 modelo)")
    print(res_global)

    # per-cluster: entrenar solo con sus filas train, evaluar en sus filas test
    per_rows = []
    p_test = np.full(len(d), np.nan)
    for k in range(k_clusters):
        m_tr = tr & (d["region"].to_numpy() == k)
        m_te = te & (d["region"].to_numpy() == k)
        if m_tr.sum() < 500 or y[m_tr].sum() < 5:
            print(f"  região {k}: skip ({m_tr.sum():,} train rows)")
            continue
        r, m_k = fit_score(m_tr, m_te, f"región {k}", return_model=True)
        p_test[m_te] = m_k.predict_proba(Xp[m_te])[:, 1]
        per_rows.append(r | {"n_train": int(m_tr.sum())})
        print(r)

    # métricaglobal per-cluster (predicciones mezcladas)
    ok = ~np.isnan(p_test)
    ap_mix = average_precision_score(y[ok & te], p_test[ok & te]) if (ok & te).sum() else np.nan
    res = {
        "k_clusters": k_clusters,
        "global": res_global,
        "per_region": per_rows,
        "per_cluster_global_ap": round(float(ap_mix), 4),
    }
    out = Path("reports/cluster_regions.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"\nout -> {out} · per-cluster AP (mezclado): {res['per_cluster_global_ap']} "
          f"vs global {res_global['pr_auc']}")
    return res


if __name__ == "__main__":
    main()
