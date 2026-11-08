"""Teacher 2: Kumo Tabular (NVIDIA, foundation model ICL) — experimento completo.

Pasos:
  A. Mismo eval subset que TabPFN (137 pos test + 20k neg): AUC comparable.
  B. DIAGNÓSTICO OOD: puntuar las 420k filas full-grid del set de destilación
     y comparar con la frecuencia real (y_true reconstruida desde EGIF).
     Si Kumo calibra mejor OOD que TabPFN, la destilación v2 será válida.
  C. Almacena kumo_p en el set de destilación para el student v2.

Uso: uv run python -m ardintra.models.kumo_teacher
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path("data/processed/features/v1/dataset.parquet")
DISTILL = Path("data/processed/features/v1/distill_set.parquet")
IG = Path("data/processed/labels/v1/ignitions.parquet")
OUT = Path("reports/kumo_teacher.json")
N_SUBSAMPLES = 6
SUBSIZE = 10_000
QUERY_CHUNK = 20_000
N_TEST_NEG = 20_000

DYN = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
       "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
       "ffmc_anom", "dmc_anom", "dc_anom", "fwi_7d", "fwi_prev", "ffmc_7d",
       "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
       "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
       "es_finde", "dens_x_weekend", "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
       "lat_f", "lon_f", "doy_sin", "doy_cos"]
STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
          "dist_to_road_log", "road_density_log", "road_fuel_interface", "road_class"]
FEATS = STATIC + DYN


def main():
    import torch
    import sdm

    dev = torch.device("cuda")

    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True)
    y_all = df["big10"].to_numpy()
    yr = df["fecha"].dt.year.to_numpy()
    tr = yr <= 2022
    X_all = df[FEATS].copy()
    for c in X_all.columns:
        if not (pd.api.types.is_numeric_dtype(X_all[c]) or pd.api.types.is_bool_dtype(X_all[c])):
            X_all[c] = pd.Categorical(X_all[c]).astype(str)

    rng = np.random.default_rng(42)
    pos_all = np.where(y_all == 1)[0]
    neg_all = np.where(y_all == 0)[0]

    def make_table(d: pd.DataFrame, yyy: np.ndarray | None):
        d = d.copy()
        if yyy is not None:
            d["big10"] = pd.Categorical(pd.Series(yyy).map({0: "0", 1: "1"})).astype(str)
        st = sdm.infer_stypes(d, overrides={"big10": "categorical"})
        return sdm.TableTensor.from_pandas(df=d, stypes=st, device=dev)

    def score(model, Xxx: pd.DataFrame, ctx_idx: np.ndarray, chunk: int = QUERY_CHUNK) -> np.ndarray:
        ctx_full = make_table(X_all.iloc[ctx_idx][FEATS], y_all[ctx_idx])
        ctx_x = ctx_full.drop_columns("big10")
        ctx_y = ctx_full[:, "big10"]
        out = np.zeros(len(Xxx))
        for s in range(0, len(Xxx), chunk):
            qt = make_table(Xxx.iloc[s:s + chunk][FEATS], None)
            with torch.amp.autocast(dev.type, dtype=torch.float16):
                probs = model(x_context=ctx_x, y_context=ctx_y,
                              x_query=qt, num_estimators=8)
            pdf = probs.to_pandas()
            out[s:s + len(pdf)] = pdf["1"].to_numpy()
            if (s + chunk) % 200_000 == 0 or s + chunk >= len(Xxx):
                print(f"    {s + len(pdf):,}/{len(Xxx):,}", flush=True)
            torch.cuda.empty_cache()
        return out

    model = sdm.models.KumoTabular(task="classification", device=dev)
    print("KumoTabular large cargado", flush=True)

    # --- A) eval subset estándar
    Xy = df[yr >= 2023].reset_index(drop=True)
    Xte = Xy[FEATS].copy()
    for c in Xte.columns:
        if not (pd.api.types.is_numeric_dtype(Xte[c]) or pd.api.types.is_bool_dtype(Xte[c])):
            Xte[c] = pd.Categorical(Xte[c]).astype(str)
    yte = Xy["big10"].to_numpy()
    pos_te = np.where(yte == 1)[0]
    neg_te = np.where(yte == 0)[0]
    eval_idx = np.sort(np.concatenate([pos_te, rng.choice(neg_te, size=N_TEST_NEG, replace=False)]))
    Xev, yev = Xte.iloc[eval_idx], yte[eval_idx]
    print(f"eval: {len(eval_idx):,} filas ({len(pos_te)} positivos)", flush=True)

    preds = []
    for k in range(N_SUBSAMPLES):
        neg_k = rng.choice(np.where(tr)[0][y_all[tr] == 0], size=SUBSIZE - (y_all[tr] == 1).sum(), replace=False)
        idx_k = np.concatenate([np.where(tr)[0][y_all[tr] == 1], neg_k]).astype(int)
        p = score(model, Xev, idx_k, chunk=min(QUERY_CHUNK, len(Xev)))
        auc_k = roc_auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yev, p)
        preds.append(p)
        print(f"  subsample {k + 1}/{N_SUBSAMPLES}: AUC={auc_k:.4f}", flush=True)
    p_avg = np.mean(preds, axis=0)
    res = {
        "kumo_teacher": {
            "auc": round(float(__import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(yev, p_avg)), 4),
            "ap": round(float(__import__("sklearn.metrics", fromlist=["x"]).average_precision_score(yev, p_avg)), 4),
            "subsamples": N_SUBSAMPLES, "subsize": SUBSIZE,
        },
        "ref_tabpfn": {"auc": 0.9387, "ap": 0.1653},
        "ref_lgbm": {"auc": 0.8887, "ap": 0.0922},
    }
    print("A:", json.dumps(res, indent=2), flush=True)

    # --- B) diagnóstico OOD: puntuar el set de destilación
    dist = pd.read_parquet(DISTILL)
    Xd = dist[FEATS].copy()
    for c in Xd.columns:
        if not (pd.api.types.is_numeric_dtype(Xd[c]) or pd.api.types.is_bool_dtype(Xd[c])):
            Xd[c] = pd.Categorical(Xd[c]).astype(str)
    print(f"B: scoring {len(Xd):,} filas full-grid...", flush=True)
    ctx_all = np.concatenate([np.where(tr)[0][y_all[tr] == 1],
                              rng.choice(np.where(tr)[0][y_all[tr] == 0],
                                         size=SUBSIZE - (y_all[tr] == 1).sum(), replace=False)]).astype(int)
    kumo_p = score(model, Xd, ctx_all)

    # y_true en filas full-grid: join EGIF por (h9, fecha)
    ig = pd.read_parquet(IG)
    ig["fecha"] = pd.to_datetime(ig["fecha"]).dt.normalize()
    ig["big"] = (ig["surface_ha"] >= 10).astype(int)
    lab = ig[["h9", "fecha", "big"]].groupby(["h9", "fecha"])["big"].max().reset_index()
    dist["fecha_n"] = pd.to_datetime(dist["fecha"]).dt.normalize()
    mj = dist.merge(lab, left_on=["h9", "fecha_n"], right_on=["h9", "fecha"], how="left")
    y_true = mj["big"].fillna(0).to_numpy()
    print(f"y_true en distill: {int(y_true.sum()):,} positivos de {len(y_true):,}")

    def reliability(p, y, bins):
        rows = []
        for i in range(len(bins) - 1):
            m = (p >= bins[i]) & (p < bins[i + 1])
            if m.sum() == 0:
                continue
            rows.append({"bin": f"{bins[i]:.2f}-{bins[i + 1]:.2f}", "n": int(m.sum()),
                         "fuegos": int(y[m].sum()),
                         "p_mediana": round(float(np.median(p[m])), 4),
                         "freq_real": round(float(y[m].mean()), 5)})
        return pd.DataFrame(rows)

    bins = [0, .01, .02, .05, .10, .20, .50, .80, 1.01]
    rel_kumo = reliability(kumo_p, y_true, bins)
    rel_tab = reliability(dist["teacher_p"].to_numpy(), y_true, bins)
    print("\n=== RELIABILITY FULL-GRID (y_true EGIF) ===")
    print("Kumo:")
    print(rel_kumo.to_string(index=False))
    print("TabPFN:")
    print(rel_tab.to_string(index=False))
    rel_kumo.to_csv("reports/kumo_reliability_fullgrid.csv", index=False)
    rel_tab.to_csv("reports/tabpfn_reliability_fullgrid.csv", index=False)

    # AUC en distill rows (OOD)
    from sklearn.metrics import roc_auc_score as _auc
    print(f"\nAUC OOD full-grid: Kumo={_auc(y_true, kumo_p):.4f} · "
          f"TabPFN={_auc(y_true, dist['teacher_p'].to_numpy()):.4f}")

    dist["kumo_p"] = kumo_p.round(6)
    dist.to_parquet(DISTILL, index=False)
    res["ood"] = {"auc_kumo": round(float(_auc(y_true, kumo_p)), 4),
                  "auc_tabpfn": round(float(_auc(y_true, dist["teacher_p"].to_numpy())), 4)}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
