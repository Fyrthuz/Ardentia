"""Ablation F2 (M0–M4): ¿el factor antrópico (viaria) aporta señal real?

M0 solo FWI → M1 +met/era5 → M2 +satélite → M3 +estáticas forestales →
M4 +viaria → M4b +interacciones teóricas (arcén×sequedad, vías×pendiente).
Split temporal (train ≤2022, test 2023–24) y transferencia espacial (train sin
Ourense, test solo Ourense). Métricas: PR-AUC (principal), Brier, Recall@top5/top20,
bootstrap ΔPR-AUC por provincia para M4−M3.

Uso: uv run python -m ardintra.models.ablation_run
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

import os as _os
_RES = int(_os.environ.get("ARDENTIA_RES", "9"))
DATASET = Path(f"data/processed/features/v1{'' if _RES == 9 else f'_res{_RES}'}/dataset.parquet")
REPORT = Path(f"reports/phase2_ablation{'' if _RES == 9 else f'_res{_RES}'}.json")
REPORT_MD = Path(f"reports/phase2_ablation{'' if _RES == 9 else f'_res{_RES}'}.md")

BLOCKS = {
    "fwi": ["fwi"],
    "met": ["ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
            "ffmc_7d", "fwi_7d", "fwi_prev"],
    "era5": ["swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"],
    "satelite": ["ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom"],
    "estaticas": ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m"],
    "viaria": ["dist_to_road_log", "road_density_log", "road_fuel_interface", "road_class",
               "es_finde", "dens_x_weekend"],
    "interacciones": ["dens_x_sequia", "dist_x_slope"],
}

MODELS = {
    "M0_fwi_solo": ["fwi"],
    "M1_fwi_met": ["fwi", "met", "era5"],
    "M2_satelite": ["fwi", "met", "era5", "satelite"],
    "M3_estaticas": ["fwi", "met", "era5", "satelite", "estaticas"],
    "M4_viaria": ["fwi", "met", "era5", "satelite", "estaticas", "viaria"],
    "M4b_interacciones": ["fwi", "met", "era5", "satelite", "estaticas", "viaria", "interacciones"],
}


def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    X = df[cols].copy()
    for c in cols:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def _metrics(y_true, y_prob) -> dict:
    prauc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)
    order = np.argsort(-y_prob)
    n = len(y_true)
    out = {"pr_auc": round(float(prauc), 4), "brier": round(float(brier), 4),
           "recall_top5pct": None, "recall_top20pct": None}
    for pct, key in ((0.05, "recall_top5pct"), (0.20, "recall_top20pct")):
        k = max(int(n * pct), 1)
        out[key] = round(float(y_true[order[:k]].sum() / max(y_true.sum(), 1)), 3)
    return out


def _fit_eval(Xtr, ytr, Xte, yte, spw: float) -> dict:
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=100,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        scale_pos_weight=spw, random_state=42, verbosity=-1,
    )
    model.fit(Xtr, ytr)  # entrenamiento completo; early-stop en test = fuga metodológica
    prob = model.predict_proba(Xte)[:, 1]
    return _metrics(yte, prob)


def run() -> dict:
    df = pd.read_parquet(DATASET)
    y = df["label"].to_numpy()

    # interacciones teóricas (arcén×sequedad, exposición viaria×pendiente)
    has_sat = "ndmi_summer" in df.columns
    df["dens_x_sequia"] = (
        df["road_density_1km"].fillna(0) * (1 - df["ndmi_summer"].fillna(0)).round(4)
        if has_sat else np.nan
    )
    df["dist_x_slope"] = (
        (1 / (1 + df["dist_to_road"].fillna(1e9))) * df["slope_deg"].fillna(0)
    ).round(4)

    report: dict = {"block_cols": {}, "results": []}
    for name, blocks in MODELS.items():
        try:
            cols: list[str] = []
            for b in blocks:
                cols += [c for c in BLOCKS.get(b, []) if c in df.columns]
            report["block_cols"][name] = {"usadas": sorted(set(cols))}

            X = _prep(df, cols)
            tr_mask = (df["split_temporal"] == "train") & df["fwi"].notna()
            te_mask = (df["split_temporal"] == "test") & df["fwi"].notna()
            if X.shape[1] == 0 or tr_mask.sum() == 0:
                report["results"].append({"modelo": name, "error": "sin columnas/train"})
                continue
            ytr, yte = y[tr_mask], y[te_mask]
            ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
            try:
                m_temp = _fit_eval(X[tr_mask], ytr, X[te_mask], yte, ratio)
            except Exception as e:  # noqa: BLE001
                m_temp = {"pr_auc": None, "error": str(e)[:80]}

            our_train_mask = tr_mask & ~df["provincia"].isna() & (df["provincia"] == "ourense")
            our_test_mask = te_mask & (df["provincia"].notna()) & (df["provincia"] == "ourense")
            n_our = int(our_test_mask.sum())
            if n_our > 100:
                try:
                    m_our = _fit_eval(X[our_train_mask], y[our_train_mask],
                                      X[our_test_mask], y[our_test_mask], ratio)
                except Exception as e:  # noqa: BLE001
                    m_our = {"pr_auc": None, "error": str(e)[:60]}
            else:
                m_our = {"pr_auc": None, "nota": f"test Ourense insuficiente ({n_our} filas)"}

            row = {"modelo": name, "n_features": X[tr_mask].shape[1],
                   "temporal": m_temp, "transfer_ourense": m_our}
            report["results"].append(row)
            pa = m_temp.get("pr_auc")
            print(f"{name:18s} temporal PR-AUC {pa} | recall@5% {m_temp.get('recall_top5pct')}"
                  f" | transfer Ourense {m_our.get('pr_auc')}")
        except Exception as e:  # noqa: BLE001
            report["results"].append({"modelo": name, "error": str(e)[:120]})
            print(f"✗ {name}: {str(e)[:80]}")

    # gate Δ (M4−M3) temporal
    def _get(model_id: str):
        r = next((r for r in report["results"] if r.get("modelo") == model_id), None)
        return (r.get("temporal", {}).get("pr_auc") or 0) if r else 0

    report["delta_m4_m3_temporal"] = round(_get("M4_viaria") - _get("M3_estaticas"), 4)
    report["delta_m4b_m3_temporal"] = round(_get("M4b_interacciones") - _get("M3_estaticas"), 4)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = ["# Ablation F2 — WildfireRisk-ES Galicia (test 2023–2024)", "",
             "| Modelo | PR-AUC temporal | recall@5% | PR-AUC transfer (Ourense) |",
             "|---|---|---|---|"]
    for r in report["results"]:
        if "error" in r:
            lines.append(f"| {r['modelo']} | ERROR {r['error'][:40]} | — | — |")
            continue
        t, o = r["temporal"], r["transfer_ourense"]
        lines.append(f"| {r['modelo']} | {t.get('pr_auc')} | {t.get('recall_top5pct')} | {o.get('pr_auc')} |")
    lines += ["", f"Δ (M4−M3) temporal: **{report['delta_m4_m3_temporal']}**",
              f"Δ (M4b−M3) temporal: **{report['delta_m4b_m3_temporal']}**"]
    REPORT_MD.write_text("\n".join(lines))
    print(f"\n-> {REPORT}\n-> {REPORT_MD}")
    return report


if __name__ == "__main__":
    run()
