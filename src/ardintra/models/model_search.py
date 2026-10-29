"""Búsqueda multi-modelo con Optuna (Fase 2.5): LGBM · XGB · CatBoost · HGB ·
RandomForest · ExtraTrees · MLP · Logística (baseline lineal).

Split metodológico SIN tocar el test en la búsqueda:
  train_inner 2017–2021 · val 2022 · test 2023–2024 · transfer Ourense.
El mejor de cada familia se re-ajusta con train_full (≤2022) y evalúa en test,
con PR-AUC + recall@5% + Brier + transferencia.

Uso:
  uv run python -m ardintra.models.model_search --fams lgbm,mlp --trials 12
  uv run python -m ardintra.models.model_search            # todo, 15 trials/familia
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATASET = Path("data/processed/features/v1/dataset.parquet")
OUT_JSON = Path("reports/model_search.json")
OUT_MD = Path("reports/model_search.md")

COLS_NUM = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_deg", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "ffmc_7d", "fwi_7d", "fwi_prev", "es_finde", "dens_x_weekend",
    "dens_x_sequia", "dist_x_slope",
]
COL_CAT = ["road_class"]


def load_xy():
    df = pd.read_parquet(DATASET)
    if "dens_x_sequia" not in df.columns:
        df["dens_x_sequia"] = df["road_density_1km"].fillna(0) * (1 - df["ndmi_summer"].fillna(0)).round(4)
        df["dist_x_slope"] = ((1 / (1 + df["dist_to_road"].fillna(1e9))) * df["slope_deg"].fillna(0)).round(4)
    y = df["label"].to_numpy().astype(int)
    X = pd.concat([df[COLS_NUM].copy(), df[COL_CAT].copy().astype("category")], axis=1)
    return df, X, y


def _masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    yi = df["fecha"].dt.year
    ok = df["fwi"].notna()
    return {
        "train": ((yi <= 2021) & ok).to_numpy(),
        "val": ((yi == 2022) & ok).to_numpy(),
        "full": ((yi <= 2022) & ok).to_numpy(),
        "test": ((yi >= 2023) & ok).to_numpy(),
    }


def _cat_idx(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if str(X[c].dtype) == "category"]


# ---------------------------------------------------------------- build dicts
def build_lgbm(p: dict):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=p["n_est"], learning_rate=p["lr"], num_leaves=p["leaves"],
        min_child_samples=p["min_child"], subsample=p["subsample"], subsample_freq=1,
        colsample_bytree=p["colsample"], reg_lambda=p["reg_l"],
        scale_pos_weight=p["spw"], random_state=42, verbosity=-1,
    )


def grid_lgbm(t):
    return {
        "n_est": t.suggest_int("n_est", 200, 1400, step=100),
        "lr": t.suggest_float("lr", 0.01, 0.2, log=True),
        "leaves": t.suggest_int("leaves", 16, 128),
        "min_child": t.suggest_int("min_child", 20, 250),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample": t.suggest_float("colsample", 0.5, 1.0),
        "reg_l": t.suggest_float("reg_l", 1e-3, 10, log=True),
        "spw": t.suggest_float("spw", 5, 30),
    }


def build_xgb(p: dict):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=p["n_est"], learning_rate=p["lr"], max_depth=p["depth"],
        min_child_weight=p["mcw"], subsample=p["subsample"], colsample_bytree=p["colsample"],
        reg_lambda=p["reg_l"], scale_pos_weight=p["spw"],
        tree_method="hist", enable_categorical=True, random_state=42, verbosity=0,
    )


def grid_xgb(t):
    return {
        "n_est": t.suggest_int("n_est", 200, 1400, step=100),
        "lr": t.suggest_float("lr", 0.01, 0.2, log=True),
        "depth": t.suggest_int("depth", 3, 12),
        "mcw": t.suggest_float("mcw", 1, 100, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample": t.suggest_float("colsample", 0.5, 1.0),
        "reg_l": t.suggest_float("reg_l", 1e-3, 10, log=True),
        "spw": t.suggest_float("spw", 5, 30),
    }


def build_cat(p: dict):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=p["iters"], learning_rate=p["lr"], depth=p["depth"],
        l2_leaf_reg=p["l2"], auto_class_weights="Balanced",
        random_seed=42, verbose=0, allow_writing_files=False,
    )


def grid_cat(t):
    return {
        "iters": t.suggest_int("iters", 300, 1200, step=100),
        "lr": t.suggest_float("lr", 0.01, 0.2, log=True),
        "depth": t.suggest_int("depth", 4, 10),
        "l2": t.suggest_float("l2", 1, 30, log=True),
    }


def build_hgb(p: dict):
    return HistGradientBoostingClassifier(
        max_iter=p["iters"], learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
        min_samples_leaf=p["min_leaf"], l2_regularization=p["l2"],
        categorical_features="from_dtype", class_weight="balanced", random_state=42,
    )


def grid_hgb(t):
    return {
        "iters": t.suggest_int("iters", 200, 1000, step=100),
        "lr": t.suggest_float("lr", 0.01, 0.2, log=True),
        "leaves": t.suggest_int("leaves", 15, 127),
        "min_leaf": t.suggest_int("min_leaf", 20, 200),
        "l2": t.suggest_float("l2", 0.0, 5.0),
    }


def build_rf(p: dict, et: bool):
    ctr = ExtraTreesClassifier if et else RandomForestClassifier
    return ctr(
        n_estimators=p["n_est"], max_depth=p["depth"], min_samples_leaf=p["min_leaf"],
        max_features=p["mf"], class_weight="balanced", n_jobs=-1, random_state=42,
    )


def grid_rf_et(t, et: bool):
    p = {
        "n_est": t.suggest_int("n_est", 200, 800, step=100),
        "depth": t.suggest_int("depth", 5, 30),
        "min_leaf": t.suggest_int("min_leaf", 1, 50),
        "mf": t.suggest_float("mf", 0.2, 0.8),
    }
    p["is_et"] = et
    return p


def build_mlp(p: dict, Xtr: pd.DataFrame, ytr: np.ndarray):
    from sklearn.pipeline import Pipeline

    if "hidden" not in p:  # best_params guarda sugiere por unidad: h0/h1
        p = dict(p, hidden=(int(p.get("h0", 64)), int(p.get("h1", 32))))

    # muestreo 1:3 (MLP se hunde con 1:20): subsample determinista
    idx_pos = np.where(ytr == 1)[0]
    idx_neg = np.where(ytr == 0)[0]
    rng = np.random.default_rng(42)
    neg_keep = rng.choice(idx_neg, size=min(len(idx_neg), 3 * len(idx_pos)), replace=False)
    sel = np.concatenate([idx_pos, neg_keep])

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=p["hidden"], alpha=p["alpha"],
            learning_rate_init=p["lr_init"], batch_size=512, max_iter=120,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=8,
            random_state=42,
        )),
    ])
    pipe.fit(Xtr.iloc[sel], ytr[sel])
    return pipe


def grid_mlp(t):
    return {
        "hidden": tuple(t.suggest_int(f"h{i}", 16, 128, step=16) for i in range(2)),
        "alpha": t.suggest_float("alpha", 1e-5, 1.0, log=True),
        "lr_init": t.suggest_float("lr_init", 1e-4, 1e-2, log=True),
    }


def build_lr(p: dict):
    return LogisticRegression(
        C=p["C"], penalty="elasticnet", l1_ratio=p["l1ratio"],
        solver="saga", max_iter=2000, class_weight="balanced", random_state=42,
    )


def grid_lr(t):
    return {"C": t.suggest_float("C", 1e-3, 10, log=True),
            "l1ratio": t.suggest_float("l1ratio", 0.0, 1.0)}


if __name__ == "__main__":
    pass


def run_family(name: str, df, X, y, masks, n_trials: int, report: dict) -> None:
    import optuna

    tr, val, full, test = masks["train"], masks["val"], masks["full"], masks["test"]

    Xf = None
    if name == "lr":
        Xf = X.drop(columns=COL_CAT).copy()
        Xf = Xf.fillna(Xf.median(numeric_only=True))
    elif name == "mlp":
        Xf = X.drop(columns=COL_CAT).copy()
        med = Xf.median(numeric_only=True)
        Xf = Xf.fillna(med).fillna(0)
        Xf = Xf.fillna(Xf.median(numeric_only=True))  # segunda pasada por columnas all-NaN
        Xf = Xf.fillna(0)
    elif name in ("rf", "et", "hgb"):
        Xf = X.copy()
        Xf[COL_CAT] = Xf[COL_CAT].apply(lambda s: s.cat.codes)
        Xf = Xf.fillna(-999)
    else:
        Xf = X.copy()

    if name == "lr":
        def _fit(params, Xm, ym):
            scaler = StandardScaler().fit(Xm.to_numpy())
            Xs = np.nan_to_num(scaler.transform(Xm.to_numpy()))
            m = build_lr(params)
            m.fit(Xs, ym)
            return m, scaler
        def _proba(params, m, scaler, Xm):
            return m.predict_proba(np.nan_to_num(scaler.transform(Xm.to_numpy())))[:, 1]
        def objective(t):
            params = grid_lr(t)
            m, scaler = _fit(params, Xf[tr], y[tr])
            return average_precision_score(y[val], _proba(params, m, scaler, Xf[val]))
        def final(params):
            m, scaler = _fit(params, Xf[full], y[full])
            our = df["provincia"].fillna("").astype(str).eq("ourense")
            return (m, scaler, _proba(params, m, scaler, Xf[test]),
                    _proba(params, m, scaler, Xf[test & our]), None)
    else:
        builders = {"lgbm": build_lgbm, "xgb": build_xgb, "catboost": build_cat,
                    "hgb": build_hgb, "rf": lambda p: build_rf(p, et=False),
                    "et": lambda p: build_rf(p, et=True)}
        grids = {"lgbm": grid_lgbm, "xgb": grid_xgb, "catboost": grid_cat,
                 "hgb": grid_hgb, "rf": lambda t: grid_rf_et(t, et=False),
                 "et": lambda t: grid_rf_et(t, et=True)}

        def _fit_fam(params, Xm, ym):
            m = builders[name](params)
            if name == "catboost":
                Xm = Xm.copy()
                for c in COL_CAT:
                    Xm[c] = Xm[c].astype(str)
                params["_cat_cols"] = COL_CAT
                m.fit(Xm, ym, cat_features=COL_CAT)
            else:
                m.fit(Xm, ym)
            return m

        if name == "mlp":
            def objective(t):
                params = grid_mlp(t)
                pipe = build_mlp(params, Xf[tr], y[tr])
                return average_precision_score(y[val], pipe.predict_proba(Xf[val])[:, 1])
            def final(params):
                pipe = build_mlp(params, Xf[full], y[full])
                our = df["provincia"].fillna("").astype(str).eq("ourense")
                return (pipe, None, pipe.predict_proba(Xf[test])[:, 1],
                        pipe.predict_proba(Xf[test & our])[:, 1], pipe)
        else:
            def objective(t):
                m = _fit_fam(grids[name](t), Xf[tr], y[tr])
                return average_precision_score(y[val], m.predict_proba(Xf[val])[:, 1])
            def final(params):
                m = _fit_fam(params, Xf[full], y[full])
                our = df["provincia"].fillna("").astype(str).eq("ourense")
                return (m, None, m.predict_proba(Xf[test])[:, 1],
                        m.predict_proba(Xf[test & our])[:, 1], m)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    model, scaler, p_test, p_our, obj = final(best)
    yte = y[test]
    our = df["provincia"].fillna("").astype(str).eq("ourense")
    y_our = y[test & our]

    res = {
        "best_params": best,
        "val_pr_auc": round(float(study.best_value), 4),
        "test_pr_auc": round(float(average_precision_score(yte, p_test)), 4),
        "brier": round(float(brier_score_loss(yte, p_test)), 4),
        "recall_top5pct": round(float(yte[np.argsort(-p_test)[: max(int(len(p_test) * .05), 1)]].sum())
                                / max(float(yte.sum()), 1), 3),
        "transfer_ourense_pr_auc": (
            round(float(average_precision_score(y_our, p_our)), 4)
            if len(np.unique(y_our)) > 1 and len(y_our) > 50 else None
        ),
    }
    report["familias"][name] = res
    print(f"{name:10s} val={res['val_pr_auc']} test={res['test_pr_auc']} "
          f"recall@5%={res['recall_top5pct']} transfer={res['transfer_ourense_pr_auc']}", flush=True)


def main(fams: list[str], n_trials: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df, X, y = load_xy()
    masks = _masks(df)
    print(f"dataset: {len(df):,} filas · features {X.shape[1]} | train {masks['train'].sum():,} · "
          f"val {masks['val'].sum():,} · test {masks['test'].sum():,}", flush=True)

    import io
    prev = {}
    if OUT_JSON.exists():
        prev = json.load(open(OUT_JSON)).get("familias", {})
    report: dict = {"familias": dict(prev), "info": {"n_trials": n_trials, "fams": fams}}
    for name in fams:
        t0 = time.time()
        try:
            run_family(name, df, X, y, masks, n_trials, report)
        except Exception as e:  # noqa: BLE001
            report["familias"][name] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
            print(f"✗ {name}: {str(e)[:90]}", flush=True)
        report["familias"][name]["segundos"] = round(time.time() - t0, 1)
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    lines = ["# Búsqueda multi-modelo Optuna (val 2022 · test 2023–24)", "",
             "| Familia | PR-AUC val | PR-AUC test | recall@5% | transfer | s |",
             "|---|---|---|---|---|---|"]
    for n, r in report["familias"].items():
        if "error" in r:
            lines.append(f"| {n} | ERROR {r['error'][:34]} | — | — | — |")
            continue
        lines.append(f"| {n} | {r['val_pr_auc']} | {r['test_pr_auc']} | {r['recall_top5pct']} |"
                     f" {r['transfer_ourense_pr_auc']} | {r['segundos']} |")
    OUT_MD.write_text("\n".join(lines))
    print(f"\n-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", type=str, default="lgbm,xgb,catboost,hgb,rf,et,mlp,lr")
    ap.add_argument("--trials", type=int, default=15)
    a = ap.parse_args()
    main([s.strip() for s in a.fams.split(",") if s.strip()], a.trials)
