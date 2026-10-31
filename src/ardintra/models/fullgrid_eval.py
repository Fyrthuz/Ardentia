"""Evaluación operativa FULL-GRID: predice las 245k celdas para cada día de test
con fuegos ≥10 ha y calcula el recall operativo real (no el de la muestra).

Para cada día:
  1. Met IDW para 245k celdas (vectorizado)
  2. S2 prev-summer (cacheado por año)
  3. Recencia + fire-lags (KD tree, fires estrictamente < día)
  4. Predicción LGBM
  5. Recall@K: ¿los fuegos reales de ese día cayeron en el top-K%?

Uso: uv run python -m ardintra.models.fullgrid_eval
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

DATASET = Path("data/processed/features/v1/dataset.parquet")
CELLS = Path("data/processed/h3_cells/res9_static.parquet")
OUT = Path("reports/fullgrid_eval.json")

DYN_STATION = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
               "prec_mm", "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
               "ffmc_anom", "dmc_anom", "dc_anom", "fwi_7d", "fwi_prev", "ffmc_7d"]
STATIC = ["forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
          "dist_to_road_log", "road_density_log", "road_fuel_interface",
          "dens_ign_hum_5k", "dens_ign_rayo_5k", "dens_ign_otra_5k"]


def main(target_thresh_ha: float = 10.0, shard: str = ""):
    suffix = f"_{shard}" if shard else ""
    from ardintra.features.build_dataset import load_met_stations

    # --- cargar datos
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= target_thresh_ha)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    cells = pd.read_parquet(CELLS)
    cells["aspect_sin"] = np.sin(np.radians(cells["aspect_deg"]))
    cells["aspect_cos"] = np.cos(np.radians(cells["aspect_deg"]))
    cells["dist_to_road_log"] = np.log1p(cells["dist_to_road"].clip(0, 5000))
    cells["road_density_log"] = np.log1p(cells["road_density_1km"])
    met = load_met_stations().dropna(subset=["lat", "lon"])
    met["fecha_n"] = pd.to_datetime(met["fecha"]).dt.normalize()

    # --- entrenar LGBM en el training set
    FEATS = [c for c in STATIC + DYN_STATION + [
        "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
        "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
        "es_finde", "dens_x_weekend",
        "n_ign_cell_3d", "n_ign_near_2d", "n_ign_near_7d",
        "lat_f", "lon_f", "doy_sin", "doy_cos", "road_class"] if c in df.columns]
    X_tr = df[FEATS].copy()
    for c in X_tr.columns:
        if not (pd.api.types.is_numeric_dtype(X_tr[c]) or pd.api.types.is_bool_dtype(X_tr[c])):
            X_tr[c] = X_tr[c].astype("category")
    y_tr = df["big10"].to_numpy()
    tr = df["fecha"].dt.year.to_numpy() <= 2022
    spw = max((y_tr[tr] == 0).sum() / max((y_tr[tr] == 1).sum(), 1), 0.1)
    USE_TABPFN = os.environ.get("ARDENTIA_MODEL") == "tabpfn"
    USE_DISTILLED = os.environ.get("ARDENTIA_MODEL") == "distilled"
    if USE_DISTILLED:
        # blend 50/50: student destilado del teacher TabPFN + LGBM de labels
        import joblib
        s_pack = joblib.load("models/student_lgbm.joblib")
        l_pack = joblib.load("models/lgbm_labels.joblib")
        stud, lmod = s_pack["model"], l_pack["model"]
        cat_maps_d = s_pack["cat_maps"]

        def dist_predict(day_feats: pd.DataFrame) -> np.ndarray:
            Z = day_feats[FEATS].copy()
            for c, mapping in cat_maps_d.items():
                Z[c] = Z[c].map(mapping).fillna(-1)
            Z = Z.astype("float64")
            p_s = np.clip(stud.predict(Z), 0, 1)
            p_l = lmod.predict_proba(Z)[:, 1]
            return 0.5 * (p_s + p_l)
        print("modelo DISTILLED (blend student+labels) cargado", flush=True)
    elif USE_TABPFN:
        # TabPFN v2: ensembling por subsampling (todos los positivos + negativos aleatorios)
        from tabpfn import TabPFNClassifier
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N_ENS = int(os.environ.get("TABPFN_ENSEMBLE", "10"))
        rng_pf = np.random.default_rng(42)
        pos_all = np.where(y_tr[tr] == 1)[0]
        neg_all = np.where(y_tr[tr] == 0)[0]
        Xtr_num = X_tr[tr].reset_index(drop=True).copy()
        cat_maps = {}
        for c in Xtr_num.columns:
            if pd.api.types.is_categorical_dtype(Xtr_num[c]):
                cat_maps[c] = {v: i for i, v in enumerate(Xtr_num[c].cat.categories)}
                Xtr_num[c] = Xtr_num[c].cat.codes.astype("float64")
        Xtr_num = Xtr_num.astype("float64")
        tab_models = []
        for k in range(N_ENS):
            neg_k = rng_pf.choice(neg_all, size=10_000 - len(pos_all), replace=False)
            idx_k = np.sort(np.concatenate([pos_all, neg_k]))
            clf = TabPFNClassifier(device=device, random_state=42)
            clf.fit(Xtr_num.iloc[idx_k], y_tr[tr][idx_k])
            tab_models.append(clf)
        print(f"TabPFN ensemble ×{N_ENS} listo (device={device})", flush=True)

        def tab_predict(day_feats: pd.DataFrame) -> np.ndarray:
            Z = day_feats[FEATS].copy()
            for c, mapping in cat_maps.items():
                Z[c] = Z[c].map(mapping).fillna(-1)
            Z = Z.astype("float64")
            n = len(Z)
            out = np.zeros(n)
            B = 20_000
            for s in range(0, n, B):
                chunk = Z.iloc[s:s + B]
                ps = [clf.predict_proba(chunk)[:, 1] for clf in tab_models]
                out[s:s + B] = np.mean(ps, axis=0)
            import torch
            torch.cuda.empty_cache()
            return out
    else:
        model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=100, scale_pos_weight=spw,
                                   random_state=42, verbosity=-1)
        model.fit(X_tr[tr], y_tr[tr])
    print(f"modelo entrenado · features {len(FEATS)} · tabpfn={USE_TABPFN}", flush=True)

    # --- calibración isotónica (aux <=2021, ajuste en 2022)
    from sklearn.isotonic import IsotonicRegression
    years = df["fecha"].dt.year.to_numpy()
    m21, m22 = years <= 2021, years == 2022
    spw21 = max((y_tr[m21] == 0).sum() / max((y_tr[m21] == 1).sum(), 1), 0.1)
    m_aux = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, scale_pos_weight=spw21,
                               random_state=42, verbosity=-1)
    m_aux.fit(X_tr[m21], y_tr[m21])
    raw22 = m_aux.predict_proba(X_tr[m22])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw22, y_tr[m22])
    print("isotónica ajustada (2022)", flush=True)

    # --- mapeo celda -> región res7
    import h3 as h3lib
    cell_parent7 = np.array([h3lib.cell_to_parent(h, 7) for h in cells["h9"]])
    regional_rows = []

    # --- fuegos ≥10 ha (ground truth para el test)
    fires = df[(df["big10"] == 1) & (df["fecha"].dt.year >= 2023)][["h9", "fecha", "big10"]]
    fires = fires.merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    fire_tree = cKDTree(np.radians(fires[["lat", "lon"]].to_numpy()))
    fire_dates = pd.to_datetime(fires["fecha"]).to_numpy().astype("datetime64[D]")
    fire_ll = np.radians(fires[["lat", "lon"]].to_numpy())

    # --- KD tree de fuegos para los lags (todas las igniciones ≥0.1 ha)
    ig_all = df[df["label"] == 1][["h9", "fecha"]].copy()
    ig_all = ig_all.merge(cells[["h9", "lat", "lon"]], on="h9", how="left").dropna(subset=["lat", "lon"])
    lag_tree = cKDTree(np.radians(ig_all[["lat", "lon"]].to_numpy()))
    lag_dates = pd.to_datetime(ig_all["fecha"]).to_numpy().astype("datetime64[D]")

    # --- pivot de met para IDW vectorizado
    met_piv = {}
    ERA5_COLS = {"swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"}
    for var in DYN_STATION:
        if var in ERA5_COLS:
            continue  # ERA5 features se rellenan con mediana del train en el loop de días
        piv = met.pivot_table(index="fecha_n", columns="idema", values=var, aggfunc="first")
        met_piv[var] = piv

    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    st_tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    cell_ll = np.radians(cells[["lat", "lon"]].to_numpy())
    dd, ii = st_tree.query(cell_ll, k=3)
    w_idw = 1.0 / np.maximum(dd, 1e-5) ** 2
    w_idw = w_idw / w_idw.sum(axis=1, keepdims=True)
    st_ids = st_meta["idema"].to_numpy()

    # forecast D-1 para días de test (ARDENTIA_MET_FC=1): sustituye met obs del día D
    FC_VARS = ["tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
               "ffmc", "dmc", "dc", "isi", "bui", "fwi", "ffmc_7d", "fwi_7d", "fwi_prev"]
    fc_piv = {}
    if os.environ.get("ARDENTIA_MET_FC") == "1":
        mfc = pd.read_parquet("data/raw/openmeteo/met_fc.parquet")
        mfc["fecha"] = pd.to_datetime(mfc["fecha"]).dt.normalize()
        for var in FC_VARS:
            fc_piv[var] = mfc.pivot_table(index="fecha", columns="idema", values=var, aggfunc="first")
        print(f"met FORECAST activa para test ({len(fc_piv)} vars)", flush=True)

    # S2 cache por año
    s2_cache = {}

    # cache de densidad de igniciones por causa (años < Y, radio 5 km) — igual que en training
    from scipy.spatial import cKDTree as KDTree
    igc = df[(df["label"] == 1) & df["cause"].notna()][["h9", "fecha", "cause"]].copy()
    igc = igc.merge(cells[["h9", "lat", "lon"]], on="h9", how="left").dropna(subset=["lat", "lon"])
    igc["cause"] = pd.to_numeric(igc["cause"], errors="coerce")
    igc["_cls"] = np.select([igc["cause"].between(300, 499), igc["cause"].between(100, 199)],
                            ["hum", "rayo"], default="otra")
    dens_cache = {}
    for y in range(2017, 2025):
        dens_cache[y] = {}
        for cls in ("hum", "rayo", "otra"):
            hist = igc[(igc["_cls"] == cls) & (igc["fecha"].dt.year < y)]
            if hist.empty:
                dens_cache[y][cls] = np.zeros(len(cells))
                continue
            t = KDTree(np.radians(hist[["lat", "lon"]].to_numpy()))
            hits = t.query_ball_point(cell_ll, 5000.0 / 6_371_000.0, workers=-1)
            dens_cache[y][cls] = np.log1p(np.array([len(h) for h in hits], dtype=float))
    print("cache de densidad por causa listo", flush=True)

    # --- días de test con fuegos ≥10 ha
    test_fires = fires[fires["fecha"].dt.year >= 2023]
    test_days = sorted(test_fires["fecha"].dt.normalize().unique())
    print(f"días de test con fuegos ≥10 ha: {len(test_days)} de {len(test_fires)} fuegos", flush=True)
    if shard:
        a, b = shard.split(":")
        test_days = test_days[int(a):int(b)]
        print(f"shard {shard}: días {int(a)}..{int(b)} ({len(test_days)})", flush=True)

    # --- KD tree de celdas para los lags
    cell_tree = cKDTree(cell_ll)

    results_daily = []
    all_recall_1, all_recall_5, all_recall_10 = [], [], []
    n_processed = 0

    for day_np in test_days:
        day = pd.Timestamp(day_np)
        day_str = str(day_np)

        # 1) met IDW
        day_feats = cells.copy()
        # densidad histórica por causa (años < año del día)
        for cls in ("hum", "rayo", "otra"):
            day_feats[f"dens_ign_{cls}_5k"] = dens_cache[day.year][cls]
        # ERA5 features (swc/vpd) no vienen de estaciones: fill con mediana del train
        for era5_col in ("swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"):
            if era5_col in FEATS:
                day_feats[era5_col] = df.loc[df[era5_col].notna(), era5_col].median() if era5_col in df.columns else 0.0
        for var in DYN_STATION:
            if var in ("swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"):
                continue
            piv = fc_piv.get(var) if (fc_piv and day.year >= 2023 and var in fc_piv) else met_piv.get(var)
            if piv is None or day not in piv.index:
                day_feats[var] = np.nan
                continue
            vals = piv.loc[day].reindex(st_ids).to_numpy()
            vv = vals[ii]
            valid = ~np.isnan(vv)
            day_feats[var] = np.nansum(np.where(valid, vv, 0) * np.where(valid, w_idw, 0), axis=1) / \
                             np.maximum(np.where(valid, w_idw, 0).sum(axis=1), 1e-9)

        # 2) S2 prev-summer
        year = day.year
        if year - 1 not in s2_cache:
            import rasterio
            from pyproj import Transformer
            tif = Path(f"data/raw/s2/s2_summer_{year-1}.tif")
            if tif.exists():
                with rasterio.open(tif) as src:
                    from pyproj import Transformer as T2
                    tr_t = T2.from_crs("EPSG:4326", src.crs, always_xy=True)
                    xs, ys = tr_t.transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
                    s2_cache[year - 1] = np.array([x for x in src.sample(list(zip(xs, ys)))], dtype="float32")
            else:
                s2_cache[year - 1] = np.full((len(cells), 3), np.nan, dtype="float32")
        s2 = s2_cache[year - 1]
        day_feats["ndmi_summer"], day_feats["ndvi_summer"], day_feats["nbr_summer"] = s2[:, 0], s2[:, 1], s2[:, 2]
        day_feats["ndmi_anom"] = s2[:, 0] - np.nanmean(s2[:, 0])

        # 3) calendario
        day_feats["es_finde"] = int(day.dayofweek >= 5)
        day_feats["dens_x_weekend"] = day_feats["road_density_1km"].fillna(0) * day_feats["es_finde"]
        doy = day.dayofyear
        day_feats["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        day_feats["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        day_feats["lat_f"] = cells["lat"].to_numpy()
        day_feats["lon_f"] = cells["lon"].to_numpy()

        # 4) recencia + fire-lags (KD tree queries, VECTORIZADO — sin loop por celda)
        ang5 = 5000.0 / 6_371_000.0
        ang25 = 2500.0 / 6_371_000.0
        rec = np.full(len(cells), 1825.0)
        n7 = np.zeros(len(cells))
        n2 = np.zeros(len(cells))
        day64 = np.datetime64(day_np)
        for ang, radius_m, out_rec, out_cnt, win in (
            (ang5, 5000.0, rec, n7, 7), (ang25, 2500.0, None, n2, 2),
        ):
            hits = lag_tree.query_ball_point(cell_ll, ang, workers=-1)
            lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
            if not lens.any():
                continue
            cell_rep = np.repeat(np.arange(len(cells)), lens)
            fire_flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])
            dts = (day64 - lag_dates[fire_flat]) / np.timedelta64(1, "D")
            ok = (dts > 0) & (dts <= 1825)  # solo fuegos anteriores al día
            cv, dv = cell_rep[ok], dts[ok]
            if out_rec is not None:
                np.minimum.at(out_rec, cv, dv)
                m7 = dv <= win
                np.add.at(out_cnt, cv[m7], 1.0)
            else:
                mwin = dv <= win
                np.add.at(out_cnt, cv[mwin], 1.0)
        np.clip(rec, 0, 1825, out=rec)
        day_feats["dias_ultimo_fuego_cercano"] = rec
        day_feats["dist_ultimo_fuego_m"] = np.where(rec < 1825, 500.0, 5000.0)
        day_feats["ardio_1km_1an"] = (rec <= 365).astype(int)
        day_feats["n_ign_near_7d"] = n7
        day_feats["n_ign_near_2d"] = n2
        # n_ign_cell_3d: aproximado con near_2d (la celda propia está en el radio 2.5 km)
        day_feats["n_ign_cell_3d"] = n2

        # 5) predicción
        Xd = day_feats[FEATS].copy()
        for c in Xd.columns:
            if not (pd.api.types.is_numeric_dtype(Xd[c]) or pd.api.types.is_bool_dtype(Xd[c])):
                Xd[c] = Xd[c].astype("category")
        if USE_DISTILLED:
            p_day = dist_predict(day_feats)
        elif USE_TABPFN:
            p_day = tab_predict(day_feats)
        else:
            p_day = model.predict_proba(Xd)[:, 1]
        p_cal_day = iso.predict(p_day) if iso is not None else p_day

        # 6) ¿los fuegos reales de ese día cayeron en el top-K%?
        # fuegos de ese día: celdas con big10 == 1 en este día
        day_fires_cells = set(fires[fires["fecha"].dt.normalize() == day]["h9"].to_numpy())
        cell_h9 = cells["h9"].to_numpy()
        fire_mask = np.array([h in day_fires_cells for h in cell_h9])
        n_fires_day = fire_mask.sum()

        # 6b) agregación regional (res7): lambda = sum(p_cal), P(>=1) = 1-exp(-lambda)
        if iso is not None:
            tdf = pd.DataFrame({"r7": cell_parent7, "p": p_cal_day,
                                "fire": fire_mask.astype(int)})
            g = tdf.groupby("r7").agg(lam=("p", "sum"), n_fuegos_obs=("fire", "sum")).reset_index()
            g["p_alguno"] = 1 - np.exp(-g["lam"])
            g["fecha"] = day_str
            regional_rows.append(g[["fecha", "r7", "lam", "p_alguno", "n_fuegos_obs"]])

        if n_fires_day == 0:
            continue

        o = np.argsort(-p_day)
        recalls = {}
        for pct in (0.01, 0.05, 0.10):
            k = max(int(len(p_day) * pct), 1)
            captured = fire_mask[o[:k]].sum()
            recalls[f"recall@{pct:.0%}"] = int(captured)
            recalls[f"precision@{pct:.0%}"] = round(float(captured / k), 4)

        results_daily.append({
            "fecha": day_str,
            "n_fires": int(n_fires_day),
            **recalls,
            "p_max": round(float(p_day.max()), 4),
            "p_mean": round(float(p_day.mean()), 5),
        })
        all_recall_1.append(recalls.get("recall@1%", 0) / max(n_fires_day, 1))
        all_recall_5.append(recalls.get("recall@5%", 0) / max(n_fires_day, 1))
        all_recall_10.append(recalls.get("recall@10%", 0) / max(n_fires_day, 1))
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"  {n_processed}/{len(test_days)} días procesados ({day_str})", flush=True)

    # resumen
    total_fires = int(fire_mask.sum()) if len(test_days) else 0
    summary = {
        "n_dias_evaluados": n_processed,
        "n_fires_big10_total": int(sum(r["n_fires"] for r in results_daily)),
        "recall_top1_mean": round(float(np.mean(all_recall_1)), 3),
        "recall_top5_mean": round(float(np.mean(all_recall_5)), 3),
        "recall_top10_mean": round(float(np.mean(all_recall_10)), 3),
        "recall_top1_median": round(float(np.median(all_recall_1)), 3),
        "recall_top5_median": round(float(np.median(all_recall_5)), 3),
        "recall_top10_median": round(float(np.median(all_recall_10)), 3),
    }
    print(f"\n=== EVALUACIÓN FULL-GRID OPERATIVA ===")
    print(json.dumps(summary, indent=2))

    OUT = Path(f"reports/fullgrid_eval{suffix}.json")
    OUT.write_text(json.dumps({"summary": summary, "daily": results_daily}, indent=2))
    print(f"-> {OUT}")

    # --- guardado regional + reliability a nivel región
    if regional_rows:
        reg = pd.concat(regional_rows, ignore_index=True)
        REG_OUT = Path(f"reports/fullgrid_regional{suffix}.parquet")
        reg.to_parquet(REG_OUT, index=False)
        print(f"-> {REG_OUT} ({len(reg):,} regiones-día)")
        # reliability: bins de p_alguno vs frecuencia real de >=1 fuego
        reg["hubo_fuego"] = (reg["n_fuegos_obs"] > 0).astype(int)
        bins = np.array([0, .001, .005, .01, .02, .05, .10, .20, .50, 1.01])
        reg["bin"] = pd.cut(reg["p_alguno"], bins)
        rel = reg.groupby("bin", observed=True).agg(
            n=("hubo_fuego", "size"),
            dias_con_fuego=("hubo_fuego", "sum"),
            p_mediana=("p_alguno", "median"),
        ).reset_index()
        rel["frecuencia_real"] = (rel["dias_con_fuego"] / rel["n"]).round(4)
        print("\n=== RELIABILITY REGIONAL (P(>=1) vs realidad) ===")
        print(rel.to_string(index=False))
        rel.to_csv(Path("reports/regional_reliability.csv"), index=False)


if __name__ == "__main__":
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    shard = sys.argv[2] if len(sys.argv) > 2 else ""
    main(thresh, shard)
