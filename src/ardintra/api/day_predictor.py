"""Predicción diaria completa: el pipeline de producción sobre cualquier día.

Entrada una fecha → salida el forecast operativa del sistema:
  1. Features full-grid 245k celdas (meteo del día vía forecast D-1 dentro de la
     cobertura de met_fc; obs para histórico; S2 prev-summer; recencia/lags;
     estáticas viarias/forestales)
  2. LGBM 48 features → score crudo
  3. Calibración isotónica (ajustada en 2022) → P(ignición ≥10 ha | celda, día)
  4. Agregación regional Stage 2 (GLM + Moran Eigenvector Maps):
     μ_r = exp(α + β·log(1+λ_r) + MEM·γ) → P(≥1 fuego en la región) y λ diario
  5. Colores por celda: percentil vs la distribución de temporada de train

Uso CLI: uv run python -m ardintra.api.day_predictor 2024-07-15
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")

DYN_STATION = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
               "prec_mm", "spei30", "spei90", "ffmc_anom", "dmc_anom", "dc_anom",
               "fwi_7d", "fwi_prev", "ffmc_7d"]
ERA5_FILL = ["swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean"]
ERA5_COLS = set(ERA5_FILL)


@lru_cache(maxsize=1)
def _cells() -> pd.DataFrame:
    d = pd.read_parquet(DATA / "h3_cells/res9_static.parquet")
    d["aspect_sin"] = np.sin(np.radians(d["aspect_deg"]))
    d["aspect_cos"] = np.cos(np.radians(d["aspect_deg"]))
    d["dist_to_road_log"] = np.log1p(d["dist_to_road"].clip(0, 5000))
    d["road_density_log"] = np.log1p(d["road_density_1km"])
    return d


@lru_cache(maxsize=1)
def _lgbm_pack() -> dict:
    import joblib
    return joblib.load(MODELS / "lgbm_labels.joblib")


@lru_cache(maxsize=1)
def _iso() -> object:
    import joblib
    return joblib.load(MODELS / "iso_calibrator.joblib")


@lru_cache(maxsize=1)
def _stage2() -> dict:
    m = np.load(REPORTS / "stage2_model.npz", allow_pickle=True)
    regions = [str(r) for r in m["regions"]]
    regall = pd.read_parquet(REPORTS / "fullgrid_regional.parquet")
    ridx = {r: i for i, r in enumerate(regions)}
    ri = regall["r7"].map(ridx).to_numpy()
    ok = ri >= 0
    feat = np.zeros((len(regall), 1 + m["mem"].shape[1]))
    feat[ok, 0] = np.log1p(regall["lam"].to_numpy()[ok])
    feat[ok, 1:] = m["mem"][ri[ok]]
    p_ref = 1 - np.exp(-np.exp(float(m["intercept"]) + feat @ m["coef"]))
    return {
        "regions": regions,
        "ridx": {r: i for i, r in enumerate(regions)},
        "mem": m["mem"], "coef": m["coef"], "b0": float(m["intercept"]),
        "base_rate": float((regall["n_fuegos_obs"] > 0).mean()),
        "q_levels_90": float(np.quantile(p_ref, 0.90)),
    }


@lru_cache(maxsize=1)
def _met_context() -> dict:
    from ardintra.features.build_dataset import load_met_stations
    met = load_met_stations()
    met["fecha_n"] = pd.to_datetime(met["fecha"]).dt.normalize()
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    cells = _cells()
    dd, ii = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=3)
    w = 1.0 / np.maximum(dd, 1e-5) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    piv = {v: met.pivot_table(index="fecha_n", columns="idema", values=v, aggfunc="first")
           for v in DYN_STATION}
    fc = pd.read_parquet("data/raw/openmeteo/met_fc.parquet")
    fc["fecha"] = pd.to_datetime(fc["fecha"]).dt.normalize()
    fc_piv = {v: fc.pivot_table(index="fecha", columns="idema", values=v, aggfunc="first")
              for v in DYN_STATION if v in fc.columns}
    ds = pd.read_parquet(DATA / "features/v1/dataset.parquet", columns=ERA5_FILL)
    era5_med = {c: float(ds[c].median()) for c in ERA5_FILL}

    ig = pd.read_parquet(DATA / "labels/v1/ignitions.parquet")
    ig = ig.merge(cells[["h9", "lat", "lon"]], on="h9", how="left").dropna(subset=["lat", "lon"])
    lag_tree = cKDTree(np.radians(ig[["lat", "lon"]].to_numpy()))
    lag_dates = pd.to_datetime(ig["fecha"]).to_numpy().astype("datetime64[D]")

    return {"ii": ii, "w": w, "st_ids": st_meta["idema"].to_numpy(),
            "obs_piv": piv, "fc_piv": fc_piv, "fc_min": pd.Timestamp(fc["fecha"].min()),
            "era5_med": era5_med, "lag_tree": lag_tree, "lag_dates": lag_dates}


@lru_cache(maxsize=16)
def _s2_for_year(year: int) -> np.ndarray:
    import rasterio
    from pyproj import Transformer
    tif = Path(f"data/raw/s2/s2_summer_{year - 1}.tif")
    cells = _cells()
    if tif.exists():
        with rasterio.open(tif) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = tr.transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
            return np.array([x for x in src.sample(list(zip(xs, ys)))], dtype="float32")
    return np.full((len(cells), 3), np.nan, dtype="float32")


def _idw(day: pd.Timestamp, ii: np.ndarray, w: np.ndarray, st_ids,
         pivots: dict) -> dict:
    out = {}
    for var, piv in pivots.items():
        if var in ERA5_COLS or day not in piv.index:
            continue
        vals = piv.loc[day].reindex(st_ids).to_numpy()
        vv = vals[ii]
        valid = ~np.isnan(vv)
        out[var] = np.nansum(np.where(valid, vv, 0) * np.where(valid, w, 0), axis=1) / \
            np.maximum(np.where(valid, w, 0).sum(axis=1), 1e-9)
    return out


def predict_day(date_str: str) -> dict:
    day = pd.Timestamp(date_str).normalize()
    cells = _cells()
    mc = _met_context()
    pack = _lgbm_pack()
    model, feats, cat_maps = pack["model"], pack["feats"], pack["cat_maps"]

    d = cells.copy()
    day64 = np.datetime64(day.date())

    # meteo: forecast D-1 dentro de la cobertura de met_fc, obs fuera
    src = "forecast D-1"
    idx_pivs = mc["fc_piv"]
    if day < mc["fc_min"]:
        src = "obs (histórico)"
        idx_pivs = mc["obs_piv"]
    elif not any(day in p.index for p in mc["fc_piv"].values() if hasattr(p, "index")):
        src = "obs (sin cobertura forecast)"
        idx_pivs = mc["obs_piv"]
    met_vals = _idw(day, mc["ii"], mc["w"], mc["st_ids"], idx_pivs)
    # índices lentos (SPEI, anomalías de códigos): siempre obs aunque el día sea forecast
    for var in ("spei30", "spei90", "ffmc_anom", "dmc_anom", "dc_anom"):
        if var not in met_vals:
            _obs_slow = _idw(day, mc["ii"], mc["w"], mc["st_ids"], {var: mc["obs_piv"][var]})
            met_vals[var] = _obs_slow.get(var, np.nan)
    n_afectadas = [v for v in DYN_STATION if v not in met_vals]
    for var in DYN_STATION:
        d[var] = met_vals.get(var, np.nan)
    for c in ERA5_FILL:
        d[c] = mc["era5_med"][c]

    # calendario
    d["es_finde"] = int(day.day_of_week >= 5)
    d["dens_x_weekend"] = d["road_density_1km"].fillna(0) * d["es_finde"]
    doy = day.day_of_year
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    d["lat_f"] = d["lat"].to_numpy()
    d["lon_f"] = d["lon"].to_numpy()

    # S2 prev-summer
    s2 = _s2_for_year(day.year)
    d["ndmi_summer"], d["ndvi_summer"], d["nbr_summer"] = s2[:, 0], s2[:, 1], s2[:, 2]
    d["ndmi_anom"] = s2[:, 0] - np.nanmean(s2)

    # recencia + lags (vectorizado)
    cell_ll = np.radians(cells[["lat", "lon"]].to_numpy())
    rec = np.full(len(cells), 1825.0)
    n7 = np.zeros(len(cells))
    n2 = np.zeros(len(cells))
    for ang, out_cnt, win in ((5000.0 / 6_371_000.0, n7, 7), (2500.0 / 6_371_000.0, n2, 2)):
        hits = mc["lag_tree"].query_ball_point(cell_ll, ang, workers=-1)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if not lens.any():
            continue
        crep = np.repeat(np.arange(len(cells)), lens)
        fire_flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])
        dts = (day64 - mc["lag_dates"][fire_flat]) / np.timedelta64(1, "D")
        ok = (dts > 0) & (dts <= 1825)
        cv, dv = crep[ok], dts[ok]
        if out_cnt is n7:
            np.minimum.at(rec, cv, dv)
        mwin = dv <= win
        np.add.at(out_cnt, cv[mwin], 1.0)
    np.clip(rec, 0, 1825, out=rec)
    d["dias_ultimo_fuego_cercano"] = rec
    d["dist_ultimo_fuego_m"] = np.where(rec < 1825, 500.0, 5000.0)
    d["ardio_1km_1an"] = (rec <= 365).astype(int)
    d["n_ign_near_7d"] = n7
    d["n_ign_near_2d"] = n2
    d["n_ign_cell_3d"] = n2

    # 1-2) LGBM + isotónica
    Z = d[feats].copy()
    for c, mapping in cat_maps.items():
        if c in Z.columns:
            Z[c] = Z[c].map(mapping).fillna(-1)
    Z = Z.astype("float64")
    p_raw = model.predict_proba(Z)[:, 1]
    p_cal = _iso().predict(p_raw)

    # 3) Stage 2 regional
    s2m = _stage2()
    parents = np.array([h3.cell_to_parent(x, 7) for x in cells["h9"]])
    tdf = pd.DataFrame({"r7": parents, "p": p_cal})
    g = tdf.groupby("r7")["p"].sum()          # Series por región (¡sin named agg!)
    lams = np.zeros(len(s2m["regions"]))
    for r7, lam in g.items():
        i = s2m["ridx"].get(r7)
        if i is not None:
            lams[i] = lam
    feat = np.zeros((len(s2m["regions"]), 1 + s2m["mem"].shape[1]))
    feat[:, 0] = np.log1p(lams)
    feat[:, 1:] = s2m["mem"]
    mu = np.exp(s2m["b0"] + feat @ s2m["coef"])
    p_reg = 1 - np.exp(-mu)

    order = np.argsort(-p_reg)[:3]
    top_regions = []
    for i in order:
        lat_c, lng_c = h3.cell_to_latlng(s2m["regions"][i])
        top_regions.append({
            "region": s2m["regions"][i], "lat": lat_c, "lon": lng_c,
            "p_alguno": round(float(p_reg[i]), 5),
            "riesgo_x": round(float(p_reg[i]) / s2m["base_rate"], 1),
        })
    lam_total = float(mu.sum())
    top_idx = np.argsort(-p_cal)[:25]
    top_cells = [{"h9": cells["h9"].iloc[j], "lat": round(float(cells["lat"].iloc[j]), 5),
                  "lon": round(float(cells["lon"].iloc[j]), 5),
                  "p": round(float(p_cal[j]), 5)} for j in top_idx]

    # 4) niveles de celda: PERCENTIL INTRA-DÍA consciente de empates
    #    (los colores muestran DÓNDE se concentra el riesgo del día — el nivel
    #    absoluto por celda no es recuperable fuera de la distribución de
    #    calibración; la magnitud del día la dan λ y las regiones ×N del panel)
    from scipy.stats import rankdata
    pct = (rankdata(p_cal) - 0.5) / len(p_cal)
    levels = np.digitize(pct, [0.40, 0.75, 0.90, 0.97])

    return {
        "fecha": str(day.date()),
        "meteo_source": src,
        "vars_rellenadas": n_afectadas,
        "lambda_galicia": round(lam_total, 3),
        "base_rate_region_dia": round(s2m["base_rate"], 6),
        "p_celda_max": round(float(p_cal.max()), 5),
        "regiones_top": top_regions,
        "celdas_top": top_cells,
        "_arrays": {"p_cal": p_cal, "levels": levels, "h9": cells["h9"].to_numpy()},
    }


def fuegos_del_dia(date_str: str) -> list[dict]:
    """Fuegos EGIF ≥0.1 ha del día (los ≥10 ha como estrellas)."""
    ig = pd.read_parquet(DATA / "labels/v1/ignitions.parquet")
    ig["superficie"] = pd.to_numeric(ig.get("surface_ha"), errors="coerce").fillna(0)
    ig["fecha"] = pd.to_datetime(ig["fecha"]).dt.normalize()
    g = ig[ig["fecha"] == pd.Timestamp(date_str)]
    cells = _cells().set_index("h9")
    out = []
    for _, r in g.iterrows():
        if r["h9"] not in cells.index:
            continue
        c = cells.loc[r["h9"]]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        out.append({"lon": round(float(c["lon"]), 5), "lat": round(float(c["lat"]), 5),
                    "ha": round(float(r["superficie"]), 1)})
    return out


if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2024-07-15"
    out = predict_day(date)
    out.pop("_arrays")
    print(json.dumps(out, indent=2, ensure_ascii=False))
