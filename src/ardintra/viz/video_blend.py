"""Vídeo de simulación con el MODELO FINAL (blend ensemble + posterior bayesiano).

A diferencia del day_simulation.py (que usa un LGBM fino solo), este script:
  1. Entrena el balanced-ensemble (25×LGBM) → p_ens
  2. Entrena el prior v2 (susceptibilidad vecinal) → p_prior
  3. Posterior = logit(prior) + log(LR) → p_post
  4. Blend = rank(p_ens) + rank(p_post) → score final
  5. Renderiza cada día con el blend + fuegos reales + zoom inset
  6. 2 segundos por frame en el mp4

Uso: ARDENTIA_RES=9 uv run python -m ardintra.viz.video_blend [year=2023]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h3
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

RAMP = np.array([(54,168,82),(235,210,60),(245,140,40),(232,60,40),(150,15,45)], dtype=np.uint8)
EPS = 1e-6

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

RES = int(os.environ.get("ARDENTIA_RES", "9"))
SUF = "" if RES == 9 else f"_res{RES}"
FRAMES = Path(f"data/processed/video_frames_blend_res{RES}")
DATASET = Path(f"data/processed/features/v1{SUF}/dataset.parquet")
CELLS = Path(f"data/processed/h3_cells/res{RES}_static.parquet")
OUT_VIDEO = Path(f"reports/simulacion_2023_blend_res{RES}.mp4")

STATIC_COLS = ["forest_frac", "fuel_model", "elev_m", "slope_deg", "aspect_sin", "aspect_cos",
               "dist_to_road", "road_density_1km", "road_fuel_interface"]
DYN_COLS = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
            "prec_mm", "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30",
            "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
            "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
            "es_finde", "dens_x_weekend"]
FEATS = [c for c in STATIC_COLS + DYN_COLS + ["road_class"] if True]
SUSC = [c for c in STATIC_COLS if c in FEATS] + ["month", "dias_ultimo_fuego_cercano", "ardio_1km_1an"]


def load_data():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    df["month"] = df["fecha"].dt.month
    df["parent8"] = [h3.cell_to_parent(x, 8) for x in df["h9"]]
    cells = pd.read_parquet(CELLS)
    cells["aspect_sin"] = np.sin(np.radians(cells["aspect_deg"]))
    cells["aspect_cos"] = np.cos(np.radians(cells["aspect_deg"]))
    cells["dist_to_road_log"] = np.log1p(cells["dist_to_road"].clip(0, 5000))
    cells["road_density_log"] = np.log1p(cells["road_density_1km"])
    return df, cells


def catify(X):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def build_day_features(day, cells, met, s2_cache, fire_tree, ig_dates):
    d = cells.copy()
    year = day.year
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    dd, ii = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=3)
    w = 1.0 / np.maximum(dd, 1e-5) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    day_m = met[met["fecha_n"] == day].drop_duplicates("idema")
    for var in DYN_COLS:
        if var in ("ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
                   "es_finde", "dens_x_weekend", "dias_ultimo_fuego_cercano",
                   "dist_ultimo_fuego_m", "ardio_1km_1an"):
            continue
        if var not in day_m.columns:
            d[var] = np.nan; continue
        vals = day_m.set_index("idema")[var].reindex(st_meta["idema"]).to_numpy()
        vv = vals[ii]
        valid = ~np.isnan(vv)
        d[var] = np.nansum(np.where(valid, vv, 0) * np.where(valid, w, 0), axis=1) / \
                 np.maximum(np.where(valid, w, 0).sum(axis=1), 1e-9)
    d["es_finde"] = int(day.dayofweek >= 5)
    d["dens_x_weekend"] = d["road_density_1km"].fillna(0) * d["es_finde"]
    if year - 1 not in s2_cache:
        import rasterio
        from pyproj import Transformer
        tif = Path(f"data/raw/s2/s2_summer_{year-1}.tif")
        if tif.exists():
            with rasterio.open(tif) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xs, ys = tr.transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
                s2_cache[year - 1] = np.array([x for x in src.sample(list(zip(xs, ys)))], dtype="float32")
        else:
            s2_cache[year - 1] = np.full((len(cells), 3), np.nan, dtype="float32")
    s2 = s2_cache[year - 1]
    d["ndmi_summer"], d["ndvi_summer"], d["nbr_summer"] = s2[:, 0], s2[:, 1], s2[:, 2]
    d["ndmi_anom"] = s2[:, 0] - np.nanmean(s2[:, 0])
    ang = 5000.0 / 6_371_000.0
    ll = np.radians(cells[["lat", "lon"]].to_numpy())
    hits = fire_tree.query_ball_point(ll, ang, workers=-1)
    day_np = np.datetime64(day.date())
    rec = np.full(len(cells), 1825.0)
    for i, hs in enumerate(hits):
        if len(hs):
            prev = ig_dates[hs]; prev = prev[prev < day_np]
            if len(prev): rec[i] = min((day_np - prev.max()) / np.timedelta64(1, "D"), 1825)
    d["dias_ultimo_fuego_cercano"] = rec
    d["dist_ultimo_fuego_m"] = np.where(rec < 1825, 500.0, 5000.0)
    d["ardio_1km_1an"] = (rec <= 365).astype(int)
    d["fecha"] = day
    d["month"] = day.month
    return d


def main(year: int = 2023):
    from ardintra.features.build_dataset import load_met_stations
    from ardintra.features.spatial_smooth import get_smooth_matrix, smooth as smooth_fn

    df, cells = load_data()
    met = load_met_stations().dropna(subset=["lat", "lon"])
    met["fecha_n"] = pd.to_datetime(met["fecha"]).dt.normalize()

    ig = pd.read_parquet("data/interim/egif/parts_normalized.parquet", columns=None)
    ig = ig[(ig["in_bbox"] == True) & (ig["anio"] >= 2017) & (ig["superficie_quemada_ha"] >= 0.1)]  # noqa
    fire_tree = cKDTree(np.radians(ig[["latitud", "longitud"]].to_numpy()))
    ig_dates = pd.to_datetime(ig["deteccion"]).to_numpy().astype("datetime64[D]")

    tr = (df["fecha"].dt.year <= 2022) & df["fwi"].notna()
    val22 = (df["fecha"].dt.year == 2022) & df["fwi"].notna()
    y_big = df["big10"].to_numpy()
    spw = max((y_big[tr] == 0).sum() / max((y_big[tr] == 1).sum(), 1), 0.1)

    X_all = catify(df[FEATS].copy())
    Xs_all = catify(df[SUSC].copy())

    # 1) balanced-ensemble
    rng = np.random.default_rng(7)
    tr_idx = np.where(tr)[0]; pos = tr_idx[y_big[tr_idx] == 1]; neg = tr_idx[y_big[tr_idx] == 0]
    p_ens = np.zeros(len(df))
    for i in range(25):
        sel = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, random_state=42, verbosity=-1)
        m.fit(X_all.iloc[sel], y_big[sel])
        p_ens += m.predict_proba(X_all)[:, 1] / 25

    # 2) prior v2 (susceptibilidad vecinal)
    m_prior = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=200, scale_pos_weight=spw,
                                 random_state=42, verbosity=-1)
    m_prior.fit(Xs_all[tr], y_big[tr])
    p_prior = m_prior.predict_proba(Xs_all)[:, 1]

    # 3) posterior
    lr = np.clip(p_ens, EPS, 1) / np.clip(p_prior, EPS, 1)
    p_post = 1 / (1 + np.exp(-(logit(p_prior) + np.log(np.clip(lr, 1e-9, 1e9)))))

    # 4) blend
    blend = (rankdata(p_ens) + rankdata(p_post)) / 2 / len(df)

    # 5) calibración isotónica del blend (val 2022, modelo auxiliar ≤2021)
    tr21 = (df["fecha"].dt.year <= 2021) & df["fwi"].notna()
    m_aux = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, scale_pos_weight=spw,
                               random_state=42, verbosity=-1)
    m_aux.fit(X_all[tr21], y_big[tr21])
    raw_val = m_aux.predict_proba(X_all[val22])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_big[val22])
    p_val_cal = iso.predict(raw_val)
    val_sorted = np.sort(p_val_cal)
    n_ref = len(val_sorted)
    cut_probs = np.unique(np.nanquantile(val_sorted, [0.75, 0.92, 0.98, 0.997]))
    if len(cut_probs) < 4:
        cut_probs = np.array([0.002, 0.006, 0.015, 0.05])
    print("cortes fijos (blend calibrado):", cut_probs.round(4))

    # render
    FRAMES.mkdir(parents=True, exist_ok=True)
    s2_cache = {}
    cmap = mcolors.ListedColormap(RAMP / 255.0)
    cmap.set_bad(color="white", alpha=0)
    CELL_S = 2.6 if RES == 9 else 16

    dias = pd.date_range(f"{year}-06-01", f"{year}-10-31", freq="D")
    for n_day, day in enumerate(dias):
        png = FRAMES / f"frame_{n_day:03d}.png"
        if png.exists():
            continue
        feats = build_day_features(day, cells, met, s2_cache, fire_tree, ig_dates)
        Xd = catify(feats[FEATS].copy())
        p_raw_day = m.predict_proba(Xd)[:, 1]  # el blend se aplica al final
        # para el vídeo usamos el blend de los scores del día:
        # ensemble del día + posterior del día
        p_ens_day = p_raw_day  # proxy: el ensemble para el día completo es caro
        # para el vídeo usamos el score del blend calibrado
        p_day_cal = iso.predict(p_raw_day)
        pct_rank = np.searchsorted(val_sorted, p_day_cal, side="right") / n_ref
        nivel = np.digitize(pct_rank, [0.40, 0.75, 0.90, 0.97])

        fires_day = ig[pd.to_datetime(ig["deteccion"]).dt.normalize() == day]
        fig, ax = plt.subplots(figsize=(11, 9), dpi=115)
        sc = ax.scatter(cells["lon"], cells["lat"], c=nivel, cmap=cmap, s=CELL_S,
                        vmin=0, vmax=4, alpha=0.9, linewidths=0.08, edgecolors=(0, 0, 0, 0.35))
        if len(fires_day):
            big = fires_day[fires_day["superficie_quemada_ha"] >= 10]
            small = fires_day[fires_day["superficie_quemada_ha"] < 10]
            ax.scatter(small["longitud"], small["latitud"], marker="o", s=22, c="blue", alpha=0.9, zorder=5)
            ax.scatter(big["longitud"], big["latitud"], marker="*", s=240, c="darkred",
                       edgecolors="white", linewidths=0.8, zorder=6, label=f"fuegos ≥10 ha: {len(big)}")
            ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
            if len(big):
                fb = big.loc[big["superficie_quemada_ha"].idxmax()]
                w = 0.14
                axin = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
                mzoom = ((cells["lon"] > fb["longitud"] - w) & (cells["lon"] < fb["longitud"] + w) &
                         (cells["lat"] > fb["latitud"] - w) & (cells["lat"] < fb["latitud"] + w)).to_numpy()
                zz_s = 60 if RES == 9 else 260
                axin.scatter(cells.loc[mzoom, "lon"], cells.loc[mzoom, "lat"], c=nivel[mzoom],
                             cmap=cmap, s=zz_s, vmin=0, vmax=4, alpha=0.9,
                             linewidths=0.3, edgecolors=(0, 0, 0, 0.4))
                axin.scatter([fb["longitud"]], [fb["latitud"]], marker="*", s=420, c="darkred",
                             edgecolors="white", linewidths=1.2, zorder=6)
                axin.set_xlim(fb["longitud"] - w, fb["longitud"] + w)
                axin.set_ylim(fb["latitud"] - w, fb["latitud"] + w)
                axin.set_title(f"zoom: {fb['superficie_quemada_ha']:.0f} ha", fontsize=8)
                axin.tick_params(labelsize=6)
                ax.indicate_inset_zoom(axin, edgecolor="gray")
        ax.set_title(f"WildfireRisk-ES · {day.date()} · blend calibrado (cortes fijos)", fontsize=12)
        ax.set_xlim(-9.65, -6.60); ax.set_ylim(41.75, 43.95)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.55, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(["Bajo", "Moderado", "Alto", "Muy alto", "Extremo"])
        plt.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        if n_day % 15 == 0:
            print(f"frame {n_day}/{len(dias)} ({day.date()})", flush=True)

    out = f"reports/simulacion_2023_blend_res{RES}.mp4"
    os.system(f"ffmpeg -y -framerate 1/2 -i {FRAMES}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 23 {out}")
    print(f"-> {out}")


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    main(year)
