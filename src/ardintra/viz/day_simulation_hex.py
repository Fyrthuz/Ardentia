"""Vídeo de simulación con POLÍGONOS HEXAGONALES H3 (no puntos).

Cada celda se dibuja como su hexágono real (h3.cell_to_boundary) usando
matplotlib PolyCollection. Los colores cambian por día (el baremo es fijo).
Los fuegos reales aparecen como marcadores. Zoom-inset en fuegos ≥10 ha.

La renderización usa rasterio para crear el overlay PNG del fondo (más rápido
que PolyCollection para 245k polígonos en res9; para res8 usa PolyCollection).

Uso: ARDENTIA_RES=9|8 uv run python -m ardintra.viz.day_simulation [year]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h3
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

RAMP = np.array([(54,168,82),(235,210,60),(245,140,40),(232,60,40),(150,15,45)], dtype=np.uint8)
CUTS = np.array([0.20, 0.35, 0.52, 0.72])
NAMES = ["Bajo", "Moderado", "Alto", "Muy alto", "Extremo"]

RES = int(os.environ.get("ARDENTIA_RES", "9"))
SUF = "" if RES == 9 else f"_res{RES}"
DATASET = Path(f"data/processed/features/v1{SUF}/dataset.parquet")
CELLS = Path(f"data/processed/h3_cells/res{RES}_static.parquet")
FRAMES = Path(f"data/processed/video_frames_res{RES}")
OUT_VIDEO = Path(f"reports/simulacion_2023_res{RES}.mp4")
FEATS = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
    "ffmc_anom", "dmc_anom", "dc_anom",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend", "road_class",
]


def load_data():
    df = pd.read_parquet(DATASET)
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    cells = pd.read_parquet(CELLS)
    cells["aspect_sin"] = np.sin(np.radians(cells["aspect_deg"]))
    cells["aspect_cos"] = np.cos(np.radians(cells["aspect_deg"]))
    cells["dist_to_road_log"] = np.log1p(cells["dist_to_road"].clip(0, 5000))
    cells["road_density_log"] = np.log1p(cells["road_density_1km"])
    return df, cells


def prep(df):
    X = df[[c for c in FEATS if c in df.columns]].copy()
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
    for var in ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh",
                "prec_mm", "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
                "ffmc_anom", "dmc_anom", "dc_anom"]:
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
                from pyproj import Transformer as T2
                tr = T2.from_crs("EPSG:4326", src.crs, always_xy=True)
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
    return d


def hex_polys(cells_df):
    """Precomputa los vértices de los hexágonos H3 → lista de listas de (lon, lat)."""
    polys = []
    for h9 in cells_df["h9"]:
        ring = h3.cell_to_boundary(h9)
        polys.append([(lng, lat) for lat, lng in ring])
    return polys


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

    # modelo fino
    X_all = prep(df)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1)
    m.fit(X_all[tr], y_big[tr])
    p_raw_all = m.predict_proba(X_all)[:, 1]

    # calibración OOS (modelo auxiliar ≤2021)
    tr21 = (df["fecha"].dt.year <= 2021) & df["fwi"].notna()
    m_aux = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, scale_pos_weight=spw,
                               random_state=42, verbosity=-1)
    m_aux.fit(X_all[tr21], y_big[tr21])
    raw_val = m_aux.predict_proba(X_all[val22])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_big[val22])

    # referencia = temporada (may-oct) de train
    ref_pool = df[(df["fecha"].dt.year <= 2022) & (df["fecha"].dt.month.isin([5,6,7,8,9,10]))]
    p_ref_cal = iso.predict(m.predict_proba(prep(ref_pool))[:, 1])
    val_sorted = np.sort(p_ref_cal)
    n_ref = len(val_sorted)

    # suavizado espacial: se aplica POR DÍA en el loop (W es por celda, no por celda-día)
    W_smooth = get_smooth_matrix(list(cells["h9"]))

    # polígonos hexagonales (precomputados una vez)
    print("precomputando hexágonos…", flush=True)
    polys = hex_polys(cells)
    print(f"polígonos: {len(polys):,}")

    FRAMES.mkdir(parents=True, exist_ok=True)
    cmap = mcolors.ListedColormap(RAMP / 255.0)
    dias = pd.date_range(f"{year}-06-01", f"{year}-10-31", freq="D")

    for n_day, day in enumerate(dias):
        png = FRAMES / f"frame_{n_day:03d}.png"
        if png.exists():
            continue

        feats = build_day_features(day, cells, met, s2_cache := {}, fire_tree, ig_dates)
        Xd = prep(feats)
        p_raw_day = m.predict_proba(Xd)[:, 1]
        p_smooth_day = smooth_fn(p_raw_day, W_smooth)
        p_cal_day = iso.predict(p_smooth_day)
        pct_rank_day = np.searchsorted(val_sorted, p_cal_day, side="right") / n_ref
        nivel_day = np.digitize(pct_rank_day, [0.40, 0.75, 0.90, 0.97])
        day_colors = RAMP[nivel_day]

        fires_day = ig[pd.to_datetime(ig["deteccion"]).dt.normalize() == day]

        fig, ax = plt.subplots(figsize=(11, 9), dpi=115)
        pc = PolyCollection(polys, facecolors=day_colors / 255.0, edgecolors=(0, 0, 0, 0.15),
                            linewidths=0.15 if RES == 9 else 0.3)
        ax.add_collection(pc)
        ax.set_xlim(-9.65, -6.60); ax.set_ylim(41.75, 43.95)
        ax.set_title(f"WildfireRisk-ES · {day.date()} · percentil vs referencia histórica", fontsize=12)

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
                axin.scatter(cells.loc[mzoom, "lon"], cells.loc[mzoom, "lat"], c=nivel_day[mzoom],
                             cmap=cmap, s=zz_s, vmin=0, vmax=4, alpha=0.9,
                             linewidths=0.3, edgecolors=(0, 0, 0, 0.4))
                axin.scatter([fb["longitud"]], [fb["latitud"]], marker="*", s=420, c="darkred",
                             edgecolors="white", linewidths=1.2, zorder=6)
                axin.set_xlim(fb["longitud"] - w, fb["longitud"] + w)
                axin.set_ylim(fb["latitud"] - w, fb["latitud"] + w)
                axin.set_title(f"zoom: {fb['superficie_quemada_ha']:.0f} ha", fontsize=8)
                axin.tick_params(labelsize=6)
                ax.indicate_inset_zoom(axin, edgecolor="gray")

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 4)),
                            ax=ax, shrink=0.55, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(NAMES)
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        if n_day % 15 == 0:
            print(f"frame {n_day}/{len(dias)} ({day.date()})", flush=True)

    out = str(OUT_VIDEO)
    os.system(f"ffmpeg -y -framerate 1/2 -i {FRAMES}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 23 {out}")
    print(f"-> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
