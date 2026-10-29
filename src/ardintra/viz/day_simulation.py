"""Simulación visual diaria: predicciones del modelo sobre TODO el territorio y
los fuegos reales apareciendo, para la temporada 2023. Colores corporativos
(verde→amarillo→naranja→rojo→granate según nivel 0–4).

Genera PNG por día (frames/) → mp4 vía ffmpeg.

Uso: uv run python -m ardintra.viz.day_simulation [año=2023]
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import h3
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ardintra.features.build_dataset import load_met_stations

import os as _os
_RES_V = int(_os.environ.get("ARDENTIA_RES", "9"))
FRAMES = Path(f"data/processed/video_frames_res{_RES_V}")
FEATS_NUM = [
    "fwi", "ffmc", "dmc", "dc", "isi", "bui", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm",
    "swc_l1_mean", "swc_l2_mean", "vpd_max", "vpd_mean", "spei30", "spei90",
    "ffmc_anom", "dmc_anom", "dc_anom",
    "ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
    "forest_frac", "fuel_model", "slope_deg", "aspect_sin", "aspect_cos", "elev_m",
    "dist_to_road", "road_density_1km", "road_fuel_interface",
    "dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an",
    "es_finde", "dens_x_weekend",
]
FEATS = FEATS_NUM + ["road_class"]
RAMP = np.array([(54,168,82),(235,210,60),(245,140,40),(232,60,40),(150,15,45)], dtype=np.uint8)
CUTS = np.array([0.20, 0.35, 0.52, 0.72])


def prep(df):
    X = df[FEATS].copy()
    for c in FEATS:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def load_all():
    import os as _os
    RES = int(_os.environ.get("ARDENTIA_RES", "9"))
    suf = "" if RES == 9 else f"_res{RES}"
    df = pd.read_parquet(f"data/processed/features/v1{suf}/dataset.parquet")
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].copy()
    cells = pd.read_parquet(f"data/processed/h3_cells/res{RES}_static.parquet")
    cells["aspect_sin"] = np.sin(np.radians(cells["aspect_deg"]))
    cells["aspect_cos"] = np.cos(np.radians(cells["aspect_deg"]))
    cells["dist_to_road_log"] = np.log1p(cells["dist_to_road"].clip(0, 5000))
    cells["road_density_log"] = np.log1p(cells["road_density_1km"])
    return df, cells


def build_day_features(day: pd.Timestamp, df_tr_cells: pd.DataFrame, cells: pd.DataFrame,
                       met: pd.DataFrame, s2_cache: dict, fire_tree, ig_dates) -> pd.DataFrame:
    """Features para TODAS las celdas en un día concreto (sin look-ahead)."""
    d = cells.copy()
    year = day.year
    # met: IDW de las 3 estaciones más cercanas (pivot del día)
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    dd, ii = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=3)
    w = 1.0 / np.maximum(dd, 1e-5) ** 2
    w = w / w.sum(axis=1, keepdims=True)
    day_m = met[met["fecha"] == day]
    for var in FEATS_NUM:
        if var in ("ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom",
                   "es_finde", "dens_x_weekend", "dias_ultimo_fuego_cercano",
                   "dist_ultimo_fuego_m", "ardio_1km_1an"):
            continue  # cubiertos abajo
        if var not in day_m.columns:
            d[var] = np.nan
            continue
        vals = (day_m.drop_duplicates("idema")
                .set_index("idema")[var].reindex(st_meta["idema"]).to_numpy())
        vv = vals[ii]
        d[var] = np.nansum(vv * w, axis=1) / np.maximum((~np.isnan(vv)).astype(float).sum(axis=1), 1)

    # calendario
    d["es_finde"] = int(day.dayofweek >= 5)
    d["dens_x_weekend"] = d["road_density_1km"].fillna(0) * d["es_finde"]

    # satélite: composite del verano del año anterior (cache por año)
    if year - 1 not in s2_cache:
        import rasterio
        tif = Path(f"data/raw/s2/s2_summer_{year-1}.tif")
        if tif.exists():
            with rasterio.open(tif) as src:
                from pyproj import Transformer
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xs, ys = tr.transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
                s2_cache[year - 1] = np.array([x for x in src.sample(list(zip(xs, ys)))], dtype="float32")
        else:
            s2_cache[year - 1] = np.full((len(cells), 3), np.nan, dtype="float32")
    s2 = s2_cache[year - 1]
    d["ndmi_summer"], d["ndvi_summer"], d["nbr_summer"] = s2[:, 0], s2[:, 1], s2[:, 2]
    d["ndmi_anom"] = s2[:, 0] - np.nanmean(s2[:, 0])  # anomalía local vs media territorial

    # recencia: días desde el último fuego <5 km ANTES de `day`
    ang = 5000.0 / 6_371_000.0
    ll = np.radians(cells[["lat", "lon"]].to_numpy())
    hits = fire_tree.query_ball_point(ll, ang, workers=-1)
    day_np = np.datetime64(day.date())
    rec_days = np.full(len(cells), 1825.0)
    for i, hs in enumerate(hits):
        if len(hs) == 0:
            continue
        prev = ig_dates[hs]
        prev = prev[prev < day_np]
        if len(prev):
            rec_days[i] = min((day_np - prev.max()) / np.timedelta64(1, "D"), 1825)
    d["dias_ultimo_fuego_cercano"] = rec_days
    d["dist_ultimo_fuego_m"] = np.where(rec_days < 1825, 500.0, 5000.0)  # aproximación
    d["ardio_1km_1an"] = (rec_days <= 365).astype(int)
    d["fecha"] = day
    return d


def main(year: int = 2023):
    import os as _os
    globals()["RES_G"] = int(_os.environ.get("ARDENTIA_RES", "9"))
    import matplotlib.colors as mcolors

    df, cells = load_all()
    met = load_met_stations().dropna(subset=["lat", "lon"])
    # entrenar fino en train (≤2022)
    tr = (df["fecha"].dt.year <= 2022) & df["fwi"].notna()
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    y_big = ((df["label"] == 1) & (surf >= 10)).astype(int)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100,
                           scale_pos_weight=max((y_big[tr] == 0).sum() / max((y_big[tr] == 1).sum(), 1), 0.1),
                           random_state=42, verbosity=-1)
    m.fit(prep(df)[tr], y_big[tr])

    # calibración isotónica sobre val 2022 — con modelo AUXILIAR entrenado SOLO ≤2021:
    # las predicciones de val son entonces out-of-sample (la CDF de referencia no puede
    # estar degenerada por memorización; ese era el bug del todo-rojo)
    val22 = (df["fecha"].dt.year == 2022) & df["fwi"].notna()
    tr21 = (df["fecha"].dt.year <= 2021) & df["fwi"].notna()
    from sklearn.isotonic import IsotonicRegression
    m_aux = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=100,
        scale_pos_weight=max((y_big[tr21] == 0).sum() / max((y_big[tr21] == 1).sum(), 1), 0.1),
        random_state=42, verbosity=-1)
    m_aux.fit(prep(df)[tr21], y_big[tr21])
    raw_val = m_aux.predict_proba(prep(df)[val22])[:, 1]
    # BAREMO FIJO sobre el SCORE crudo (como los umbrales del FWI): cuantiles de las
    # predicciones out-of-sample de la TEMPORADA (may-oct) de val-2022
    val_mask_summer = val22 & df["fecha"].dt.month.isin([5, 6, 7, 8, 9, 10])
    raw_val_summer = m_aux.predict_proba(prep(df)[val_mask_summer])[:, 1]
    cut_scores = np.nanquantile(raw_val_summer, [0.75, 0.92, 0.98, 0.997])
    if len(np.unique(cut_scores)) < 4:
        cut_scores = np.array([0.002, 0.006, 0.015, 0.05])
    print("baremo fijo (score):", cut_scores.round(4))

    from ardintra.features.spatial_smooth import get_smooth_matrix, smooth
    W_smooth = get_smooth_matrix(list(cells["h9"]))

    # CORTES FIJOS EN PROBABILIDAD (estimados una vez de la referencia):
    def nivel_de(p_day):
        return np.digitize(p_day, cut_scores)

    ig = pd.read_parquet("data/interim/egif/parts_normalized.parquet", columns=None)
    ig = ig[(ig["in_bbox"] == True) & (ig["anio"] >= 2017)]  # noqa: E712
    ig = ig[ig["superficie_quemada_ha"] >= 0.1]
    fire_tree = cKDTree(np.radians(ig[["latitud", "longitud"]].to_numpy()))
    ig_dates = pd.to_datetime(ig["deteccion"]).to_numpy().astype("datetime64[D]")

    FRAMES.mkdir(parents=True, exist_ok=True)
    s2_cache: dict = {}
    cmap = mcolors.ListedColormap(RAMP / 255.0)
    cmap.set_bad(color="white", alpha=0)

    dias = pd.date_range(f"{year}-06-01", f"{year}-10-31", freq="D")
    for n_day, day in enumerate(dias):
        png = FRAMES / f"frame_{n_day:03d}.png"
        if png.exists():
            continue
        feats = build_day_features(day, df, cells, met, s2_cache, fire_tree, ig_dates)
        p = m.predict_proba(prep(feats))[:, 1]   # score crudo del modelo
        # suavizado espacial: campo continuo (vecinos comparten clima/combustible);
        # propio 0.5 · anillo-1 0.35 · anillo-2 0.15 — elimina el sal y pimienta
        p = smooth(p, W_smooth)
        # NIVELES FIJOS por probabilidad calibrada — iguales todos los días, comparables
        nivel = nivel_de(p)
        fires_day = ig[pd.to_datetime(ig["deteccion"]).dt.normalize() == day]

        fig, ax = plt.subplots(figsize=(11, 9), dpi=110)
        RES = int(__import__("os").environ.get("ARDENTIA_RES", "9"))
        CELL_S = 2.6 if RES == 9 else 16
        sc = ax.scatter(cells["lon"], cells["lat"], c=nivel, cmap=cmap, s=CELL_S,
                        vmin=0, vmax=4, alpha=0.9, linewidths=0.08, edgecolors=(0, 0, 0, 0.35))
        if len(fires_day):
            big = fires_day[fires_day["superficie_quemada_ha"] >= 10]
            small = fires_day[fires_day["superficie_quemada_ha"] < 10]
            ax.scatter(small["longitud"], small["latitud"], marker="o", s=22, c="blue", alpha=0.9, zorder=5)
            ax.scatter(big["longitud"], big["latitud"], marker="*", s=240, c="darkred",
                       edgecolors="white", linewidths=0.8, zorder=6, label=f"fuegos ≥10 ha: {len(big)}")
            ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
            if len(big):   # INSET de zoom sobre el mayor fuego del día
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
        ax.set_title(f"WildfireRisk-ES · {day.date()} · prob. calibrada (cortes fijos)", fontsize=12)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xlim(-9.65, -6.60); ax.set_ylim(41.75, 43.95)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.55, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(["Bajo", "Moderado", "Alto", "Muy alto", "Extremo"])
        plt.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        if n_day % 15 == 0:
            print(f"frame {n_day}/{len(dias)} ({day.date()})", flush=True)

    # mp4
    RES = int(__import__("os").environ.get("ARDENTIA_RES", "9"))
    out = f"reports/simulacion_{year}_res{RES}.mp4"
    os.system(f"ffmpeg -y -framerate 1/3 -i {FRAMES}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 23 {out}")
    print(f"-> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
