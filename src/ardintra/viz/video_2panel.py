"""Vídeo de simulación 2-paneles: predicción CELDA (res9) + REGIÓN (res7 parent).

Izquierda: celdas res9 coloreadas por P(fire) calibrada — detalle fino
Derecha: regiones res7 coloreadas por max(P) de sus celdas — vista operativa
Centro: fuegos reales del día (⭐ ≥10 ha, 🔵 <10 ha)

2 segundos por frame.
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
NAMES = ["Bajo", "Moderado", "Alto", "Muy alto", "Extremo"]

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


def prep(df):
    X = df[[c for c in FEATS if c in df.columns]].copy()
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            X[c] = X[c].astype("category")
    return X


def build_day_feats(day, cells, met, s2_cache, fire_tree, ig_dates):
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
    return d


def main(year: int = 2023):
    from ardintra.features.build_dataset import load_met_stations

    df = pd.read_parquet("data/processed/features/v1/dataset.parquet")
    surf = pd.to_numeric(df["surface_ha"], errors="coerce").fillna(0)
    df["big10"] = ((df["label"] == 1) & (surf >= 10)).astype(int)
    df = df[df["fwi"].notna()].reset_index(drop=True).copy()
    cells = pd.read_parquet("data/processed/h3_cells/res9_static.parquet")
    cells["aspect_sin"] = np.sin(np.radians(cells["aspect_deg"]))
    cells["aspect_cos"] = np.cos(np.radians(cells["aspect_deg"]))
    cells["dist_to_road_log"] = np.log1p(cells["dist_to_road"].clip(0, 5000))
    cells["road_density_log"] = np.log1p(cells["road_density_1km"])

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
    X_all = prep(df)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                           min_child_samples=100, scale_pos_weight=spw,
                           random_state=42, verbosity=-1)
    m.fit(X_all[tr], y_big[tr])
    p_raw_all = m.predict_proba(X_all)[:, 1]

    # calibración OOS
    tr21 = (df["fecha"].dt.year <= 2021) & df["fwi"].notna()
    m_aux = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, scale_pos_weight=spw,
                               random_state=42, verbosity=-1)
    m_aux.fit(X_all[tr21], y_big[tr21])
    raw_val = m_aux.predict_proba(X_all[val22])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_big[val22])

    # referencia = temporada train
    ref_pool = df[(df["fecha"].dt.year <= 2022) & (df["fecha"].dt.month.isin([5,6,7,8,9,10]))]
    p_ref_cal = iso.predict(m.predict_proba(prep(ref_pool))[:, 1])
    val_sorted = np.sort(p_ref_cal)
    n_ref = len(val_sorted)

    # suavizado
    p_smooth_all = p_raw_all
    p_cal_all = iso.predict(p_smooth_all)

    # recalibración REGIONAL: Stage 2 (GLM + Moran Eigenvector Maps, reports/stage2_model.npz)
    # μ_r = exp(α + β·log(1+λ_raw) + MEM·γ) → P(≥1) = 1-exp(-μ); fallback: Poisson cruda
    REG_PARQ = Path("reports/fullgrid_regional.parquet")
    S2_MODEL = Path("reports/stage2_model.npz")
    iso_reg = None
    s2 = None
    base_rate_reg = 0.0003
    if S2_MODEL.exists() and REG_PARQ.exists():
        s2m = np.load(S2_MODEL, allow_pickle=True)
        s2_regions = list(s2m["regions"]); s2_mem = s2m["mem"]
        s2_coef, s2_b0 = s2m["coef"], float(s2m["intercept"])
        s2_idx = {r: i for i, r in enumerate(s2_regions)}
        regall = pd.read_parquet(REG_PARQ)
        base_rate_reg = float((regall["n_fuegos_obs"] > 0).mean())
        ri_all = regall["r7"].map(s2_idx).to_numpy()
        ok = ri_all >= 0
        feat = np.zeros((len(regall), 1 + s2_mem.shape[1]))
        feat[ok, 0] = np.log1p(regall["lam"].to_numpy()[ok])
        feat[ok, 1:] = s2_mem[ri_all[ok]]
        p_ref = 1 - np.exp(-np.exp(s2_b0 + feat @ s2_coef))
        reg_ref_thr = list(np.quantile(p_ref, [0.50, 0.90, 0.97]))
        print(f"Stage 2 cargado ({s2_mem.shape[1]} MEM) · base rate region-día = {base_rate_reg:.5f}")
    else:
        s2_regions, s2_mem, s2_coef, s2_b0, s2_idx = [], None, None, 0.0, {}
        reg_ref_thr = [0.0001, 0.0003, 0.0008]
        print("AVISO: sin reports/stage2_model.npz — fallback Poisson cruda")

    # niveles por percentil vs referencia de temporada
    pct_rank_all = np.searchsorted(val_sorted, p_cal_all, side="right") / n_ref
    nivel_all = np.digitize(pct_rank_all, [0.40, 0.75, 0.90, 0.97])
    colors_all = RAMP[nivel_all]

    # región: parent res7
    df["parent7"] = [h3.cell_to_parent(x, 7) for x in df["h9"]]

    # polígonos hexagonales res9 + res7
    print("precomputando hexágonos…", flush=True)
    polys9 = [[(lng, lat) for lat, lng in h3.cell_to_boundary(h9)] for h9 in cells["h9"]]
    unique_r7 = df["parent7"].drop_duplicates().reset_index(drop=True)
    polys7 = [[(lng, lat) for lat, lng in h3.cell_to_boundary(h7)] for h7 in unique_r7]
    r7_idx = {h: i for i, h in enumerate(unique_r7)}
    print(f"polígonos: res9 {len(polys9):,} · res7 {len(polys7):,}")

    FRAMES = Path("data/processed/video_frames_2panel")
    FRAMES.mkdir(parents=True, exist_ok=True)
    cmap = mcolors.ListedColormap(RAMP / 255.0)
    s2_cache = {}

    dias = pd.date_range(f"{year}-06-01", f"{year}-10-31", freq="D")
    for n_day, day in enumerate(dias):
        png = FRAMES / f"frame_{n_day:03d}.png"
        if png.exists():
            continue

        feats = build_day_feats(day, cells, met, s2_cache, fire_tree, ig_dates)
        Xd = prep(feats)
        p_raw_day = m.predict_proba(Xd)[:, 1]
        p_smooth_day = p_raw_day
        p_cal_day = iso.predict(p_smooth_day)
        pct_rank_day = np.searchsorted(val_sorted, p_cal_day, side="right") / n_ref
        nivel_day = np.digitize(pct_rank_day, [0.40, 0.75, 0.90, 0.97])
        day_colors = RAMP[nivel_day]

        # regional: Stage 2 — μ_r = exp(α + β·log(1+λ) + MEM·γ), P(≥1) = 1-exp(-μ)
        # los feats del día tienen una fila por CELDA (no por dataset row) → 1:1 con los polígonos
        feats["p_cal_day"] = p_cal_day
        feats["parent7"] = [h3.cell_to_parent(x, 7) for x in feats["h9"]]
        reg_agg = feats.groupby("parent7")["p_cal_day"].agg(lam="sum", p_max="max")
        reg_lam = np.array([reg_agg["lam"].get(h, 0.0) for h in unique_r7])
        if s2_mem is not None:
            lam_feat = np.log1p(reg_lam)
            ri_v = np.array([s2_idx.get(h, -1) for h in unique_r7])
            mu = np.zeros(len(unique_r7))
            known = ri_v >= 0
            mu[known] = np.exp(s2_b0 + s2_coef[0] * lam_feat[known]
                               + s2_mem[ri_v[known]] @ s2_coef[1:])
            reg_p = 1.0 - np.exp(-mu)                     # P(≥1 fuego en la región)
            lam_total = float(mu.sum())                   # Nº esperado de fuegos (honesto)
        else:
            reg_p = 1.0 - np.exp(-reg_lam)
            lam_total = float(reg_lam.sum())
        # niveles por percentil de la distribución regional Stage 2
        reg_pct = np.searchsorted(np.sort(p_ref), reg_p, side="right") / len(p_ref) \
            if s2_mem is not None else reg_p / max(reg_ref_thr[-1], 1e-9)
        reg_nivel = np.digitize(reg_pct, [0.50, 0.90, 0.97])
        reg_colors = RAMP[reg_nivel]
        # celdas: percentil vs referencia de temporada (las p por celda son diminutas)
        pct_rank_day = np.searchsorted(val_sorted, p_cal_day, side="right") / n_ref
        nivel_day = np.digitize(pct_rank_day, [0.40, 0.75, 0.90, 0.97])
        day_colors = RAMP[nivel_day]

        fires_day = ig[pd.to_datetime(ig["deteccion"]).dt.normalize() == day]

        # FIGURA 2 PANELES
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=110)
        fig.suptitle(f"WildfireRisk-ES · {day.date()} · predicción calibrada", fontsize=14, y=0.98)

        # panel izquierdo: celdas res9
        pc9 = PolyCollection(polys9, facecolors=day_colors / 255.0,
                             edgecolors=(0, 0, 0, 0.1), linewidths=0.1)
        ax1.add_collection(pc9)
        ax1.set_xlim(-9.65, -6.60); ax1.set_ylim(41.75, 43.95)
        ax1.set_title("Predicción por CELDA (res9 · 105 m)", fontsize=11)
        if len(fires_day):
            big = fires_day[fires_day["superficie_quemada_ha"] >= 10]
            small = fires_day[fires_day["superficie_quemada_ha"] < 10]
            ax1.scatter(small["longitud"], small["latitud"], marker="o", s=18, c="blue", alpha=0.85, zorder=5)
            ax1.scatter(big["longitud"], big["latitud"], marker="*", s=200, c="darkred",
                       edgecolors="white", linewidths=0.7, zorder=6)
            if len(big):
                fb = big.loc[big["superficie_quemada_ha"].idxmax()]
                ax1.annotate(f"{fb['superficie_quemada_ha']:.0f} ha",
                            (fb["longitud"], fb["latitud"]), fontsize=8, color="white",
                            ha="center", va="bottom", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="darkred", alpha=0.8))

        # panel derecho: regiones res7 — P(>=1 fuego) Poisson
        pc7 = PolyCollection(polys7, facecolors=reg_colors / 255.0,
                             edgecolors=(0, 0, 0, 0.3), linewidths=0.5)
        ax2.add_collection(pc7)
        ax2.set_xlim(-9.65, -6.60); ax2.set_ylim(41.75, 43.95)
        ax2.set_title("P(≥1 fuego ≥10 ha) por REGIÓN (res7 · 5.2 km)", fontsize=11)
        # top-3 regiones del día: riesgo como múltiplo de la base rate
        # (offset vertical + flecha para evitar solape entre regiones contiguas)
        top3 = np.argsort(-reg_p)[:3]
        shown, k_off = [], 0
        ref_sorted = np.sort(p_ref) if s2_mem is not None else np.array([reg_ref_thr[-1]])
        q90 = np.quantile(ref_sorted, 0.9)
        for t in top3:
            if reg_p[t] < q90:
                continue
            lat_c, lng_c = h3.cell_to_latlng(unique_r7[t])
            if any(abs(lat_c - s[0]) < 0.09 and abs(lng_c - s[1]) < 0.09 for s in shown):
                k_off += 1
            else:
                k_off = 0
            shown.append((lat_c, lng_c))
            mult = reg_p[t] / base_rate_reg
            ax2.annotate(f"×{mult:.0f}", (lng_c, lat_c + 0.10 + 0.09 * k_off),
                         fontsize=9, fontweight="bold", color="white",
                         ha="center", va="center", zorder=7,
                         arrowprops=dict(arrowstyle="-", color="white", lw=0.8, alpha=0.7),
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65))
        if len(fires_day):
            big_r = fires_day[fires_day["superficie_quemada_ha"] >= 10]
            ax2.scatter(big_r["longitud"], big_r["latitud"], marker="*", s=200, c="darkred",
                       edgecolors="white", linewidths=0.8, zorder=6)

        # colorbar compartido
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 4))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(NAMES, fontsize=9)

        # leyenda fuegos + lambda del día
        footer = f"Nº esperado de fuegos ≥10 ha hoy: {lam_total:.2f}  ·  ×N = riesgo vs media ({base_rate_reg:.3%}/región-día)"
        if len(fires_day):
            footer += f"   ·   🔵 <10 ha ({len(fires_day)-len(big)})  ·  ⭐ ≥10 ha ({len(big)})"
        fig.text(0.5, 0.02, footer, ha="center", fontsize=10, color="white",
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

        for ax in (ax1, ax2):
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(labelsize=7)

        plt.subplots_adjust(wspace=0.05, left=0.02, right=0.90, top=0.92, bottom=0.05)
        fig.savefig(png)
        plt.close(fig)
        if n_day % 15 == 0:
            print(f"frame {n_day}/{len(dias)} ({day.date()})", flush=True)

    out = f"reports/simulacion_2023_2panel.mp4"
    os.system(f"ffmpeg -y -framerate 1/2 -i {FRAMES}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 23 {out}")
    print(f"-> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
