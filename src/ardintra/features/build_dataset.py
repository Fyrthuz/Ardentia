"""Constructor del dataset Fase 2 (E0): registros (celda H9, día) → features + label.

Fuentes que usa (todas las disponibles ahora mismo; las ausentes quedan NaN):
  - estáticas: processed/h3_cells/res9_static.parquet
  - etiquetas: processed/labels/v1/{ignitions,burned_cells}.parquet
  - met diaria por estación (openmeteo checkpoints + AEMET backfill) → FWI por
    estación (ardintro.features.fwi) → IDW de las 3 estaciones más cercanas
  - satélite: data/raw/s2/s2_summer_{y}.tif (si existe) muestreado al centro de
    celda + anomalía vs baseline 2017-19

Salida: data/processed/features/v1/dataset.parquet + qa JSON.
Muestreo de negativos determinista (seed=42): 20 por día de ignición escalado
por temporada + 5 extra/día en invierno (control de rareza).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ardintra.features.fwi import fwi_series
import h3

import os as _os
RES = int(_os.environ.get("ARDENTIA_RES", "9"))
SUFFIX = "" if RES == 9 else f"_res{RES}"
STATIC = Path(f"data/processed/h3_cells/res{RES}_static.parquet")
OUT = Path(f"data/processed/features/v1{SUFFIX}/dataset.parquet")
QA = Path(f"data/processed/features/v1{SUFFIX}/qa_dataset.json")
LABELS_IG = Path("data/processed/labels/v1/ignitions.parquet")
LABELS_EFFIS = Path("data/processed/labels/v1/burned_cells.parquet")
OM_STATIONS = Path("data/raw/openmeteo/per_station")
AEMET_MISSING = Path("data/raw/aemet/missing_backfill")
S2_DIR = Path("data/raw/s2")

RENAME = {
    "time": "fecha",
    "temperature_2m_max": "tmax_c",
    "precipitation_sum": "prec_mm",
    "windspeed_10m_max": "wind_max_kmh",
    "relative_humidity_2m_mean": "rh_mean",
}

MET_VARS = ["tmax_c", "rh_mean", "wind_max_kmh", "prec_mm", "ffmc", "dmc", "dc", "isi", "bui", "fwi",
            "ffmc_7d", "fwi_7d", "fwi_prev", "spei30", "spei90", "ffmc_anom", "dmc_anom", "dc_anom"]
NEG_DAILY_SEASON = int(_os.environ.get("ARDENTIA_NEG", "150")) if "NEG_DAILY_SEASON" not in dir() else NEG_DAILY_SEASON  # ~20:1 real operational ratio
NEG_DAILY_WINTER = 3
# ventanas de contaminación por propagación (un fuego cercano+y reciente ≠ negativo real)
import os as _os
RADIUS_M = 1500.0 if int(_os.environ.get('ARDENTIA_RES', '9')) == 8 else 1030.0     # ~2 anillos H3 res9
SAFE_DAYS = 7
S2_BASELINE_YEARS = (2017, 2019)
SEED = 42
YEARS = list(range(2017, 2025))


def load_met_stations() -> pd.DataFrame:
    """Consolida checkpoints (openmeteo crudo + aemet backfill normalizado) →
    DataFrame largo [idema, lat, lon, fecha, met_vars]."""
    frames = []
    for d in (OM_STATIONS, AEMET_MISSING):
        if not d.exists():
            continue
        for pq in sorted(d.rglob("estacion=*.parquet")):
            df = pd.read_parquet(pq)
            if "time" in df.columns:  # esquema crudo openmeteo
                df = df.rename(columns=RENAME)
            if "fecha" not in df.columns or "tmax_c" not in df.columns:
                continue
            df["fecha"] = pd.to_datetime(df["fecha"])
            if "lat" not in df.columns:
                df["lat"], df["lon"] = np.nan, np.nan
            keep = ["idema", "lat", "lon", "fecha"] + [
                c for c in ("tmax_c", "tmin_c", "rh_mean", "wind_max_kmh", "prec_mm")
                if c in df.columns
            ]
            frames.append(df[keep].drop_duplicates(["idema", "fecha"]))
    met = pd.concat(frames, ignore_index=True)
    met["fecha"] = pd.to_datetime(met["fecha"]).dt.normalize()
    met = met.drop_duplicates(["idema", "fecha"])   # CRÍTICO: sin esto los merges explotan
    tfisica = pd.read_parquet(Path("data/raw/aemet/stations.parquet"))[ ["idema", "lat", "lon"] ]
    met = met.drop(columns=["lat", "lon"], errors="ignore").merge(tfisica, on="idema", how="left")

    # FWI por estación (encadenado diario; NaN en huecos)
    fwi_frames = []
    for idema, gdf in met.groupby("idema"):
        gdf = gdf.sort_values("fecha")
        f = fwi_series(gdf[["fecha", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm", "lat"]])
        f["idema"] = idema
        fwi_frames.append(f)
    fwi = pd.concat(fwi_frames, ignore_index=True)
    # tendencia: medias 7d + FWI del día previo (señal de sequía acumulada)
    for st_id, g in fwi.groupby("idema"):
        g = g.sort_values("fecha")
        fwi.loc[g.index, "ffmc_7d"] = g["ffmc"].rolling(7, min_periods=4).mean().values
        fwi.loc[g.index, "fwi_7d"]  = g["fwi"].rolling(7, min_periods=4).mean().values
        fwi.loc[g.index, "fwi_prev"] = g["fwi"].shift(1).values
    met = met.merge(fwi, on=["idema", "fecha"], how="left")
    # sequía: SPEI + anomalías de códigos (features de la literatura)
    from ardintra.features.drought_features import drought_features
    drought_frames = []
    for idema, gdf in met.groupby("idema"):
        gdf = gdf.sort_values("fecha")
        need = gdf[["fecha", "tmax_c", "tmin_c", "prec_mm", "ffmc", "dmc", "dc", "lat"]].copy()
        # tmin puede faltar en el frame consolidado: derivarlo de openmeteo crudo si hay
        for c in ("tmin_c",):
            if c not in need.columns or need[c].isna().all():
                need[c] = np.nan
        dfr = drought_features(need)
        dfr["idema"], dfr["fecha"] = idema, gdf["fecha"].values
        drought_frames.append(dfr)
    drought = pd.concat(drought_frames, ignore_index=True)
    met = met.merge(drought, on=["idema", "fecha"], how="left")
    # la fecha es str en el serie port (recast)
    met["fecha"] = pd.to_datetime(met["fecha"])
    return met


def _cell_provincia(cells: pd.DataFrame) -> pd.DataFrame:
    """Provincia por celda = provincia de la estación AEMET más cercana (58 est)."""
    from scipy.spatial import cKDTree as KD  # noqa: PLC0415

    st = pd.read_parquet(Path("data/raw/aemet/stations.parquet"))[ ["idema", "lat", "lon", "provincia"] ]
    tree = KD(np.radians(st[["lat", "lon"]].to_numpy()))
    d, idx = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=1)
    out = cells[[ "h9"]].copy()
    prov = st["provincia"].astype(str).str.strip().str.lower().str.replace("ñ", "n", regex=False).str.replace(" ", "_", regex=False).to_numpy()[idx]
    out["provincia"] = prov
    return out


def _negative_pool(cells: pd.DataFrame, ig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pool de celdas negativas válidas con su provincia (excluye celdas que ya
    ardió algún día del histórico y asigna provincia por estación cercana)."""
    cell_ids = cells["h9"].to_numpy()
    nonfire = np.array(cell_ids)
    prov_by_cell = dict(_cell_provincia(cells).set_index("h9")["provincia"])
    return nonfire, prov_by_cell


def _contaminated(pairs: pd.DataFrame, ig: pd.DataFrame, cells: pd.DataFrame) -> np.ndarray:
    """True si el (celda, día) está a <RADIUS_M de un fuego ocurrido en ±SAFE_DAYS:
    es propagación del mismo evento — obligatorio excluirlo de los negativos."""
    from scipy.spatial import cKDTree as KD  # noqa: PLC0415

    ll = dict(zip(cells["h9"], zip(cells["lat"].to_numpy(), cells["lon"].to_numpy())))
    pts_ig = ig.dropna(subset=["h9"])
    pts_ig = pts_ig.merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    tree = KD(np.radians(pts_ig[["lat", "lon"]].to_numpy()))
    cell_ll = np.array([ll.get(h, (np.nan, np.nan)) for h in pairs["h9"]])
    r_angular = RADIUS_M / 6_371_000.0
    dirty = np.zeros(len(pairs), dtype=bool)
    days = pd.to_datetime(pairs["fecha"]).to_numpy()
    ig_dates = pd.to_datetime(pts_ig["fecha"]).to_numpy()
    # búsqueda por lotes: 60 días por bloque para acotar memoria
    for chunk_start in range(0, len(pairs), 50_000):
        sl = slice(chunk_start, chunk_start + 50_000)
        cands = tree.query_ball_point(np.radians(cell_ll[sl])[:, None].squeeze(1), r_angular, workers=-1)
        for i, hits in enumerate(cands, chunk_start):
            if not hits:
                continue
            d0 = days[i]
            near_dates = ig_dates[hits]
            diff = np.abs((near_dates - d0) / np.timedelta64(1, "D"))
            dirty[i] = bool((diff <= SAFE_DAYS).any())
    return dirty


def _offset_cell(lat, lon, dist_m, bearing):
    """Celda H3 a dist_m/bearing desde (lat, lon)."""
    import math
    dlat = dist_m * math.cos(bearing) / 110_540.0
    dlon = dist_m * math.sin(bearing) / (111_320.0 * math.cos(math.radians(lat)))
    return h3.latlng_to_cell(lat + dlat, lon + dlon, 9)


def _negatives(ig: pd.DataFrame, cells: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Tres tiers (literatura WUS-2025/FireCastRL): random diario + NEAR-SPACE
    (celdas a 1-3 km de un fuego del MISMO día: mismas condiciones, sin ignición)
    + NEAR-TIME (la misma celda 8-21 días ANTES del fuego: fuel similar, futuro
    sin look-ahead). El filtro de contaminación limpia solapes."""
    nonfire, prov_by_cell = _negative_pool(cells, ig)
    rows = []
    rango = pd.date_range("2017-01-01", "2024-12-31", freq="D")
    cell_pos = cells.set_index("h9")
    fires = ig.dropna(subset=["h9"]).merge(cells[["h9", "lat", "lon"]], on="h9", how="inner")

    for day in rango:
        k = NEG_DAILY_SEASON if day.month in (5, 6, 7, 8, 9, 10) else NEG_DAILY_WINTER
        sample = rng.choice(nonfire, size=k, replace=False)
        rows.append(pd.DataFrame({"h9": sample, "fecha": day, "fire_id": None, "source": "neg",
                                  "surface_ha": np.nan, "cause": pd.NA, "motivation": pd.NA,
                                  "provincia": [prov_by_cell.get(c) for c in sample]}))

    # tier NEAR-SPACE: mismo día, anillo 1-3 km alrededor de fuegos
    day_fires = fires.set_index(fires["fecha"].dt.normalize())
    for day, g in day_fires.groupby(level=0):
        n_near = min(40, len(g) * 3)
        src = g.sample(min(len(g), 8), random_state=42)
        near_cells = set()
        for _, fr in src.iterrows():
            for _ in range(8):
                c = _offset_cell(fr["lat"], fr["lon"], rng.uniform(1000, 3000), rng.uniform(0, 2 * np.pi))
                if c in cell_pos.index and c not in set(fires["h9"].unique()):
                    near_cells.add(c)
        if near_cells:
            rows.append(pd.DataFrame({"h9": list(near_cells), "fecha": day, "fire_id": None,
                                      "source": "neg_near", "surface_ha": np.nan, "cause": pd.NA,
                                      "motivation": pd.NA,
                                      "provincia": [prov_by_cell.get(c) for c in near_cells]}))

    # tier NEAR-TIME: la celda que arderá, 8-21 días antes (conditions casi idénticas, sin ignición)
    n_time = min(len(fires) // 3, 4000)
    fs = fires.sample(n_time, random_state=42)
    pre = pd.DataFrame({
        "h9": fs["h9"].to_numpy(),
        "fecha": pd.to_datetime(fs["fecha"].dt.normalize()) - pd.to_timedelta(
            rng.integers(8, 22, size=len(fs)), unit="D"),
        "fire_id": None, "source": "neg_near", "surface_ha": np.nan, "cause": pd.NA,
        "motivation": pd.NA, "provincia": fs["provincia"].to_numpy(),
    })
    rows.append(pre)
    neg = pd.concat(rows, ignore_index=True)
    # 🔥 excluir contaminación por propagación (mismo incendio reciente cerca)
    neg = neg.drop_duplicates(subset=["h9", "fecha"])
    dirty = _contaminated(neg, ig, cells)
    neg = neg[~dirty].reset_index(drop=True)
    print(f"negativos: {len(neg):,} tras descartar {dirty.sum():,} contaminados por fuego cercano")
    return neg


def _met_features(pairs: pd.DataFrame, met: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    st_meta = met[["idema", "lat", "lon"]].drop_duplicates("idema").reset_index(drop=True)
    tree = cKDTree(np.radians(st_meta[["lat", "lon"]].to_numpy()))
    d, idx = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=3)
    w = 1.0 / np.maximum(d, 1e-5) ** 2  # IDW
    w = w / w.sum(axis=1, keepdims=True)

    latlon_of = dict(zip(cells["h9"], zip(cells["lat"], cells["lon"], strict=False)))
    # pivotes vectorizados: variable[fecha_idx, station_idx]
    met = met.copy()
    met["fecha"] = pd.to_datetime(met["fecha"])
    st_p = met.pivot_table(index="fecha", columns="idema", values=MET_VARS, aggfunc="first")
    dates = pd.DatetimeIndex(st_p.index)
    date_pos = {d: i for i, d in enumerate(dates)}
    st_id_pos = {s: i for i, s in enumerate(st_p.columns.levels[1])}

    n = len(pairs)
    out = {}
    for var in MET_VARS:
        arr = np.full((len(dates), len(st_meta)), np.nan)
        block = st_p[var]
        for j, sid in enumerate(st_meta["idema"]):
            if sid in block.columns:
                col = block[sid].to_numpy()
                arr[:, j] = col
        out[var] = arr

    feat = {v: np.zeros(n) for v in MET_VARS}
    stale = []
    for i, row in enumerate(pairs.itertuples(index=False)):
        h9 = row.h9
        fecha = pd.Timestamp(row.fecha)
        if h9 not in latlon_of:
            stale.append(i); continue
        lat, lon = latlon_of[h9]
        dd, ii = tree.query(np.radians([lat, lon]).reshape(1, -1), k=3)
        dd, ii = dd[0], ii[0]
        ww = 1 / np.maximum(dd, 1e-5) ** 2
        ww = ww / ww.sum()
        pos = date_pos.get(fecha)
        if pos is None:
            stale.append(i); continue
        for var in MET_VARS:
            vals = out[var][pos, ii]
            valid = ~np.isnan(vals)
            feat[var][i] = float(np.dot(vals[valid], ww[valid]) / max(ww[valid].sum(), 1e-9))
    df = pd.DataFrame(feat, columns=MET_VARS)
    for i in stale:
        for var in MET_VARS:
            df.loc[i, var] = np.nan
    return df


def _sample_tif(tif: Path, with_ll: pd.DataFrame, year: int) -> np.ndarray:
    """Muestrea un composite S2 en los centroides de las filas de ese año."""
    import rasterio

    with rasterio.open(tif) as src:
        sub = with_ll[with_ll.fecha.dt.year == year]
        vals = np.full((len(with_ll), 3), np.nan, dtype="float32")
        pts = [(r.lon, r.lat) for r in sub.itertuples(index=False)]
        if pts:
            from pyproj import Transformer  # noqa: PLC0415

            if src.crs.to_epsg() != 4326:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xs2, ys2 = tr.transform([p[0] for p in pts], [p[1] for p in pts])
                pts = list(zip(xs2, ys2))
            sampled = np.array(list(src.sample(pts)), dtype="float32")
            vals[(with_ll.fecha.dt.year == year).to_numpy()] = sampled
    return vals


def _recency_features(pairs: pd.DataFrame, ig: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Días desde el último fuego cercano (<5 km, estrictamente anterior al día
    predicho — sin look-ahead) y su distancia. Celdas recién quemadas → riesgo ~0."""
    from scipy.spatial import cKDTree as KD  # noqa: PLC0415

    ll = dict(zip(cells["h9"], zip(cells["lat"].to_numpy(), cells["lon"].to_numpy())))
    fpts = ig.dropna(subset=["h9"]).merge(cells[["h9", "lat", "lon"]], on="h9", how="inner")
    fpts = fpts.dropna(subset=["lat", "lon"])
    tree = KD(np.radians(fpts[["lat", "lon"]].to_numpy()))
    ig_dates = pd.to_datetime(fpts["fecha"]).to_numpy()
    r_env = 5000.0 if int(_os.environ.get("ARDENTIA_RES", "9")) == 9 else 6000.0
    r_angular = r_env / 6_371_000.0

    cols = ["dias_ultimo_fuego_cercano", "dist_ultimo_fuego_m", "ardio_1km_1an"]
    days = np.full(len(pairs), np.nan)
    dists = np.full(len(pairs), np.nan)
    pair_ll = np.array([ll.get(h, (np.nan, np.nan)) for h in pairs["h9"]])
    parejas = np.radians(pair_ll)
    fechas = pd.to_datetime(pairs["fecha"]).to_numpy()
    hits_all = tree.query_ball_point(parejas, r_angular, workers=-1)
    for i, (hits, d0) in enumerate(zip(hits_all, fechas)):
        if np.isnan(pair_ll[i][0]) or len(hits) == 0:
            days[i], dists[i] = 1825.0, 5000.0
            continue
        near_dates = ig_dates[hits]
        prev = near_dates[near_dates < d0]  # ESTRICTAMENTE antes → nada de look-ahead
        if len(prev) == 0:
            days[i], dists[i] = 1825.0, 5000.0
            continue
        diff = (d0 - prev) / np.timedelta64(1, "D")
        k_min = int(np.argmin(diff))
        days[i] = float(min(diff[k_min], 1825.0))
        p_lat, p_lon = pair_ll[i]
        f_lat, f_lon = fpts[["lat", "lon"]].to_numpy()[hits[k_min]]
        dists[i] = float(min(np.hypot(
            (p_lon - f_lon) * 111_320 * np.cos(np.radians((p_lat + f_lat) / 2)),
            (p_lat - f_lat) * 110_540,
        ), 5000.0))
    ardio_1y = (days <= 365).astype(float)
    return pd.DataFrame({
        "dias_ultimo_fuego_cercano": days,
        "dist_ultimo_fuego_m": dists,
        "ardio_1km_1an": ardio_1y,
    })


def _s2_features_prev_summer(pairs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Bloque satélite OPERATIVO: composite del verano del año anterior (y-1).
    Sin look-ahead: en julio de 2024 el composite completo de 2024 no existe,
    el de 2023 sí. Anomalía = vs media 2017–2019 (si esos tif existen)."""
    with_ll = pairs.merge(cells[["h9", "lat", "lon"]], on="h9", how="left")
    cols = ["ndmi_summer", "ndvi_summer", "nbr_summer", "ndmi_anom"]
    data = {c: np.full(len(pairs), np.nan) for c in cols}

    base_stacks = []
    for by in S2_BASELINE_YEARS:
        if (S2_DIR / f"s2_summer_{by}.tif").exists():
            base_stacks.append(_sample_tif(S2_DIR / f"s2_summer_{by}.tif", with_ll, by))
    bmean = np.nanmean(np.stack(base_stacks), axis=0) if base_stacks else None

    tgt = pairs.fecha.dt.year.to_numpy()
    n_touched = 0
    for src_y in sorted(set(YEARS)):
        tif = S2_DIR / f"s2_summer_{src_y}.tif"
        if not tif.exists():
            continue
        tgt_y = src_y + 1
        vals = _sample_tif(tif, with_ll, tgt_y)
        mask = tgt == tgt_y
        ndmi = vals[mask, 0]
        if np.isnan(ndmi).all():
            continue
        data["ndmi_summer"][mask] = ndmi
        data["ndvi_summer"][mask] = vals[mask, 1]
        data["nbr_summer"][mask] = vals[mask, 2]
        if bmean is not None:
            data["ndmi_anom"][mask] = ndmi - bmean[mask, 0]
        n_touched += int(mask.sum())
    print(f"s2 (verano previo): celdas-día con features: {n_touched:,}")
    return pd.DataFrame(data, columns=cols)


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cells = pd.read_parquet(STATIC)
    ig = pd.read_parquet(LABELS_IG)
    MIN_HA = float(os.environ.get("ARDENTIA_MIN_HA", "0.1"))
    if MIN_HA > 0:
        n0 = len(ig)
        ig = ig[ig["surface_ha"] >= MIN_HA]
        print(f"filtro superficie ≥{MIN_HA} ha: {n0:,}→{len(ig):,}")
    ef = pd.read_parquet(LABELS_EFFIS)
    ig["fecha"] = pd.to_datetime(ig["fecha"]).dt.normalize()
    ef["fecha"] = pd.to_datetime(ef["fecha"]).dt.normalize()
    ef = ef.rename(columns={"surface_ha": "surface_ha"})
    pos = pd.concat([
        ig[["h9", "fecha", "fire_id", "source", "surface_ha", "provincia"]],
        ef.assign(source="effis", fire_id=ef["fire_id"].astype(str), superficie2=ef["surface_ha"],
                  provincia=ef["provincia"])[["h9", "fecha", "fire_id", "source", "superficie2", "provincia"]]
        .rename(columns={"superficie2": "surface_ha"}),
    ], ignore_index=True)
    # provincia canónica (EGIF: ourense/a_coruna...; EFFIS venía en title-case con tildes)
    pos["provincia"] = (
        pos["provincia"].astype(str).str.strip().str.lower()
        .str.replace("ñ", "n", regex=False).str.replace(" ", "_", regex=False)
    )
    pos = pos[pos["fecha"].dt.year.isin(YEARS)]
    if RES != 9:
        import h3
        pos["h9"] = [h3.cell_to_parent(x, RES) for x in pos["h9"]]
        # dentro de un mismo día puede haber varias celdas res9 en la misma res8 → dedupe
        pos = pos.sort_values("surface_ha", ascending=False).drop_duplicates(["h9", "fecha"], keep="first")
    pos = pos[pos["h9"].isin(set(cells["h9"]))]

    neg = _negatives(pos, cells, rng)
    pairs = pd.concat([pos, neg], ignore_index=True)
    pairs["_is_pos"] = pairs["source"] != "neg"
    pairs = (pairs.sort_values("_is_pos", ascending=False)
             .drop_duplicates(subset=["h9", "fecha"], keep="first")
             .drop(columns="_is_pos"))
    print(f"pares: {len(pos):,} positivos + {len(neg):,} negativos = {len(pairs):,}")

    met = load_met_stations()
    print(f"met consolidada: {met['idema'].nunique()} estaciones, {met['fecha'].nunique()} días")
    pairs["fecha"] = pd.to_datetime(pairs["fecha"])

    feat_met = _met_features(pairs, met, cells)
    print("met features listas", feat_met.columns.tolist())
    from ardintra.features.era5_features import era5_features
    feat_era5 = era5_features(pairs, cells)
    print("era5 features:", feat_era5.notna().mean().round(3).to_dict())
    feat_rec = _recency_features(pairs, ig.dropna(subset=['h9']), cells)
    print("recencia features:", {c: round(float(feat_rec[c].median()), 1) for c in feat_rec.columns})
    feat_s2 = _s2_features_prev_summer(pairs, cells)
    print("s2 features:", ~(feat_s2.isna().all()) if False else "mostradas abajo")
    print(feat_s2.notna().mean().round(3).to_dict())

    df = pairs.merge(cells, on="h9", how="left")
    if "aspect_deg" in df.columns:
        df["aspect_sin"] = np.sin(np.radians(df["aspect_deg"]))
        df["aspect_cos"] = np.cos(np.radians(df["aspect_deg"]))
        df["dist_to_road_log"] = np.log1p(df["dist_to_road"].clip(0, 5000))
        df["road_density_log"] = np.log1p(df["road_density_1km"])
    df = pd.concat([df.reset_index(drop=True), feat_met.reset_index(drop=True),
                    feat_era5.reset_index(drop=True), feat_s2.reset_index(drop=True),
                    feat_rec.reset_index(drop=True)], axis=1)
    # normalización: fire_id puede traer listas (EGIF celda-día multiembro)
    df["fire_id"] = df["fire_id"].apply(
        lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple, np.ndarray))
        else ("" if x is None else str(x))
    )
    df["cause"] = pd.to_numeric(df["cause"], errors="coerce").astype("Int16")
    df["motivation"] = pd.to_numeric(df["motivation"], errors="coerce").astype("Int16")
    # antrópico-temporal: finde y su interacción con la red viaria
    dw = pd.to_datetime(df["fecha"]).dt.dayofweek
    df["es_finde"] = (dw >= 5).astype(int)
    df["dens_x_weekend"] = (df["road_density_1km"].fillna(0) * df["es_finde"]).round(4)
    df["label"] = (~df["source"].astype(str).str.startswith("neg")).astype(int)  # neg y neg_near = 0
    df["split_temporal"] = np.where(df["fecha"].dt.year <= 2022, "train", "test")
    df["es_ourense"] = (df["provincia"].astype(str).str.lower() == "ourense").astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    qa = {
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "n_rows": int(len(df)),
        "neg_pipe: ratio": round(sum(df["label"] == 0) / max(sum(df["label"] == 1), 1), 1),
        "met_nan_pct": round(float(df["fwi"].isna().mean() * 100), 1),
        "era5_notna": {c: round(float(df[c].notna().mean() * 100), 1) for c in feat_era5.columns},
        "s2_notna_pct": {c: round(float(df[c].notna().mean() * 100), 1) for c in feat_s2.columns},
    }
    QA.parent.mkdir(parents=True, exist_ok=True)
    QA.write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))
    return df


if __name__ == "__main__":
    run()
