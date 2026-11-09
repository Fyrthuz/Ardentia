"""Mini API para el visor JS (Fase 3 preview): overlay ligero + detalle bajo demanda.

Endpoints:
  GET /                              → visor (api/static/index.html)
  GET /v1/risk/cell/{h9}?scenario=…  → ficha completa de la celda (cliqueo)
  GET /overlay/risk/{scenario}.png   → degradado rasterizado del escenario
  GET /overlay/ignitions.geojson     → puntos EGIF reales (negativos de validación)
  GET /overlay/effis.geojson         → celdas quemadas EFFIS
  GET /overlay/borders.png           → contornos H3 (capa constante)
  GET /overlay/roads_light.geojson   → vías principales simplificadas (capa toggle)
"""

from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageDraw
from pyproj import Transformer

from ardintra.features.risk_sim import ABS_CUTS, CLASSES, PRESETS, RAMP, risk_for_scenario
from ardintra.api.day_predictor import predict_day, fuegos_del_dia

API_DIR = Path(__file__).resolve().parent
STATIC = API_DIR / "static"
CELLS = Path("data/processed/h3_cells/res9_static.parquet")
LABELS = Path("data/processed/labels/v1")
_BORDER_DIR = Path("data/interim/h3_borders")

app = FastAPI(title="WildfireRisk-ES mini-viewer")


@lru_cache(maxsize=1)
def cells() -> pd.DataFrame:
    keep = ["h9", "lat", "lon", "forest_frac", "fuel_model", "elev_m", "slope_deg",
            "aspect_deg", "dist_to_road", "road_class", "road_density_1km",
            "road_fuel_interface"]
    return pd.read_parquet(CELLS, columns=keep)


@lru_cache(maxsize=8)
def risk_scenario(scenario: str) -> pd.DataFrame:
    if scenario not in PRESETS:
        raise HTTPException(404, f"escenario {scenario} no existe")
    return risk_for_scenario(cells(), scenario)


@lru_cache(maxsize=2)
def labels_ign() -> pd.DataFrame:
    df = pd.read_parquet(LABELS / "ignitions.parquet")
    df["n"] = df["fire_id"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)
    agg = (df.groupby("h9").agg(n_igniciones=("n", "sum"),
                                causa=("cause", lambda s: int(s.mode().iat[0]) if len(s.mode()) else -1),
                                primera=("fecha", "min"),
                                superficie=("surface_ha", "sum"),
                                provincia=("provincia", "first"))
           .reset_index())
    agg = agg.merge(cells()[["h9", "lat", "lon"]], on="h9", how="inner")
    return agg


@lru_cache(maxsize=2)
def labels_effis() -> pd.DataFrame:
    df = pd.read_parquet(LABELS / "burned_cells.parquet")
    agg = df.groupby("h9").agg(surface_ha=("surface_ha", "sum"), fecha=("fecha", "min")).reset_index()
    agg = agg.merge(cells()[["h9", "lat", "lon"]], on="h9", how="inner")
    return agg


_BBOX = (-9.65, 41.75, -6.60, 43.95)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/v1/risk/cell/{h9}")
def risk_cell(h9: str, scenario: str = "ola_de_calor") -> JSONResponse:
    idx = cells().set_index("h9")
    if h9 not in idx.index:
        raise HTTPException(404, "celda sin suelo forestal en el catálogo")
    row = idx.loc[h9].to_dict()
    r = risk_scenario(scenario).set_index("h9")
    sim = r.loc[h9, ["p_riesgo", "clase"]].to_dict() if h9 in r.index else {}
    ign = labels_ign().set_index("h9")
    if h9 in ign.index:
        irow = ign.loc[h9]
        if isinstance(irow, pd.DataFrame):
            irow = irow.iloc[0]
        ign_info = {
            "n_igniciones": int(irow["n_igniciones"]),
            "causa": int(irow["causa"]),
            "primera": str(pd.Timestamp(irow["primera"]).date()),
            "superficie_ha": round(float(irow["superficie"]), 2),
        }
    else:
        ign_info = {"n_igniciones": 0}
    out = {
        "h3": h9,
        "lat": row["lat"], "lon": row["lon"],
        "boundary": [[lng, lat] for lat, lng in h3.cell_to_boundary(h9)],  # resalte hexágono
        "static": {
            "forest_frac": row["forest_frac"],
            "fuel_model": int(row["fuel_model"]) if pd.notna(row["fuel_model"]) else None,
            "elev_m": round(float(row["elev_m"]), 0),
            "slope_deg": round(float(row["slope_deg"]), 1),
            "aspect_deg": round(float(row["aspect_deg"]), 0),
            "dist_to_road": round(float(row["dist_to_road"]), 0),
            "road_class": row["road_class"],
            "road_density_1km": round(float(row["road_density_1km"]), 2),
            "road_fuel_interface": int(row["road_fuel_interface"]),
        },
        "sim": {
            "p_riesgo": float(sim.get("p_riesgo", -1)),
            "clase": int(sim.get("clase", -1)),
            "clase_txt": CLASSES[int(sim["clase"])] if sim.get("clase", -1) >= 0 else "n/d",
            "escenario": scenario,
        },
        "egif": ign_info,
    }
    return JSONResponse(out)


def _png_grid(width: int) -> tuple[int, int]:
    """(W, H) del overlay."""
    lon0, lat0, lon1, lat1 = _BBOX
    return width, int(width / ((lon1 - lon0) / (lat1 - lat0)))


def _merc_lat_rows(H: int) -> np.ndarray:
    """Latitud real de cada fila del raster en la geometría de Web Mercator:
    el ImageSource interpola linealmente en y-mercator (no en latitud)."""
    lat0, lat1 = _BBOX[1], _BBOX[3]

    def merc(lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    ys = np.linspace(merc(lat1), merc(lat0), H)
    return np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)


_proj_m = Transformer.from_crs("EPSG:4326", "EPSG:25829", always_xy=True)


@lru_cache(maxsize=1)
def _cell_xy_m() -> tuple[np.ndarray, np.ndarray]:
    """Centros de celda en metros (ETRS89/25829) para el árbol de Voronoi."""
    d = cells()
    x, y = _proj_m.transform(d["lon"].to_numpy(), d["lat"].to_numpy())
    return x, y


def _band_colors(df: pd.DataFrame, palette: str) -> np.ndarray:
    """Color RGB (0-255) por fila: discreta (RAMP) o continua (interpolada
    DENTRO de cada banda absoluta del boletín)."""
    if palette == "continua":
        ramp = np.asarray(RAMP, dtype=float)
        p = df["p_riesgo"].to_numpy()
        k = df["clase"].astype(int).to_numpy()
        lo = np.array([0.0, *ABS_CUTS])
        hi = np.array([*ABS_CUTS, 1.0])
        t = np.clip((p - lo[k]) / np.maximum(hi[k] - lo[k], 1e-9), 0, 1)
        kb = np.clip(k + 1, 0, 4)
        rgb_f = ramp[k] * (1 - t[:, None]) + ramp[kb] * t[:, None]
        return rgb_f.astype(np.uint8)
    ramp = np.asarray(RAMP, dtype=np.uint8)
    return ramp[df["clase"].astype(int).to_numpy()]


def _voronoi_img(W: int, H: int, colors: np.ndarray, alpha: int = 215) -> Image.Image:
    """Cada píxel recibe el color de la celda H3 que lo contiene de verdad:
    el Voronoi de los centros de celda ES la teselación hexagonal."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    lon0, lon1 = _BBOX[0], _BBOX[2]
    lat_rows = _merc_lat_rows(H)
    xs = np.linspace(lon0, lon1, W)
    lon_g, lat_g = np.meshgrid(xs, lat_rows)
    px_m, py_m = _proj_m.transform(lon_g.ravel(), lat_g.ravel())

    cx, cy = _cell_xy_m()
    tree = cKDTree(np.column_stack([cx, cy]))
    dist, idx = tree.query(np.column_stack([px_m, py_m]), k=1, workers=-1)

    img = np.zeros((H * W, 4), dtype=np.uint8)
    ok = dist <= 250.0  # margen > apotema res9: sin dominio = mar/borde
    img[ok, :3] = colors[idx[ok]]
    img[ok, 3] = alpha
    return Image.fromarray(img.reshape(H, W, 4), "RGBA")


@lru_cache(maxsize=10)
def _risk_png(scenario: str, palette: str, width: int) -> bytes:
    df = risk_scenario(scenario).reset_index(drop=True)
    W, H = _png_grid(width)
    img = _voronoi_img(W, H, _band_colors(df, palette))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


@app.get("/overlay/risk/{scenario}.png")
def overlay_risk(scenario: str, width: int = 1600, palette: str = "discreta") -> Response:
    return Response(
        content=_risk_png(scenario, palette, width),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/overlay/borders.png")
def overlay_borders(width: int = 3200) -> Response:
    """Contornos de las 245k celdas H3, alineados al mismo warp mercator."""
    _BORDER_DIR.mkdir(parents=True, exist_ok=True)
    cache = _BORDER_DIR / f"borders_merc_{width}.png"
    if cache.exists():
        return Response(content=cache.read_bytes(), media_type="image/png",
                        headers={"Cache-Control": "max-age=604800"})
    W, H = _png_grid(width)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lon0, lat0, lon1, lat1 = _BBOX
    lat_rows = _merc_lat_rows(H)
    row_of = np.interp  # coordenadas específicas por celda
    for h9 in cells()["h9"]:
        pts = []
        for lat, lng in h3.cell_to_boundary(h9):
            x = (lng - lon0) / (lon1 - lon0) * (W - 1)
            # lat -> fila vía tabla de lat_rows (ya en mercator)
            y = float(np.abs(lat_rows - lat).argmin())
            pts.append((x, y))
        draw.line(pts + [pts[0]], fill=(20, 20, 20, 130), width=1)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    cache.write_bytes(bio.getvalue())
    return Response(content=bio.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "max-age=604800"})


def _geojson(df: pd.DataFrame, props: list[str]) -> dict:
    feats = []
    for _, r in df.iterrows():
        d = {}
        for k in props:
            v = r[k]
            if isinstance(v, pd.Timestamp):
                v = v.isoformat()
            elif hasattr(v, "item"):
                v = v.item()
            d[k] = None if pd.isna(v) else v
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": d,
        })
    return {"type": "FeatureCollection", "features": feats}


# ===== Predicción real del pipeline completo: selección de día ============= #

@lru_cache(maxsize=4)
def _forecast_day(date: str) -> tuple[pd.DataFrame, dict, list[dict]]:
    o = predict_day(date)
    arrays = o.pop("_arrays")
    df = pd.DataFrame({
        "h9": arrays["h9"], "p_cal": arrays["p_cal"], "level": arrays["levels"],
    })
    fires = fuegos_del_dia(date)
    return df, o, fires


@app.get("/v1/forecast/{date}")
def forecast_day(date: str) -> JSONResponse:
    """Pipeline completo sobre un día: features full-grid + LGBM + isotónica +
    Stage 2 regional. Devuelve λ diario, top regiones (×N), top celdas y los
    fuegos reales de ese día (EGIF)."""
    try:
        pd.Timestamp(date)
    except ValueError:
        raise HTTPException(400, "fecha inválida (formato AAAA-MM-DD)")
    df, summary, fires = _forecast_day(date)
    return JSONResponse({
        **summary,
        "fuegos_reales": fires,
        "n_fuegos_reales": len(fires),
        "explicacion": {
            "celda": "P(ignición ≥10 ha hoy | celda) — LGBM calibrado con isotónica",
            "region": "P(≥1 fuego ≥10 ha hoy en la región res7) — Stage 2 GLM+MEM",
            "riesgo_x": "riesgo relativo vs la media histórica de la región-día",
            "lambda_galicia": "Nº esperado de fuegos ≥10 ha en Galicia ese día (Σμ Stage 2)",
        },
    })


@app.get("/overlay/forecast/{date}.png")
def overlay_forecast(date: str, width: int = 1600, palette: str = "discreta") -> Response:
    """Mapa de niveles de riesgo por celda para el día elegido."""
    try:
        pd.Timestamp(date)
    except ValueError:
        raise HTTPException(400, "fecha inválida")
    df, _, _ = _forecast_day(date)
    W, H = _png_grid(width)
    ramp = np.asarray(RAMP, dtype=np.uint8)
    img = _voronoi_img(W, H, ramp[df["level"].to_numpy()])
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return Response(content=bio.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})



@app.get("/overlay/ignitions.geojson")
def overlay_ignitions() -> JSONResponse:
    df = labels_ign()
    return JSONResponse(
        _geojson(df, ["n_igniciones", "causa", "primera", "superficie", "provincia"]),
    )


@app.get("/overlay/effis.geojson")
def overlay_effis() -> JSONResponse:
    df = labels_effis()
    return JSONResponse(_geojson(df, ["surface_ha", "fecha"]))


@app.get("/overlay/roads_light.geojson")
def overlay_roads() -> JSONResponse:
    r = gpd.read_parquet("data/interim/osm/roads_driveable.parquet", columns=["road_class", "geometry"])
    r = r[r["road_class"].isin(["nacional", "autovia", "comarcal"])]
    # longitud en METROS (25829): filtrar aquí, no en grados (bug anterior: 0 features)
    r = r.to_crs("EPSG:25829")
    r = r.set_geometry(r.geometry.simplify(30, preserve_topology=False))
    r = r[r.geometry.length > 300]
    r = r.to_crs("EPSG:4326")
    feats = [
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": [
            [round(x, 5), round(y, 5)] for x, y in g.coords
         ]},
         "properties": {"c": rc}}
        for g, rc in zip(r.geometry, r["road_class"], strict=True)
    ]
    return JSONResponse({"type": "FeatureCollection", "features": feats})
