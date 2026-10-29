"""Panel WildfireRisk-ES (Streamlit): controles + métricas, con el mapa JS embebido.

El renderizado cartográfico vive la API FastAPI (puerto 8001), que evita los límites
de payload del pydeck/Streamlit y carga los detalles por click (fetch bajo demanda).

Ejecutar:  uv run streamlit run apps/viewer.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ardintra.features.risk_sim import CLASSES, PRESETS, simulate_risk

st.set_page_config(page_title="WildfireRisk-ES · Galicia", layout="wide", page_icon="🔥")

API_BASE = "http://localhost:8001"
PRESET_NAMES = {
    "Primavera normal": "primavera_normal",
    "Viento moderado": "viento_moderado",
    "Ola de calor": "ola_de_calor",
    "Sequía extrema": "sequia_extrema",
}


@st.cache_data
def load_cells() -> pd.DataFrame:
    return pd.read_parquet(
        "data/processed/h3_cells/res9_static.parquet",
        columns=["h9", "forest_frac", "fuel_model", "slope_deg", "aspect_deg",
                 "dist_to_road", "road_class", "road_density_1km",
                 "road_fuel_interface", "elev_m"],
    )


@st.cache_data
def load_ignition_counts() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/labels/v1/ignitions.parquet")
    return (df.groupby("h9")
              .agg(n_igniciones=("fire_id", "size"),
                   superficie_ha=("surface_ha", "sum"))
              .reset_index())


st.title("🔥 WildfireRisk-ES · Piloto Galicia")
st.caption("Estáticas + etiquetas REALES · fondo = escenario simulado (Fase 2 lo sustituye por el modelo entrenado). "
           "Haz click en cualquier celda del mapa para cargar su ficha.")

with st.sidebar:
    st.header("Escenario meteorológico")
    preset_label = st.radio("Preset", list(PRESET_NAMES), index=2)
    p = PRESETS[PRESET_NAMES[preset_label]]
    rh = st.slider("Humedad relativa (%)", 5, 100, p["rh"])
    temp = st.slider("T. máxima (°C)", 5, 45, p["temp_max"])
    viento = st.slider("Viento sostenido (km/h)", 0, 90, p["viento"])
    p30d = st.slider("P30d (mm)", 0, 120, p["p30d"])
    ndmi = st.slider("NDMI foliar (proxy)", -1.0, 0.5, p["ndmi"], 0.05)

    st.divider()
    st.markdown("**Escena seleccionada del dayo**: todas celdas registran regresión de nivel 0 a nivel 4 "
                "por cuantiles de la distribución del propio día. La calibración absoluta con PR-AUC es Fase 2.")

cells = load_cells()

# riesgo simulado con los sliders (para métricas/top) — por preset para el fondo del mapa
risk = simulate_risk(cells, rh, temp, viento, p30d, ndmi)
differs = (rh, temp, viento, p30d, ndmi) != (
    p["rh"], p["temp_max"], p["viento"], p["p30d"], p["ndmi"]
)
escenario_slug = PRESET_NAMES[preset_label]  # el overlay raster siempre usa el preset

top = risk.nlargest(12, "p_riesgo")[
    ["p_riesgo", "fuel_model", "road_class", "road_density_1km", "dist_to_road", "slope_deg"]
].rename(columns={
    "p_riesgo": "p", "fuel_model": "fuel", "road_class": "vía",
    "road_density_1km": "dens km/km²", "dist_to_road": "dist vía m", "slope_deg": "pend °",
})

c1, c2, c3 = st.columns(3)
c1.metric("Nivel Alto (2)", int((risk["clase"] == 2).sum()))
c2.metric("Nivel Muy alto (3)", int((risk["clase"] == 3).sum()))
c3.metric("Nivel Extremo (4)", int((risk["clase"] == 4).sum()))

url = f"{API_BASE}/?scenario={escenario_slug}&t={int(time.time())}"  # cache-buster del iframe
st.caption(f"Mapa interactivo listo en {API_BASE} (escenario activo: **{preset_label}**)")
components.iframe(url, height=640, scrolling=False)

st.subheader("Top 12 celdas con mayor riesgo simulado")
st.dataframe(top, hide_index=True, use_container_width=True)

ign = load_ignition_counts()
st.caption(f"Fuentes reales cargadas: {len(cells):,} celdas forestales · "
           f"{len(ign):,} celdas con ignición EGIF (2017–2023)")
