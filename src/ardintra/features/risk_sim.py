"""Motor de riesgo simulado (Fase 1.5, "casuística") compartido por visores.

Será reemplazado por el LightGBM en Fase 2. Acepta (df, scenario dict) y
devuelve df con p_riesgo + clase por cuantiles operativos del propio día.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAMP = [
    (54, 168, 82),    # 0 bajo
    (235, 210, 60),   # 1 moderado
    (245, 140, 40),   # 2 alto
    (232, 60, 40),    # 3 muy alto
    (150, 15, 45),    # 4 extremo
]
CLASSES = ["Bajo", "Moderado", "Alto", "Muy alto", "Extremo"]

# Cortes absolutos de nivel (recalibrados con PR-AUC en Fase 2).
# Compartidos por clase numérica, color discreto y gradiente continuo.
ABS_CUTS = (0.20, 0.35, 0.52, 0.72)

PRESETS = {
    "primavera_normal": dict(rh=62, temp_max=18, viento=12, p30d=30, ndmi=-0.15),
    "viento_moderado": dict(rh=48, temp_max=24, viento=30, p30d=15, ndmi=-0.25),
    "ola_de_calor": dict(rh=25, temp_max=38, viento=35, p30d=5, ndmi=-0.60),
    "sequia_extrema": dict(rh=18, temp_max=40, viento=25, p30d=0, ndmi=-0.85),
}


def simulate_risk(
    cells: pd.DataFrame, rh: float, temp_max: float, viento: float,
    p30d: float, ndmi_proxy: float,
) -> pd.DataFrame:
    d = cells.copy()
    met = (1.0 * (1 - rh / 100.0) + 0.02 * max(temp_max, 10.0) + 0.015 * viento
           - 0.012 * min(p30d, 60.0)) / 2.2
    met = np.clip(met, 0, 1)

    fine = d["fuel_model"].isin([1, 2, 3, 4, 5, 6, 7, 8, 11]).astype(float)
    fuel = (0.35 + 0.65 * fine) * d["forest_frac"].fillna(0)

    expo_road = np.exp(-d["dist_to_road"].clip(0, 600) / 120.0)
    dens = np.log1p(d["road_density_1km"].clip(0, 25)) / np.log1p(25)
    anthro = np.clip(0.55 * expo_road + 0.35 * dens + 0.30 * d["road_fuel_interface"], 0, 1)

    solana = ((d["aspect_deg"] > 120) & (d["aspect_deg"] < 260)).astype(float)
    topo = np.clip(0.6 * (d["slope_deg"] / 35).clip(0, 1) + 0.4 * solana, 0, 1)

    foliar = np.clip(1 - (ndmi_proxy + 1.0) / 2.0, 0, 1) * fuel
    z = (0.8 * met + 0.5 * fuel + 0.9 * anthro + 0.35 * topo + 0.55 * foliar - 2.15)
    p = 1 / (1 + np.exp(-3.2 * z))

    d["p_riesgo"] = p.round(3)
    c1, c2, c3, c4 = ABS_CUTS
    d["clase"] = np.select(
        [p < c1, p < c2, p < c3, p < c4],
        [0, 1, 2, 3],
        default=4,
    )
    return d


def risk_for_scenario(cells: pd.DataFrame, scenario: str) -> pd.DataFrame:
    p = PRESETS[scenario]
    return simulate_risk(cells, p["rh"], p["temp_max"], p["viento"], p["p30d"], p["ndmi"])
