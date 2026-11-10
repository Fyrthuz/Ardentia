"""Explicabilidad TreeSHAP: por qué esta celda tiene este score hoy.

- explain_cell(date, h9): SHAP (log-odds) de las 48 features para esa celda
  ese día → top factores que SUBEN y que BAJAN el riesgo, con el valor original
  de cada feature.
- explain_day_global(date): importancia media |SHAP| sobre una muestra del
  grid del día (para el boletín).

Uso CLI: uv run python -m ardintra.api.explanation 2024-07-14
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# traducción técnica → humana de las features
FEATURE_HUM = {
    "fwi": "FWI (índice meteorológico)", "ffmc": "FFMC (combustible fino)",
    "dmc": "DMC (combustible medio)", "dc": "DC (sequía profunda)",
    "isi": "ISI (velocidad de ignición)", "bui": "BUI (combustible acumulado)",
    "tmax_c": "Temp. máxima", "rh_mean": "Humedad relativa",
    "wind_max_kmh": "Viento máx", "prec_mm": "Precipitación",
    "swc_l1_mean": "Humedad de suelo (superficie)", "swc_l2_mean": "Humedad de suelo (raíces)",
    "vpd_max": "Déficit de presión de vapor", "vpd_mean": "VPD medio",
    "spei30": "Sequía 30 días (SPEI)", "spei90": "Sequía 90 días (SPEI)",
    "ffmc_anom": "Anomalía FFMC vs 7d", "dmc_anom": "Anomalía DMC vs 7d",
    "dc_anom": "Anomalía DC vs 7d", "fwi_7d": "FWI medio 7 días",
    "fwi_prev": "FWI día anterior", "ffmc_7d": "FFMC medio 7 días",
    "ndmi_summer": "NDMI verano previo (humedad vegetación)",
    "ndvi_summer": "NDVI verano previo", "nbr_summer": "NBR verano previo",
    "ndmi_anom": "Anomalía NDMI", "forest_frac": "Fracción forestal",
    "fuel_model": "Modelo de combustible", "slope_deg": "Pendiente",
    "aspect_sin": "Orientación (solana/umbría)", "aspect_cos": "Orientación (E-O)",
    "elev_m": "Cota", "dist_to_road_log": "Distancia a vía",
    "road_density_log": "Densidad de vías", "road_fuel_interface": "Interfaz vía-combustible",
    "road_class": "Clase de vía",
    "dias_ultimo_fuego_cercano": "Días desde fuego cercano",
    "dist_ultimo_fuego_m": "Distancia al último fuego", "ardio_1km_1an": "Ardió ≤1 km en 1 año",
    "n_ign_cell_3d": "Igniciones en la celda (3d)", "n_ign_near_2d": "Igniciones ≤2.5 km (2d)",
    "n_ign_near_7d": "Igniciones ≤5 km (7d)", "es_finde": "Fin de semana",
    "dens_x_weekend": "Densidad viaria × finde", "lat_f": "Latitud", "lon_f": "Longitud",
    "doy_sin": "Estación (sen)", "doy_cos": "Estación (cos)",
}


@lru_cache(maxsize=1)
def _explainer():
    import shap
    from ardintra.api.day_predictor import _day_Z, _lgbm_pack
    _, meta = _day_Z("2024-07-14")  # fuerza carga
    pack = _lgbm_pack()
    return shap.TreeExplainer(pack["model"], feature_perturbation="tree_path_dependent")


def explain_cell(date_str: str, h9: str, top: int = 4) -> dict:
    """SHAP para una celda del día. Devuelve factores que suben/bajan."""
    from ardintra.api.day_predictor import _day_Z
    Z, meta = _day_Z(date_str)
    d_orig = meta["d"]
    if h9 not in set(Z.index if isinstance(Z, pd.DataFrame) else []) and not (set(d_orig["h9"]) & {h9}):
        h9_in = h9 in set(d_orig["h9"])
        if not h9_in:
            return {"error": f"celda {h9} no catalogada"}
    pos = int(d_orig.index[d_orig["h9"] == h9][0])
    row = Z.iloc[[pos]]
    exp_vals = _explainer().shap_values(row, check_additivity=False)
    sv = np.asarray(exp_vals)
    if sv.ndim == 3:      # (1, feats, classes) en新模式 shap
        sv = sv[0, :, 1]
    else:
        sv = sv[0]        # (feats,) log-odds binaria
    ev = _explainer().expected_value
    base = float(ev if np.ndim(ev) == 0 else ev[-1])
    d = d_orig.iloc[pos]
    feats = list(Z.columns)
    order_up = np.argsort(-sv)
    order_dn = np.argsort(sv)

    def fmt(i: int) -> dict:
        f = feats[i]
        v = d.get(f, np.nan)
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = round(float(v), 3)
        return {
            "feature": f,
            "nombre": FEATURE_HUM.get(f, f),
            "valor": v,
            "contribucion": round(float(sv[i]), 4),   # log-odds
        }

    return {
        "h9": h9, "fecha": date_str,
        "base": round(base, 4),
        "sube": [fmt(i) for i in order_up[:top] if sv[i] > 0],
        "baja": [fmt(i) for i in order_dn[:top] if sv[i] < 0],
        "p_modelo": round(float(np.exp(base) if False else np.nan), 6) if False else None,
    }


def explain_day_global(date_str: str, n_sample: int = 1500) -> list[dict]:
    """Importancia global (media |SHAP|) para el día del grid."""
    from ardintra.api.day_predictor import _day_Z
    Z, _ = _day_Z(date_str)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Z), size=min(n_sample, len(Z)), replace=False)
    sample = Z.iloc[idx]
    sv = np.asarray(_explainer().shap_values(sample.iloc[:512], check_additivity=False))
    if sv.ndim == 3:
        sv = sv[0, :, 1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)[:6]
    return [{"feature": Z.columns[i], "nombre": FEATURE_HUM.get(Z.columns[i], Z.columns[i]),
             "importancia": round(float(imp[i]), 5)} for i in order]


if __name__ == "__main__":
    import json
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2023-10-12"
    from ardintra.api.day_predictor import predict_day
    o = predict_day(date)
    top0 = o["celdas_top"][0]["h9"]
    print(json.dumps(explain_cell(date, top0), indent=2, ensure_ascii=False))
    print(json.dumps(explain_day_global(date), indent=2, ensure_ascii=False))
