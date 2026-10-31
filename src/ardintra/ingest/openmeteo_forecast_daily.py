"""Forecast histórico (lead ~24 h) vía Open-Meteo Historical Forecast API.

A diferencia del Previous Runs API (GFS, solo desde 2024-01), el Historical
Forecast API archiva las pasadas del HRES/seamless desde 2022-01 y expone
variables daily SIN sufijo: el valor del día D proviene de la pasada emitida
el día anterior (régimen operativo: por la mañana conoces el forecast de hoy).

Salida: data/raw/openmeteo/forecast_daily.parquet + per_station_fc/
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import httpx
import pandas as pd

from ardintra.ingest.aemet.aemet_obs import stations_galicia
from ardintra.ingest.common.manifest import build_manifest

OUT = Path("data/raw/openmeteo/forecast_daily.parquet")
STATION_DIR = Path("data/raw/openmeteo/per_station_fc")
FC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
]

RENAME = {
    "time": "fecha",
    "temperature_2m_max": "tmax_c",
    "precipitation_sum": "prec_mm",
    "windspeed_10m_max": "wind_max_kmh",
    "relative_humidity_2m_mean": "rh_mean",
}


def fetch_bulk(stations: pd.DataFrame, start: str, end: str) -> list[pd.DataFrame]:
    lat_list = ",".join(f"{v:.4f}" for v in stations["lat"])
    lon_list = ",".join(f"{v:.4f}" for v in stations["lon"])
    params = {
        "latitude": lat_list,
        "longitude": lon_list,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Madrid",
    }
    for attempt in range(6):
        with httpx.Client(timeout=600) as c:
            r = c.get(FC_URL, params=params)
        if r.status_code == 429:
            print(f"  · 429: esperando 65 s (intento {attempt + 1})", flush=True)
            time.sleep(65)
            continue
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            payload = [payload]
        out = []
        for st, block in zip(stations.itertuples(index=False), payload, strict=True):
            daily = block.get("daily", {})
            if not daily or "time" not in daily:
                continue
            df = pd.DataFrame.from_dict(daily)
            df["idema"] = getattr(st, "idema")
            df["provincia"] = getattr(st, "provincia")
            df["lat"] = getattr(st, "lat")
            df["lon"] = getattr(st, "lon")
            out.append(df)
        return out
    raise RuntimeError("throttled")


def run(start: str = "2023-01-01", end: str | None = None) -> Path:
    end = end or dt.date.today().isoformat()
    stations = stations_galicia(Path("data/raw/aemet")).dropna(subset=["lat", "lon"])
    print(f"estaciones: {len(stations)}  período: {start} -> {end}", flush=True)

    STATION_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        st for _, st in stations.iterrows()
        if not (STATION_DIR / f"estacion={st['idema']}.parquet").exists()
    ]
    print(f"en checkpoint: {len(stations) - len(missing)} · por descargar: {len(missing)}", flush=True)

    i = 0
    while i < len(missing):
        batch = pd.DataFrame(missing[i:i + 8])
        try:
            got = fetch_bulk(batch, start, end)
        except (RuntimeError, httpx.HTTPStatusError) as e:
            print(f"bloque {i // 8}: {e} -> pausa 65 s", flush=True)
            time.sleep(65)
            continue
        for df in got:
            cache = STATION_DIR / f"estacion={df['idema'].iloc[0]}.parquet"
            df.to_parquet(cache, index=False)
            print(f"  {df['idema'].iloc[0]}: {len(df)} días "
                  f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})", flush=True)
        i += 8
        if i < len(missing):
            time.sleep(3)

    parts = [pd.read_parquet(p) for p in sorted(STATION_DIR.glob("estacion=*.parquet"))]
    if not parts:
        print("! sin datos")
        return OUT
    full = pd.concat(parts, ignore_index=True).rename(columns=RENAME)
    full["fecha"] = pd.to_datetime(full["fecha"])
    for c in ("tmax_c", "prec_mm", "wind_max_kmh", "rh_mean"):
        full[c] = pd.to_numeric(full[c], errors="coerce")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT, index=False)
    build_manifest(
        "openmeteo_forecast",
        OUT.parent,
        coverage={"start": start, "end": end, "n_estaciones": len(parts),
                  "n_rows": len(full), "lead": "~24h (pasada D-1, HRES seamless)",
                  "variables": DAILY_VARS},
        license="Open-Meteo CC-BY-4.0",
        description="Forecast diario emitido D-1 por estación (régimen operativo)",
    )
    print(f"-> {OUT} ({len(full):,} filas)")
    return OUT


if __name__ == "__main__":
    run()
