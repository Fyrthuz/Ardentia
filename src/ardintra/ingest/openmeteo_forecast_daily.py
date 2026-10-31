"""Forecast histórico (lead 24 h) vía Open-Meteo Previous Runs API — modelo GFS.

Variables hourly `_previous_day1` = valor del día D emitido en la pasada del
día D-1 (régimen operativo: mañana conoces el forecast de hoy). Se agregan a
diario (tmax, rh media, viento máx, precip suma) y la cadena FWI se recalcula
sobre la serie con fwi_series().

Modelo GFS: única cobertura que llega a 2023 (ECMWF archivado solo desde
2024-01; GFS desde 2021-03 según docs de Open-Meteo).

Uso previsto (1.3): train con meteo observada (≤2022), test/operativa con
forecast (2023-24) — cuantifica la degradación real del sistema desplegado.

Salida: data/raw/openmeteo/forecast_daily.parquet + per_station_fc/
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from ardintra.ingest.aemet.aemet_obs import stations_galicia
from ardintra.ingest.common.manifest import build_manifest

OUT = Path("data/raw/openmeteo/forecast_daily.parquet")
STATION_DIR = Path("data/raw/openmeteo/per_station_fc")
FC_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
]


def fetch_bulk(stations: pd.DataFrame, start: str, end: str) -> list[pd.DataFrame]:
    lat_list = ",".join(f"{v:.4f}" for v in stations["lat"])
    lon_list = ",".join(f"{v:.4f}" for v in stations["lon"])
    params = {
        "latitude": lat_list,
        "longitude": lon_list,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "kmh",
        "timezone": "Europe/Madrid",
        "models": "gfs_seamless",
    }
    for attempt in range(6):
        with httpx.Client(timeout=600) as c:
            r = c.get(FC_URL, params=params)
        if r.status_code == 429:
            print(f"  · 429: esperando 65 s (intento {attempt + 1})")
            time.sleep(65)
            continue
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            payload = [payload]
        out = []
        for st, block in zip(stations.itertuples(index=False), payload, strict=True):
            hourly = block.get("hourly", {})
            if not hourly or "time" not in hourly:
                continue
            h = pd.DataFrame.from_dict(hourly)
            h["time"] = pd.to_datetime(h["time"])
            d = h.set_index("time").resample("D").agg({
                "temperature_2m_previous_day1": "max",
                "relative_humidity_2m_previous_day1": "mean",
                "precipitation_previous_day1": "sum",
                "wind_speed_10m_previous_day1": "max",
            }).reset_index().rename(columns={"time": "fecha"})
            d["idema"] = getattr(st, "idema")
            d["provincia"] = getattr(st, "provincia")
            d["lat"] = getattr(st, "lat")
            d["lon"] = getattr(st, "lon")
            out.append(d)
        return out
    raise RuntimeError("throttled")


def run(start: str = "2023-01-01", end: str | None = None) -> Path:
    end = end or dt.date.today().isoformat()
    stations = stations_galicia(Path("data/raw/aemet")).dropna(subset=["lat", "lon"])
    print(f"estaciones: {len(stations)}  período: {start} -> {end}")

    STATION_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        st for _, st in stations.iterrows()
        if not (STATION_DIR / f"estacion={st['idema']}.parquet").exists()
    ]
    print(f"en checkpoint: {len(stations) - len(missing)} · por descargar: {len(missing)}")

    i = 0
    while i < len(missing):
        batch = pd.DataFrame(missing[i:i + 4])
        try:
            got = fetch_bulk(batch, start, end)
        except (RuntimeError, httpx.HTTPStatusError) as e:
            print(f"bloque {i // 4}: {e} -> pausa 65 s")
            time.sleep(65)
            continue
        for df in got:
            cache = STATION_DIR / f"estacion={df['idema'].iloc[0]}.parquet"
            df.to_parquet(cache, index=False)
            print(f"  {df['idema'].iloc[0]}: {len(df)} días "
                  f"({df['fecha'].iloc[0].date()} → {df['fecha'].iloc[-1].date()})")
        i += 4
        if i < len(missing):
            time.sleep(3)

    parts = [pd.read_parquet(p) for p in sorted(STATION_DIR.glob("estacion=*.parquet"))]
    if not parts:
        print("! sin datos")
        return OUT
    full = pd.concat(parts, ignore_index=True)
    full = full.rename(columns={
        "temperature_2m_previous_day1": "tmax_c",
        "relative_humidity_2m_previous_day1": "rh_mean",
        "precipitation_previous_day1": "prec_mm",
        "wind_speed_10m_previous_day1": "wind_max_kmh",
    })
    full["fecha"] = pd.to_datetime(full["fecha"])
    for c in ("tmax_c", "prec_mm", "wind_max_kmh", "rh_mean"):
        full[c] = pd.to_numeric(full[c], errors="coerce")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT, index=False)
    build_manifest(
        "openmeteo_forecast",
        OUT.parent,
        coverage={"start": start, "end": end, "n_estaciones": len(parts),
                  "n_rows": len(full), "lead": "24h (previous_day1, GFS)",
                  "variables": HOURLY_VARS},
        license="Open-Meteo CC-BY-4.0",
        description="Forecast diario emitido D-1 por estación (GFS, régimen operativo)",
    )
    print(f"-> {OUT} ({len(full):,} filas)")
    return OUT


if __name__ == "__main__":
    run()
