"""Backfill meteorológico histórico vía Open-Meteo Archive API.

Ventaja sobre AEMET OpenData para el HISTÓRICO: sin API key, sin rate-limit
operativo, y una sola petición por estación cubre 2017→hoy (AEMET exigiría
~14.000 peticiones a 31 días/tramo y se bloquea por rate limiting).

AEMET queda como fuente del TIEMPO REAL (obs horaria, crontab).

Salida: data/raw/openmeteo/daily_2017_today.parquet
Variables daily: tmax, tmin, precip, wind_max, rh_mean (FWI/DC/DMC derivable).
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import httpx
import pandas as pd

from ardintra.ingest.aemet.aemet_obs import stations_galicia
from ardintra.ingest.common.manifest import build_manifest

OUT = Path("data/raw/openmeteo/daily_2017_today.parquet")
STATION_DIR = Path("data/raw/openmeteo/per_station")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
]


def fetch_station(lat: float, lon: float, start: str, end: str) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Madrid",
    }
    for attempt in range(6):
        with httpx.Client(timeout=90) as c:
            r = c.get(ARCHIVE_URL, params=params)
        if r.status_code == 429:
            print("  · 429: esperando 63 s ...")
            time.sleep(63)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Open-Meteo sigue throttled tras {attempt + 1} intentos")


def fetch_stations_bulk(stations: pd.DataFrame, start: str, end: str,
                        chunk_stations: int = 10) -> list[pd.DataFrame]:
    """Múltiples estaciones en UNA llamada (listas de lat/lon).

    Open-Meteo acepta latitudes/longitudes separadas por comas y devuelve una
    lista de resultados en el mismo orden. Con ~19 estaciones restantes
    son 2 llamadas en vez de 19 —.Update critical para no agotar la quota diaria.
    """
    lat_list = ",".join(f"{v:.4f}" for v in stations["lat"])
    lon_list = ",".join(f"{v:.4f}" for v in stations["lon"])
    out: list[pd.DataFrame] = []
    for attempt in range(8):
        params = {
            "latitude": lat_list,
            "longitude": lon_list,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY_VARS),
            "timezone": "Europe/Madrid",
        }
        with httpx.Client(timeout=300) as c:
            r = c.get(ARCHIVE_URL, params=params)
        if r.status_code == 429:
            print("· 429 bulk:", (r.text[:120] or "").replace("\n", " "))
            time.sleep(3600)
            continue
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            payload = [payload]
        for st, block in zip(stations.itertuples(index=False), payload, strict=True):
            daily = block.get("daily", {})
            if not daily:
                continue
            df = pd.DataFrame.from_dict(daily)
            df["idema"] = st["idema"]
            df["nombre"] = st["nombre"]
            df["provincia"] = st["provincia"]
            df["lat"] = st["lat"]
            df["lon"] = st["lon"]
            out.append(df)
        return out
    raise RuntimeError("Open-Meteo sigue throttled tras 8 horas de reintentos")


def run(start: str = "2017-01-01", end: str | None = None) -> Path:
    end = end or dt.date.today().isoformat()
    stations = stations_galicia(Path("data/raw/aemet")).dropna(subset=["lat", "lon"])
    print(f"estaciones: {len(stations)}  período: {start} -> {end}")

    STATION_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    missing = [
        st for _, st in stations.iterrows()
        if not (STATION_DIR / f"estacion={st['idema']}.parquet").exists()
    ]
    print(f"ya en checkpoint: {len(stations) - len(missing)} · por descargar: {len(missing)}")

    # 1 llamada colectiva por bloque de ~10 estaciones (quota-friendly)
    i = 0
    while i < len(missing):
        batch = pd.DataFrame(missing[i : i + 10])
        try:
            got = fetch_stations_bulk(batch, start, end)
        except (RuntimeError, httpx.HTTPStatusError) as e:
            print(f"bloque {i // 10}: {e} -> reintentando en 3600 s")
            time.sleep(3600)
            continue
        for df in got:
            cache = STATION_DIR / f"estacion={df['idema'].iloc[0]}.parquet"
            df.to_parquet(cache, index=False)
            parts.append(df)
            print(f"  {df['idema'].iloc[0]}: {len(df)} días")
        i += 10
        if i < len(missing):
            time.sleep(5)

    parts.extend(pd.read_parquet(STATION_DIR / f"estacion={s['idema']}.parquet") for _, s in stations.iterrows()
                 if (STATION_DIR / f"estacion={s['idema']}.parquet").exists())

    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if len(full):
        full = full.rename(
            columns={
                "time": "fecha",
                "temperature_2m_max": "tmax_c",
                "temperature_2m_min": "tmin_c",
                "precipitation_sum": "prec_mm",
                "windspeed_10m_max": "wind_max_kmh",
                "relative_humidity_2m_mean": "rh_mean",
            }
        )
        # precip en mm/día del país (mm el sufijoSpain) ya en mm; limpiar fechas
        full["fecha"] = pd.to_datetime(full["fecha"])
        full["prec_mm"] = pd.to_numeric(full["prec_mm"], errors="coerce")
        full["rh_mean"] = pd.to_numeric(full["rh_mean"], errors="coerce")
        full["tmax_c"] = pd.to_numeric(full["tmax_c"], errors="coerce")
        full["tmin_c"] = pd.to_numeric(full["tmin_c"], errors="coerce")
        full["wind_max_kmh"] = pd.to_numeric(full["wind_max_kmh"], errors="coerce")
        full.to_parquet(OUT, index=False)
        build_manifest(
            "openmeteo_archive",
            OUT.parent,
            coverage={
                "start": start,
                "end": end,
                "n_estaciones": len(parts),
                "n_rows": len(full),
                "variables": DAILY_VARS,
            },
            license="Open-Meteo CC-BY-4.0 (https://openmeteo.org), datos basados en ERA5/ERA5-Land/CERRA",
            description="Serie diaria por estación para FWI baseline y features met",
        )
        print(f"-> {OUT} ({len(full):,} filas)")
    else:
        print("! sin datos")
    return OUT


if __name__ == "__main__":
    run()
