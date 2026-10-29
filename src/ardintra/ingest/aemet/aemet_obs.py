"""AEMET OpenData: observación horaria + climatologías diarias (backfill FWI).

Patrón 2 pasos: cada endpoint devuelve JSON con la URL real de datos (`datos`).
Salidas:
  data/raw/aemet/obs_hourly/date=YYYY-MM-DD/obs.parquet      (cron horario)
  data/raw/aemet/climatologicos/estacion={idema}/daily.parquet
  data/raw/aemet/stations.parquet (inventario, filtro Galicia)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from ardintra.ingest.common.galicia import AEMET_DAILY_URL, AEMET_OBS_URL, AEMET_STATION_URL
from ardintra.ingest.common.manifest import build_manifest

_BASE = "https://opendata.aemet.es"


def _key() -> str:
    key = os.environ.get("AEMET_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Falta AEMET_API_KEY en .env")
    return key


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4))
def _aemet_get(client: httpx.Client, url: str) -> dict:
    r = client.get(url, params={"api_key": _key()}).raise_for_status().json()
    if r.get("estado") not in (200, "200"):
        raise RuntimeError(f"AEMET estado={r.get('estado')} en {url}")
    return r


def _decode_bytes(content: bytes) -> str:
    """AEMET sirve JSON anin en latin-1 (inventario) o utf-8 según endpoint."""
    for enc in ("utf-8", "latin-1", "iso-8859-15"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _aemet_data(client: httpx.Client, url: str) -> Any:
    two = _aemet_get(client, url)
    r = client.get(two["datos"])
    if not r.content.strip():
        # cuerpo vacío: sin datos para ese tramo (o fallo transitorio) → retry/skip
        raise ValueError(f"cuerpo vacío de datos AEMET: {two['datos']}")
    return json.loads(_decode_bytes(r.content))


import re

_GMS_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})([NSEW])$")


def gms_to_decimal(v: str) -> float | None:
    """'424607N' -> 42.7686; '070103W' -> -7.0175."""
    m = _GMS_RE.match((v or "").strip())
    if not m:
        return None
    deg, minutes, seconds, hemi = int(m[1]), int(m[2]), int(m[3]), m[4]
    val = deg + minutes / 60 + seconds / 3600
    if hemi in ("S", "W"):
        val = -val
    return round(val, 6)


def stations_galicia(out_base: Path) -> pd.DataFrame:
    out = out_base / "stations.parquet"
    if out.exists():
        return pd.read_parquet(out)
    with httpx.Client(timeout=60, base_url=_BASE) as client:
        raw = _aemet_data(client, AEMET_STATION_URL)
    df = pd.DataFrame(raw)
    galicia_prov = {
        "A CORUÑA", "LUGO", "OURENSE", "PONTEVEDRA", "LA CORUÑA", "CORUÑA",
    }
    df = df[df["provincia"].isin(galicia_prov)].copy()
    df["lon"] = df["longitud"].map(gms_to_decimal)
    df["lat"] = df["latitud"].map(gms_to_decimal)
    df = df.rename(columns={"indicativo": "idema"})[
        ["idema", "nombre", "provincia", "latitud", "lat", "longitud", "lon", "altitud"]
    ].drop_duplicates("idema")
    df.to_parquet(out, index=False)
    return df


def fetch_hourly(out_base: Path) -> pd.DataFrame:
    """Observación convencional: última hora disponible (cron horario)."""
    day_dir = out_base / "obs_hourly" / f"date={dt.date.today().isoformat()}"
    day_dir.mkdir(parents=True, exist_ok=True)
    out = day_dir / "obs.parquet"
    with httpx.Client(timeout=60, base_url=_BASE) as client:
        raw = _aemet_data(client, AEMET_OBS_URL)
    df = pd.DataFrame(raw)
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old, df]).drop_duplicates(subset=["idema", "fint"], keep="last")
    df.to_parquet(out, index=False)
    return df


def backfill_daily(out_base: Path, start: str = "2017-01-01", chunk_days: int = 30) -> pd.DataFrame:
    """Descarga climatologías diarias de TODAS las estaciones de Galicia (serie larga para FWI).

    Límite conocido de la API: máx 31 días por petición (estado=400 si se excede).
    """
    stations = stations_galicia(out_base)
    end_day = dt.date.today()
    start_day = dt.date.fromisoformat(start)
    all_parts: list[pd.DataFrame] = []
    for _, st in stations.iterrows():
        idema = st["idema"]
        est_dir = out_base / "climatologicos" / f"estacion={idema}"
        est_dir.mkdir(parents=True, exist_ok=True)
        out = est_dir / "daily.parquet"
        if out.exists():
            all_parts.append(pd.read_parquet(out))
            continue
        parts: list[pd.DataFrame] = []
        base = start_day
        with httpx.Client(timeout=120, base_url=_BASE) as client:
            while base < end_day:
                end = min(base + dt.timedelta(days=chunk_days), end_day)
                url = AEMET_DAILY_URL.format(
                    start=base.isoformat() + "T00:00:00UTC",
                    end=end.isoformat() + "T23:59:59UTC",
                    idema=idema,
                )
                try:
                    raw = _aemet_data(client, url)
                except ValueError:
                    raw = []  # tramo sin datos; seguir con el siguiente chunk
                if raw:
                    parts.append(pd.DataFrame(raw))
                base = end + dt.timedelta(days=1)
                time.sleep(1.2)  # cortesía con la API pública
        if parts:
            pdf = pd.concat(parts, ignore_index=True)
            pdf["idema"] = idema
            pdf.to_parquet(out, index=False)
            all_parts.append(pdf)
            print(f"{idema}: {len(pdf)} días")
    if all_parts:
        full = pd.concat(all_parts, ignore_index=True)
        build_manifest(
            "aemet_open_data",
            out_base,
            coverage={
                "n_estaciones_galicia": len(stations),
                "backfill_daily_start": start,
                "obs_esquema": ["idema", "fint", "ta", "hr", "vv", "prec"],
            },
            license="AEMET OpenData — libre con atribución (API pública)",
            description="Obs horaria + climatologías diarias estación por estación",
        )
        return full
    return pd.DataFrame()


if __name__ == "__main__":
    base = Path("data/raw/aemet")
    print(f"estaciones Galicia: {len(stations_galicia(base))}")
    df = fetch_hourly(base)
    print(f"obs horaria hoy: {len(df)} registros")
    backfill_daily(base)
