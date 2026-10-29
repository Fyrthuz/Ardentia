"""Backfill AEMET ("paced") de las estaciones que Open-Meteo no llegó a bajar.

Momento correcto: solo 19 estaciones × 116 tramos × 2 peticiones = ~4.400 req
(a ~1,2 s petición ≈ 90-100 min) — viable en background. El set completo de 58
era lo caro (~13k peticiones); eso sí se movió a Open-Meteo.

Salida: data/raw/aemet/missing_backfill/estacion={idema}.parquet (esquema openmeteo)
  TBD checkpoint también en openmeteo/per_station/ para que la consolidación
  diaria los pique desde un solo sitio.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from ardintra.ingest.aemet.aemet_obs import (
    _BASE,
    _decode_bytes,
    _key,
    stations_galicia,
)
from ardintra.ingest.common.galicia import AEMET_DAILY_URL
from ardintra.ingest.common.manifest import build_manifest

OUT = Path("data/raw/aemet/missing_backfill")
OM_STATIONS = Path("data/raw/openmeteo/per_station")
START = "2017-01-01"
CHUNK_DAYS = 30
# Límite DOCUMENTADO por AEMET (FAQ §3.5): 40 conexiones/minuto por usuario.
# 2.5 s/req ≈ 24/min (40% del límite) para no tocar el 429.
PACE_S = 2.5

RENAMER = {
    "fecha": "fecha",
    "tmax": "tmax_c",
    "tmin": "tmin_c",
    "prec": "prec_mm",
    "velmedia": "velmedia_kmh",
    "racha": "racha_kmh",
    "hrMedia": "rh_mean",
}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _parquet_days(raw: list[dict], idema: str) -> pd.DataFrame | None:
    if not raw:
        return None
    df = pd.DataFrame(raw)
    df = df.rename(columns=RENAMER)
    df["fecha"] = pd.to_datetime(df.get("fecha"), errors="coerce")
    for c in ("tmax_c", "tmin_c", "prec_mm", "rh_mean", "velmedia_kmh", "racha_kmh"):
        df[c] = _num(df[c]) if c in df.columns else np.nan
    # Open-Meteo viene en km/h max diaria; AEMET da media y racha: usamos racha (max)
    df["wind_max_kmh"] = df["racha_kmh"].where(df["racha_kmh"].notna(), df["velmedia_kmh"])
    df["idema"] = idema
    keep = ["fecha", "tmax_c", "tmin_c", "prec_mm", "rh_mean", "wind_max_kmh", "idema"]
    return df[keep].dropna(subset=["fecha"])


def missing_stations() -> pd.DataFrame:
    stations = stations_galicia(Path("data/raw/aemet"))  # caché: stations.parquet, sin llamada API
    done = {p.name.split("=")[1].replace(".parquet", "") for p in (OM_STATIONS.rglob("*.parquet") if OM_STATIONS.exists() else [])}
    done |= {p.name.split("=")[1].replace(".parquet", "") for p in (OUT.rglob("*.parquet") if OUT.exists() else [])}
    return stations[~stations["idema"].isin(done)]


def _chunk(client: httpx.Client, idema: str, base: dt.date, end: dt.date) -> list[dict]:
    """Un tramo mensual de la serie diaria. Todo tramo problemático → [] (skip),
    salvo 5xx confirmado que se relanza. Debe NUNCA matar el proceso."""
    url = AEMET_DAILY_URL.format(
        start=base.isoformat() + "T00:00:00UTC",
        end=end.isoformat() + "T23:59:59UTC",
        idema=idema,
    )
    r = client.get(url, params={"api_key": _key()})
    if r.status_code >= 500:
        raise httpx.HTTPStatusError(f"5xx {url}", request=r.request, response=r)
    # ACK también puede venir mal codificado: decodificar tolerante
    try:
        payload = json.loads(_decode_bytes(r.content))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  ! {idema} tramo {base}: ACK no parseable ({content := r.content[:60]!r}, {type(e).__name__}) -> skipped")
        return []
    estado = payload.get("estado")
    if estado == 200:
        datos = client.get(payload["datos"])
        content = datos.content.strip()
        if not content:
            return []
        try:
            raw = json.loads(_decode_bytes(content))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  ! {idema} tramo {base}: cuerpo no-JSON ({content[:60]!r}) -> skipped")
            return []
        return raw if isinstance(raw, list) else []
    if estado in (400, 404):  # tramo sin datos/estación sin serie en ese período
        return []
    raise RuntimeError(f"AEMET estado={estado} en {url} ({str(payload.get('descripcion', ''))[:80]})")


def run(end: str | None = None, pace_s: float = PACE_S) -> dict[str, int]:
    base = Path("data/raw/aemet")
    OUT.mkdir(parents=True, exist_ok=True)
    end_day = dt.date.fromisoformat(end) if end else dt.date.today()
    miss = missing_stations()
    print(f"estaciones sin datos histórico: {len(miss)}")
    per_station: dict[str, int] = {}
    with httpx.Client(base_url=_BASE, timeout=120) as client:
        for _, st in miss.iterrows():
            idema = st["idema"]
            try:
                raw: list[dict] = []
                b = dt.date(2017, 1, 1)
                n429 = 0
                while b < end_day:
                    d_end = min(b + dt.timedelta(days=CHUNK_DAYS), end_day)
                    try:
                        raw += _chunk(client, idema, b, d_end)
                    except Exception as e:  # noqa: BLE001
                        msg = f"{type(e).__name__}: {str(e)[:90]}"
                        if "estado=429" in msg:
                            # rate limit de AEMET: NO descartar — esperar y repetir el MISMO tramo
                            n429 += 1
                            if n429 > 30:
                                print(f"  ✗ {idema}: >30 pases con 429, abandono de estación")
                                break
                            wait = 600 * (1 + n429 // 8)
                            print(f"  429 {idema} tramo {b}: espera {wait} s y reintento (intento {n429})")
                            time.sleep(wait)
                            continue  # NO avanzar b
                        else:
                            print(f"  ! {idema} tramo {b}..{d_end}: {msg} -> skipped (dato perdido)")
                    b = d_end + dt.timedelta(days=1)
                    time.sleep(pace_s)
                df = _parquet_days(raw, idema)
                if df is None or df.empty:
                    per_station[idema] = 0
                    print(f"  {idema} {st['nombre']}: sin serie diaria usable")
                    continue
                df.to_parquet(OUT / f"estacion={idema}.parquet", index=False)
                # checkpoint compartido para la consolidación future (openmeteo dir)
                (OM_STATIONS / f"estacion={idema}.parquet").parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(OM_STATIONS / f"estacion={idema}.parquet", index=False)
                per_station[idema] = len(df)
                print(f"  {idema} {st['nombre']}: {len(df)} días")
            except Exception as e:  # noqa: BLE001  (el run completo nunca por una estación)
                per_station[idema] = -1
                print(f"  ✗ {idema} {st['nombre']}: fallo estación ({type(e).__name__}: {str(e)[:80]})")

    build_manifest(
        "aemet_missing_backfill",
        OUT,
        coverage={"estaciones": per_station, "start": START,
                  "end": str(end_day), "n_estaciones": len(miss)},
        license="AEMET OpenData — libre con atribución",
        description="Backfill oficial paced de las estaciones que faltaban en Open-Meteo",
    )
    return per_station


if __name__ == "__main__":
    run()
