"""FIRMS histórico: descarga por chunks de 5 días (límite NRT) sin pasarse del rate limit.

Estrategia anti-rate-limit:
  - chunks de 5 días (máximo permitido por la API NRT)
  - pausa de 3 s entre peticiones (~20/min, bien bajo el límite)
  - reanudable por chunk (guarda cada chunk en un CSV individual)
  - solo descarga VIIRS_SNPP_NRT (375 m, mejor resolución que MODIS 1 km)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("data/raw/firms/historical")
BBOX = "-9.65,41.75,-6.60,43.95"
KEY = os.environ.get("MAP_API_KEY", "")
PAUSE_S = 3.0
CHUNK_DAYS = 5
SOURCE = "VIIRS_SNPP_NRT"


def date_chunks(start: str, end: str, chunk_days: int = CHUNK_DAYS):
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    while d0 < d1:
        d_next = min(d0 + timedelta(days=chunk_days), d1)
        yield d0.isoformat(), d_next.isoformat()
        d0 = d_next + timedelta(days=1)


def run(start: str = "2023-01-01", end: str = "2024-12-31"):
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = list(date_chunks(start, end))
    print(f"FIRMS histórico: {len(chunks)} chunks de {CHUNK_DAYS} días ({start} → {end})")
    print(f"pausa {PAUSE_S}s entre peticiones · total estimado {len(chunks) * PAUSE_S / 60:.0f} min")

    all_dfs = []
    for i, (d_start, d_end) in enumerate(chunks):
        out_file = OUT / f"viirs_{d_start}_{d_end}.csv"
        if out_file.exists() and out_file.stat().st_size > 100:
            df = pd.read_csv(out_file)
            all_dfs.append(df)
            continue
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{SOURCE}/{BBOX}/{CHUNK_DAYS}/{d_start}"
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429:
                print(f"  429 en {d_start}: espera 60 s")
                time.sleep(60)
                r = requests.get(url, timeout=120)
            if r.status_code != 200:
                print(f"  ! {d_start}: HTTP {r.status_code} → skip")
                time.sleep(PAUSE_S)
                continue
            if "Invalid" in r.text[:50]:
                print(f"  ! {d_start}: {r.text[:60]} → skip")
                time.sleep(PAUSE_S)
                continue
            df = pd.read_csv(__import__("io").StringIO(r.text))
            if len(df) > 0:
                df.to_csv(out_file, index=False)
                all_dfs.append(df)
                print(f"  {d_start}→{d_end}: {len(df)} detecciones", flush=True)
            else:
                # guardar vacío para no re-pedir
                out_file.write_text("")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {d_start}: {type(e).__name__}: {str(e)[:60]}")
        time.sleep(PAUSE_S)

    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        full.to_csv(OUT / "viirs_historical.csv", index=False)
        print(f"-> {OUT / 'viirs_historical.csv'} ({len(full):,} detecciones totales)")
    else:
        print("sin detecciones")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2024-12-31"
    run(start, end)
