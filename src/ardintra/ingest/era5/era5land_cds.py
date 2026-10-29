"""ERA5-Land (Copernicus ADS vía cdsapi): humedad del suelo, T2m y punto de rocío.

Descarga netCDF anual por variable sobre el bbox de Galicia (rejilla 0.1°).
La clave va en ~/.cdsapirc o CDS_API_KEY en .env (datasets de atmosfera: ADS).
"""

from __future__ import annotations

import os
from pathlib import Path

import cdsapi

from ardintra.ingest.common.galicia import BBOX_GALICIA
from ardintra.ingest.common.manifest import build_manifest

VARIABLES = {
    "soil_moisture_l1": "volumetric_soil_water_layer_1",
    "soil_moisture_l2": "volumetric_soil_water_layer_2",
    "t2m": "2m_temperature",
    "td2m": "2m_dewpoint_temperature",
}
YEARS = range(2017, 2025)
OUT_BASE = Path("data/raw/era5land")


def _client() -> cdsapi.Client:
    key = os.environ.get("CDS_API_KEY", "").strip()
    if key and not os.path.exists(os.path.expanduser("~/.cdsapirc")):
        # ERA5-Land está en el CDS (cds.climate.copernicus.eu), no en el ADS
        return cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=key)
    # fallback: ~/.cdsapirc ya configurado
    return cdsapi.Client()


def run(years: range | None = None) -> list[Path]:
    years = years or YEARS
    c = _client()
    done: list[Path] = []
    area = [BBOX_GALICIA[3], BBOX_GALICIA[0], BBOX_GALICIA[1], BBOX_GALICIA[2]]  # [N, W, S, E]
    for var, name in VARIABLES.items():
        for year in years:
            out = OUT_BASE / f"{var}_{year}.nc"
            if out.exists() and out.stat().st_size > 100_000:
                done.append(out)
                print(f"skip {out.name}")
                continue
            c.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": name,
                    "year": str(year),
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
                    "area": area,
                    "format": "netcdf",
                },
                target=str(out),
            )
            done.append(out)
            print(f"ok {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    build_manifest(
        "era5land_cds",
        OUT_BASE,
        coverage={"years": list(years), "variables": list(VARIABLES.values()), "area_bbox": BBOX_GALICIA},
        license="Copernicus CDS licence (CC-BY-4.0 reanalysis credit ECMWF)",
        description="ERA5-Land 2017-2024 Galicia: SWC l1/l2, T2m, Td2m (VPD derivable)",
    )
    return done


if __name__ == "__main__":
    run()
