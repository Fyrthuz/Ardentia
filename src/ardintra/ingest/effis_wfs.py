"""Descarga de perímetros quemados EFFIS (RDII) vía WFS público.

Sin autenticación. WFS 1.0.0 (devuelve lon,lat correctas; 1.1.0 invierte ejes).
Capas por año: modis.ba.poly.{2017..2025} (MMU ~30 ha).
Salida: data/raw/effis/year={yyy}/perimeters.parquet + manifest.json
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ardintra.ingest.common.galicia import BBOX_GALICIA, CRS_WGS84, EFFIS_URL
from ardintra.ingest.common.manifest import build_manifest

_GALICIA_PROVINCES = {
    "A CORUÑA", "A CORUNA", "CORUNA", "LUGO", "OURENSE", "PONTEVEDRA",
    "LA CORUÑA", "CORUN/A", "ORENSE",
}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4))
def _get_feature_xml(year: int, bbox: tuple[float, float, float, float]) -> str:
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typename": f"modis.ba.poly.{year}",
        "srsName": "EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
    }
    # timeout generoso: parámetros de un año entero pueden tardar en componerse
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        r = client.get(EFFIS_URL, params=params)
        r.raise_for_status()
        text = r.text
    if "ServiceException" in text[:2000]:
        raise RuntimeError(f"EFFIS WFS ServiceException año {year}: {text[:300]}")
    return text


def _year_to_gdf(year: int, out_parquet: Path) -> int:
    xml = _get_feature_xml(year, BBOX_GALICIA)
    with tempfile.NamedTemporaryFile("w", suffix=".gml", delete=False, encoding="utf-8") as fh:
        fh.write(xml)
        tmp = Path(fh.name)
    try:
        gdf = gpd.read_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    if gdf.empty:
        return 0
    gdf = gdf.set_crs(CRS_WGS84, allow_override=True)
    # Normaliza la estrategia de filtrado: solo quemados dentro de Galicia
    cols_lower = {c.lower(): c for c in gdf.columns}
    prov_col = cols_lower.get("province")
    if prov_col is not None:
        mask = gdf[prov_col].astype(str).str.upper().isin(_GALICIA_PROVINCES)
        gdf = gdf[mask]
    gdf["effis_year"] = year
    gdf.to_parquet(out_parquet, index=False)
    return len(gdf)


def run(out_base: Path | None = None, years: range | None = None) -> dict[str, int]:
    out_base = out_base or Path("data/raw/effis")
    years = years or range(2017, 2026)
    counts: dict[str, int] = {}
    for year in years:
        snap = out_base / f"year={year}"
        snap.mkdir(parents=True, exist_ok=True)
        out = snap / "perimeters.parquet"
        n = _year_to_gdf(year, out)
        counts[str(year)] = n
        print(f"EFFIS {year}: {n} perímetros")
    build_manifest(
        "effis_wfs",
        out_base,
        coverage={"years": list(years), "bbox": BBOX_GALICIA, "perimeters": counts},
        license="EU Data License (Copernicus EMS / EFFIS) — libre, sin auth",
        description="Perímetros quemados RDII (MODIS/VIIRS, >=30 ha) vía WFS 1.0.0",
    )
    return counts


if __name__ == "__main__":
    run()
