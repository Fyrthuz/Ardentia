"""Constantes del dominio Piloto Galicia."""

AREA = "galicia"

# BBOX Galicia (lon_min, lat_min, lon_max, lat_max), EPSG:4326
BBOX_GALICIA = (-9.65, 41.75, -6.60, 43.95)

# Provincias EGIF (idprovincia según codificación INE)
PROVINCE_CODES = {
    15: "a_coruna",
    27: "lugo",
    36: "pontevedra",
    32: "ourense",
}

# EPSG oficial de la parcelaria española (MDT25, OSM reproyecta a esto)
CRS_ETRS89_UTM29 = "EPSG:25829"
CRS_WGS84 = "EPSG:4326"

EFFIS_URL = "https://maps.effis.emergency.copernicus.eu/effis"
EFFIS_LAYER_FMT = "modis.ba.poly.{year}"
EFFIS_YEARS = range(2017, 2026)  # coherente con Sentinel-2 L2A (2017+)

AEMET_OBS_URL = "https://opendata.aemet.es/opendata/api/observacion/convencional/todas"
AEMET_STATION_URL = (
    "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"
)
AEMET_DAILY_URL = (
    "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"
    "/fechaini/{start}/fechafin/{end}/estacion/{idema}"
)
