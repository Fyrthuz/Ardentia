"""Sistema Canadian FWI (Van Wagner 1985/1987) — port 1:1 del paquete R cffdrs.

Fuente de las ecuaciones: github.com/cffdrs/cffdrs_r (fine_fuel_moisture_code,
duff_moisture_code, drought_code, initial_spread_index, buildup_index,
fire_weather_index). Vectorizado por estación: el estado (FFMC/DMC/DC) avanza
día a día.

Convenciones para el port a nuestras series:
  temp  = tmax_c (la referencia usa T al mediodía; tmax como proxy documentado)
  rh    = rh_mean (HR media diaria)
  ws    = wind_max_kmh (racha como proxy del viento al mediodía)
  prec  = prec_mm (sumatorio diario)
  lat/mon: ajuste de longitud del día por latitud (DC y DMC).
Estado inicial: FFMC=85, DMC=6, DC=15 el 1 de enero del primer año (para Jun-Sep
el encadenado ya convergió; documentado en el reporte de Fase 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0


def _ffmc(ffmc_yda, temp, rh, ws, prec):
    wmo = FFMC_COEFFICIENT * (101 - ffmc_yda) / (59.5 + ffmc_yda)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ra = prec
        # Eq. 3a-b: llama con precip > 0.5
        def wet_251(wmo, ra):
            return wmo + 42.5 * ra * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra))
        wmo = np.where(
            prec > 0.5,
            np.where(wmo > 150, wmo + 0.0015 * (wmo - 150) ** 2 * np.sqrt(ra) + wet_251(wmo, ra), wet_251(wmo, ra)),
            wmo,
        )
        wmo = np.minimum(wmo, 250.0)
        ed = 0.942 * (rh ** 0.679) + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - 1 / np.exp(rh * 0.115))
        ew = 0.618 * (rh ** 0.753) + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - 1 / np.exp(rh * 0.115))
        z0 = 0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - ((100 - rh) / 100) ** 8)
        z = np.where(wmo < ed and False, z0, z0)  # placeholder evitado abajo
        drying = (wmo < ed) & (wmo < ew)
        x_dry = z0 * 0.581 * np.exp(0.0365 * temp)
        wm = np.where(drying, ew - (ew - wmo) / (10 ** x_dry), wmo)
        z1 = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - (rh / 100) ** 8)
        x_wet = np.where(wmo > ed, z1 * 0.581 * np.exp(0.0365 * temp), x_dry)
        wm = np.where(wmo > ed, ed + (wmo - ed) / (10 ** x_wet), wm)
        ffmc1 = (59.5 * (250 - wm)) / (FFMC_COEFFICIENT + wm)
        return np.clip(ffmc1, 0, 101)


def _dmc(dmc_yda, temp, rh, prec, lat, mon):
    temp = np.maximum(temp, -1.1)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ell01 = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
        ell02 = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8])
        rk_base = 1.894 * (temp + 1.1) * (100 - rh) * ell01[mon - 1] * 1e-4
        rk = np.where((lat <= 30) & (lat > 10),
                      1.894 * (temp + 1.1) * (100 - rh) * ell02[mon - 1] * 1e-4,
                      rk_base)
        # Galicia (42-44N) usa ell01 canadiense (>=30N); mantengo el ajuste de fuente
        wet = prec > 1.5
        ra = prec
        rw = 0.92 * ra - 1.27
        wmi = 20 + 280 / np.exp(0.023 * dmc_yda)
        b = np.where(
            dmc_yda <= 33,
            100 / (0.5 + 0.3 * dmc_yda),
            np.where(dmc_yda <= 65, 14 - 1.3 * np.log(dmc_yda), 6.2 * np.log(dmc_yda) - 17.2),
        )
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        pr_wet = 43.43 * (5.6348 - np.log(np.maximum(wmr - 20, 1e-9)))
        pr = np.where(wet, pr_wet, dmc_yda)
        pr = np.maximum(pr, 0.0)
        dmc1 = pr + rk
        return np.maximum(dmc1, 0.0)


def _dc(dc_yda, temp, prec, lat, mon):
    temp = np.maximum(temp, -2.8)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        fl01 = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])
        pe = np.maximum((0.36 * (temp + 2.8) + fl01[mon - 1]) / 2, 0.0)  # lat>20N → fl01
        rw = 0.83 * prec - 1.27
        smi = 800 * np.exp(-dc_yda / 400)
        dr0 = dc_yda - 400 * np.log(1 + 3.937 * rw / np.maximum(smi, 1e-9))
        dr = np.where(prec <= 2.8, dc_yda, np.maximum(dr0, 0.0))
        return np.maximum(dr + pe, 0.0)


def _isi(ffmc, ws):
    fm = FFMC_COEFFICIENT * (101 - ffmc) / (59.5 + ffmc)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        fW = np.exp(0.05039 * ws)
        fF = 91.9 * np.exp(-0.1386 * fm) * (1 + (fm ** 5.31) / 49_300_000)
        return 0.208 * fW * fF


def _bui(dmc, dc):
    with np.errstate(divide='ignore', invalid='ignore'):
        bui1 = np.where((dmc == 0) & (dc == 0), 0.0, 0.8 * dc * dmc / (dmc + 0.4 * dc))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        p = np.where(dmc == 0, 0.0, (dmc - bui1) / dmc)
        cc = 0.92 + ((0.0114 * dmc) ** 1.7)
        bui0 = dmc - cc * p
    return np.where(bui1 < dmc, np.maximum(bui0, 0.0), bui1)


def _fwi(isi_v, bui_v):
    bb = np.where(
        bui_v > 80,
        0.1 * isi_v * (1000 / (25 + 108.64 / np.exp(0.023 * bui_v))),
        0.1 * isi_v * (0.626 * (bui_v ** 0.809) + 2),
    )
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(bb <= 1, bb, np.exp(2.72 * ((0.434 * np.log(bb)) ** 0.647)))


    FWI_COMPONENTS = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")


def fwi_series(df_st: pd.DataFrame) -> pd.DataFrame:
    """df: [fecha, tmax_c, rh_mean, wind_max_kmh, prec_mm, lat] ordenado por fecha
    → ffmc/dmc/dc/isi/bui/fwi diarios (NaN conservando estado ante huecos)."""
    ffmc = 85.0
    dmc = 6.0
    dc_s = 15.0
    lat = float(df_st["lat"].iloc[0]) if len(df_st) else 43.0
    out = []
    for row in df_st.itertuples(index=False):
        t, rh, wind, prec = row.tmax_c, row.rh_mean, row.wind_max_kmh, row.prec_mm
        if any(pd.isna(v) for v in (t, rh, wind, prec)):
            out.append((row.fecha, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue  # hueco: conserva estado
        mon = pd.Timestamp(row.fecha).month
        f_new = float(_ffmc(np.array([ffmc]), np.array([t]), np.array([rh]), np.array([wind]), np.array([prec]))[0])
        d_new = float(_dmc(np.array([dmc]), np.array([t]), np.array([rh]), np.array([prec]), np.array([lat]), mon)[0])
        c_new = float(_dc(np.array([dc_s]), np.array([t]), np.array([prec]), np.array([lat]), mon)[0])
        isi_v = float(_isi(np.array([f_new]), np.array([wind]))[0])
        bui_v = float(_bui(np.array([d_new]), np.array([c_new]))[0])
        fwi_v = float(_fwi(np.array([isi_v]), np.array([bui_v]))[0])
        ffmc, dmc, dc_s = f_new, d_new, c_new
        out.append((row.fecha, f_new, d_new, c_new, isi_v, bui_v, fwi_v))
    return pd.DataFrame(
        out, columns=["fecha", "ffmc", "dmc", "dc", "isi", "bui", "fwi"]
    )


if __name__ == "__main__":
    # smoke: una estación real con datos (checkpoint openmeteo crudo → esquema estándar)
    from pathlib import Path
    RENAME = {
        "time": "fecha",
        "temperature_2m_max": "tmax_c",
        "precipitation_sum": "prec_mm",
        "windspeed_10m_max": "wind_max_kmh",
        "relative_humidity_2m_mean": "rh_mean",
    }
    ejemplo = next(iter(Path("data/raw/openmeteo/per_station").glob("*.parquet")))
    df = pd.read_parquet(ejemplo).rename(columns=RENAME)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["lat"] = df["lat"].iloc[0]
    res = fwi_series(df[["fecha", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm", "lat"]].dropna(
        subset=["fecha", "tmax_c", "rh_mean", "wind_max_kmh", "prec_mm"]
    ))
    verano = res[(res.fecha.dt.month.isin([7, 8]))]
    print(str(ejemplo))
    print(f"días calculados: {len(res)} | verano p50 FWI: {verano.fwi.median():.1f} | p95: {verano.fwi.quantile(.95):.1f}")
    print(verano.sort_values('fwi', ascending=False).head(3).to_string(index=False))
