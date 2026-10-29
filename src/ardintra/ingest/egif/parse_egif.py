"""Parser EGIF: Xml_*.zip del buscador MITECO -> parts_normalized.parquet.

Cada Xml_*.zip contiene un XML con la colección <pifs> de la EGIF.
Extrae por parte (Pif): identificación, localización (lon/lat decimales ya
incluidos), tiempos (deteccion = fecha de ignición), causa, condiciones y
superficies quemadas.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import pandas as pd

from ardintra.ingest.common.galicia import BBOX_GALICIA, PROVINCE_CODES

# Secciones e hijos que extraemos (todas minúsculas tal cual el XML exporta)
_FIELDS = {
    "pif_comun": ["anio", "idestadopif"],
    "pif_localizacion": [
        "idprovincia",
        "idmunicipio",
        "paraje",
        "huso",
        "x",
        "y",
        "latitud",
        "longitud",
        "nummunicipiosafectados",
    ],
    "pif_tiempos": ["deteccion", "controlado", "extinguido", "llegadapmt"],
    "pif_causa": [
        "idcausa",
        "idmotivacion",
        "idgradoresponsabilidad",
        "idcertidumbrecausa",
        "idcausante",
        "diastormenta",
    ],
    "pif_condiciones": [
        "idestacionmeteorologica",
        "tempmaxima",
        "humrelativa",
        "velocidadviento",
        "direccionviento",
        "diasultimalluvia",
    ],
    "pif_perdidas": [
        "superficiearboladatotal",
        "superficienoarboladatotal",
        "superficienoarboladaagricola",
        "superficienoarboladaotras",
    ],
    "pif_incidencias": [
        "afectozonasinterfazurbanoforestal",
        "afectoespacionnatprot",
        "idnivelgravedadmaximo",
    ],
}


def _text(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    if child is None:
        return None
    txt = (child.text or "").strip()
    return txt or None


def _pif_record(pif: ET.Element) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "idpif": _text(pif, "idpif"),
        "numeroparte": _text(pif, "NumeroParte") or _text(pif, "numeroparte"),
    }
    for section, fields in _FIELDS.items():
        sec = pif.find(section)
        if sec is None:
            for f in fields:
                rec[f] = None
            continue
        for f in fields:
            rec[f] = _text(sec, f)
    return rec


def iter_pifs_from_xml(path: Path) -> Iterator[dict[str, Any]]:
    for event, pif in ET.iterparse(path, events=("end",)):
        if pif.tag == "Pif":
            yield _pif_record(pif)
            pif.clear()


def iter_pifs_from_zip_manual_dir(dir_manual: Path) -> Iterator[dict[str, Any]]:
    """Procesa todos los Xml_*.zip de data/raw/egif/manual/** (immutables original)."""
    for zpath in sorted(dir_manual.rglob("Xml_*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for inner in zf.namelist():
                if not inner.lower().endswith(".xml"):
                    continue
                with zf.open(inner) as fh:
                    for event, pif in ET.iterparse(fh, events=("end",)):
                        if pif.tag == "Pif":
                            yield _pif_record(pif)
                            pif.clear()


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower).copy()

    # Coordenadas: cuadHoja es UTM/ETRS89-H29; la trazamos a lon/lat decimales ya entregadas
    for c in ("latitud", "longitud"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dt_cols = ["deteccion", "controlado", "extinguido"]
    for c in dt_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")

    num_cols = [
        "anio",
        "idestadopif",
        "idprovincia",
        "idmunicipio",
        "superficiearboladatotal",
        "superficienoarboladatotal",
        "tempmaxima",
        "humrelativa",
        "velocidadviento",
        "diasultimalluvia",
        "idcausa",
        "idmotivacion",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Superficie total ( partes de EGIF: parcela forestal + no forestal > 0 ha)
    arbol = pd.to_numeric(df.get("superficiearboladatotal"), errors="coerce").fillna(0)
    noarbol = pd.to_numeric(df.get("superficienoarboladatotal"), errors="coerce").fillna(0)
    df["superficie_quemada_ha"] = arbol + noarbol

    # Filtro de dominio: Galicia (idprovincia en {15,27,32,36}) y bbox razonable
    lon_min, lat_min, lon_max, lat_max = BBOX_GALICIA
    in_bbox = df["longitud"].between(lon_min, lon_max)
    in_bbox &= df["latitud"].between(lat_min, lat_max)
    df["in_bbox"] = in_bbox

    if "idprovincia" in df.columns:
        df["provincia"] = df["idprovincia"].map(PROVINCE_CODES).fillna("otra")

    # dict causa simplificado (los full lookup van en causa-lookup en contracts)
    df["es_causa_intencionada"] = df.get("idcausa") == 400
    keep = [
        "numeroparte",
        "idpif",
        "anio",
        "provincia",
        "idmunicipio",
        "paraje",
        "latitud",
        "longitud",
        "deteccion",
        "controlado",
        "extinguido",
        "superficie_quemada_ha",
        "idcausa",
        "idmotivacion",
        "idgradoresponsabilidad",
        "idcertidumbrecausa",
        "idcausante",
        "tempmaxima",
        "humrelativa",
        "velocidadviento",
        "afectozonasinterfazurbanoforestal",
        "afectoespacionnatprot",
        "in_bbox",
        "es_causa_intencionada",
    ]
    return df[keep]


def qa_report(df: pd.DataFrame) -> dict[str, Any]:
    valid_coord = df["in_bbox"].fillna(False)
    has_date = df["deteccion"].notna()
    return {
        "n_partes": int(len(df)),
        "por_provincia": df["provincia"].value_counts(dropna=False).to_dict(),
        "por_anio": df["anio"].value_counts(dropna=False).sort_index().to_dict(),
        "coords_validas_pct": round(100 * valid_coord.mean(), 2),
        "con_fecha_pct": round(100 * has_date.mean(), 2),
        "sin_coords_excluidas": int((~valid_coord & has_date).sum()),
        "superficie_total_ha": round(float(df["superficie_quemada_ha"].sum()), 1),
        "causa_completa_pct": round(100 * df["idcausa"].notna().mean(), 2),
        "candidatos_entrenamiento": int((valid_coord & has_date & (df["provincia"] != "otra")).sum()),
    }


def run(
    dir_manual: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(out_dir or "data/interim/egif")
    dir_manual = dir_manual or Path("data/raw/egif/manual")
    root.mkdir(parents=True, exist_ok=True)

    records = list(iter_pifs_from_zip_manual_dir(dir_manual))
    if not records:
        raise RuntimeError(
            f"No se encontraron Xml_*.zip con <Pif> en {dir_manual}"
        )
    raw = pd.DataFrame.from_records(records)
    df = normalize(raw)
    qa = qa_report(df)
    out = root / "parts_normalized.parquet"
    df.to_parquet(out, index=False)

    # QA además como JSON portable
    qa_path = root / "qa_egif.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    print(f"\n-> {out}  ({len(df):,} partes)")
    return qa


if __name__ == "__main__":
    run()
