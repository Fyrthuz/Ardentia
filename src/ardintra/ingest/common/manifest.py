"""Manifiestos de provenance para data/raw/ y data/interim/.

Cada fuente deja un manifest.json con: fuente, fecha de ejecución, cobertura,
lista de ficheros (path, sha256, bytes) y licencia. Los manifests se versionan
en git aunque los datos no.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ARDENTIA_ROOT", Path(__file__).resolve().parents[4]))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_manifest(
    source: str,
    out_dir: Path,
    *,
    coverage: dict[str, Any],
    license: str,
    description: str = "",
    skip_ext: tuple[str, ...] = () or (),
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name == "manifest.json":
            continue
        if p.suffix.lower() in skip_ext:
            continue
        files.append(
            {
                "path": str(p.relative_to(out_dir)),
                "sha256": sha256_of(p),
                "bytes": p.stat().st_size,
            }
        )
    try:
        rel_out = str(out_dir.relative_to(ROOT))
    except ValueError:
        rel_out = str(out_dir)
    manifest = {
        "source": source,
        "run_date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "description": description,
        "coverage": coverage,
        "license": license,
        "out_dir": rel_out,
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_manifest(out_dir: Path) -> dict[str, Any] | None:
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        return None
    return json.loads(mpath.read_text(encoding="utf-8"))


_SNAP_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def snapshot_dir(base: Path, day: str | None = None) -> Path:
    """Subdirectorio por fecha de descarga (por defecto hoy, UTC)."""
    if day is None:
        day = dt.date.today().isoformat()
    if not _SNAP_RE.fullmatch(day):
        raise ValueError(f"fecha inválida: {day}")
    d = base / day
    d.mkdir(parents=True, exist_ok=True)
    return d
