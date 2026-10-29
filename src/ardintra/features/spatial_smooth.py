"""Suavizado espacial sobre la malla H3: media ponderada por anillos vecinos.

El usuario acertó: la probabilidad cruda por celda es "sal y pimienta" (una celda
0.7 junto a otra 0.0001). El post-proceso hace que el campo sea una distribución
suave: p_suavizada = Σ w_k · p_vecino, con peso propio 0.5, anillo-1 0.35 y
anillo-2 0.15 repartidos. Matriz sparse construida UNA vez y cacheada en disco.

Uso:
  from ardintra.features.spatial_smooth import get_smooth_matrix
  W = get_smooth_matrix(h9_list)          # scipy.sparse csr (N×N), cacheada
  p_suave = W @ p
"""

from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

CACHE = Path("data/interim/h3_smooth_matrix.npz")


def get_smooth_matrix(h9_list, k: int = 2, w_self: float = 0.5) -> csr_matrix:
    """Matriz de suavizado (row-normalized) para la lista de celdas H3 dada."""
    if CACHE.exists():
        try:
            z = np.load(CACHE, allow_pickle=True)
            if list(z["h9"]) == list(h9_list):
                return csr_matrix((z["data"], z["rows"], z["cols"]), shape=z["shape"])
        except Exception:  # noqa: BLE001
            pass

    idx = {h: i for i, h in enumerate(h9_list)}
    rows, cols, vals = [], [], []
    for i, h in enumerate(h9_list):
        ring = h3.grid_disk(h, k)
        w_ring = {n: (w_self if n == h else (0.35 / 6 if h3.grid_distance(h, n) == 1 else 0.15 / 12))
                  for n in ring if n in idx}
        tot = sum(w_ring.values())
        for n, w in w_ring.items():
            rows.append(i)
            cols.append(idx[n])
            vals.append(w / tot)

    W = csr_matrix((vals, (rows, cols)), shape=(len(h9_list), len(h9_list)))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, h9=np.array(h9_list), rows=np.array(rows), cols=np.array(cols),
                        data=np.array(vals), shape=np.array(W.shape))
    return W


def smooth(p: np.ndarray, W: csr_matrix) -> np.ndarray:
    return np.asarray(W @ p).ravel()
