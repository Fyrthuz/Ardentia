"""Stage 2: modelo jerárquico espacial sobre los scores del Stage 1.

Réplica python del enfoque "XGBoost meets INLA" (arXiv 2508.09896) adaptado:

    log μ_{r,d} = α + β·log(1+λ_raw_{r,d}) + Σ_k γ_k·MEM_k(r)

- λ_raw: intensidad Stage 1 agregada por hexágono res7 (Σ p calibrada por celda)
- MEM_k: Moran Eigenvector Maps de la adjacency hexagonal res7 — base espacial
  que aproxima el campo gaussiano Matérn del SPDE (efecto aleatorio espacial)
- β recalibra la intensidad Stage 1 (corrige la sobreenconfianza de nivel)
- u_r = MEM·γ captura el patrón espacial residual persistente (clustering de
  igniciones no explicado por las features dinámicas)

Ajuste: 2023 (PoissonRegressor, ridge sobre MEM) · Validación: 2024 (split temporal).
Baselines: (a) λ_raw cruda, (b) isotónica regional 2023 (la usada en producción).

Uso: uv run python -m ardintra.models.stage2_lgcp
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import roc_auc_score

REG = Path("reports/fullgrid_regional.parquet")
OUT = Path("reports/stage2_lgcp.json")
REL_OUT = Path("reports/stage2_reliability.csv")
N_MEM = 80


def moran_eigenvectors(regions: list[str], k: int = N_MEM) -> np.ndarray:
    """Eigenvectors de Moran de la matriz W hexagonal (doble-centrada, simétrica).

    Los eigenvectors con autovalor positivo ordenan patrones espaciales por
    autocorrelación decreciente — la base con la que un campo suave (Matérn)
    se representa de forma óptima en la malla (Griffith 1996; Dray 2006).
    """
    idx = {r: i for i, r in enumerate(regions)}
    n = len(regions)
    rows, cols = [], []
    for r in regions:
        for nb in h3.grid_disk(r, 1):
            if nb != r and nb in idx:
                rows.append(idx[r]); cols.append(idx[nb])
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W = W + W.T
    W.data = np.ones_like(W.data)                      # simetrizada binaria
    deg = np.asarray(W.sum(axis=1)).ravel()
    Ws = sp.diags(1.0 / np.maximum(deg, 1)) @ W        # row-standardized simétrica asim.
    # simetrizar la row-standardizada: D^-1/2 W D^-1/2 (misma estructura de Moran)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1))
    Wsym = sp.diags(dinv) @ W @ sp.diags(dinv)
    vals, vecs = eigsh(Wsym, k=min(k, n - 2), which="LM")
    # autocorrelación de Moran ∝ autovalor de Wsym (positiva = clustering)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-6
    mem = vecs[:, keep] * dinv[:, None]                # deshacer el escalado D^-1/2
    return mem / np.maximum(np.linalg.norm(mem, axis=0, keepdims=True), 1e-9)


def main():
    reg = pd.read_parquet(REG)
    reg["fecha"] = pd.to_datetime(reg["fecha"])
    reg["year"] = reg["fecha"].dt.year
    regions = sorted(reg["r7"].unique())
    ridx = {r: i for i, r in enumerate(regions)}
    print(f"regiones res7: {len(regions):,} · regiones-día: {len(reg):,}")

    print("calculando Moran Eigenvector Maps...", flush=True)
    mem = moran_eigenvectors(regions, N_MEM)
    print(f"MEM: {mem.shape[1]} vectores con autocorrelación positiva")

    tr = reg[reg["year"] == 2023]
    te = reg[reg["year"] == 2024].copy()
    y_tr = tr["n_fuegos_obs"].to_numpy().astype(float)
    y_te = te["n_fuegos_obs"].to_numpy().astype(float)

    def design(d: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(d), 1 + mem.shape[1]), dtype=np.float32)
        X[:, 0] = np.log1p(d["lam"].to_numpy())          # señal Stage 1
        ri = d["r7"].map(ridx).to_numpy()
        X[:, 1:] = mem[ri]
        return X

    X_tr, X_te = design(tr), design(te)
    print(f"matrices de diseño: {X_tr.shape} / {X_te.shape}", flush=True)

    print("ajustando Poisson GLM (2023)...", flush=True)
    glm = PoissonRegressor(alpha=float(__import__('os').environ.get('STAGE2_ALPHA', '1e-6')), max_iter=2000, tol=1e-8)
    glm.fit(X_tr, y_tr)
    beta = glm.coef_[0]
    print(f"β (Stage 1): {beta:.3f} · intercept: {glm.intercept_:.3f} · "
          f"||γ_MEM||: {np.linalg.norm(glm.coef_[1:]):.3f}")

    # --- predicciones 2024
    mu_tr = glm.predict(X_tr)
    mu_te = glm.predict(X_te)
    te["mu_stage2"] = mu_te
    te["p_stage2"] = 1 - np.exp(-mu_te)
    # baselines
    te["p_iso"] = np.nan
    iso23 = tr.assign(hubo=(tr["n_fuegos_obs"] > 0).astype(int))
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["p_alguno"], iso23["hubo"])
    te["p_iso"] = iso.predict(te["p_alguno"])
    te["p_raw"] = te["p_alguno"]

    print("\n=== VALIDACIÓN 2024 ===")
    for col, name in (("p_raw", "λ cruda Stage1"), ("p_iso", "isotónica regional"), ("p_stage2", "Stage 2 (GLM+MEM)")):
        auc = roc_auc_score((te["n_fuegos_obs"] > 0).astype(int), te[col])
        r10 = 0.0
        for _, g in te.groupby("fecha"):
            k = max(int(len(g) * 0.10), 1)
            top = g.nlargest(k, col)
            r10 += top["n_fuegos_obs"].clip(upper=1).sum()
        n_fire_regions = int((te["n_fuegos_obs"] > 0).sum())
        print(f"{name:24s} AUC={auc:.3f} · rec@10% región-día={r10 / n_fire_regions:.1%}")

    # reliability por deciles
    te["bin"] = pd.qcut(te["p_stage2"].rank(method="first"), 10, duplicates="drop")
    rel = te.groupby("bin", observed=True).agg(
        n=("n_fuegos_obs", "size"),
        con_fuego=("n_fuegos_obs", lambda s: (s > 0).sum()),
        p_med=("p_stage2", "median")).reset_index()
    rel["freq_real"] = (rel["con_fuego"] / rel["n"]).round(5)
    print("\nreliability Stage 2 (deciles):")
    print(rel.to_string(index=False))
    rel.to_csv(REL_OUT, index=False)

    # efecto espacial: top regiones por u_r
    u = mem @ glm.coef_[1:]
    u_ser = pd.Series(u, index=regions).sort_values(ascending=False)
    print("\ntop-5 regiones por efecto espacial residual u_r:",
          [(r[:8], round(float(v), 3)) for r, v in u_ser.head(5).items()])

    te[["fecha", "r7", "p_alguno", "p_iso", "mu_stage2", "p_stage2", "n_fuegos_obs"]] \
        .to_parquet("reports/stage2_predictions_2024.parquet", index=False)

    # persistir el modelo para el vídeo y uso operativo
    np.savez_compressed(
        "reports/stage2_model.npz",
        regions=np.array(regions), mem=mem.astype(np.float32),
        coef=glm.coef_.astype(np.float64), intercept=float(glm.intercept_))
    print("-> reports/stage2_model.npz")
    OUT.write_text(json.dumps({
        "beta_stage1": round(float(beta), 4),
        "mem_k": int(mem.shape[1]),
        "norm_gamma_mem": round(float(np.linalg.norm(glm.coef_[1:])), 4),
        "auc_2024": {"raw": round(float(roc_auc_score((te['n_fuegos_obs']>0).astype(int), te['p_raw'])), 4),
                      "iso": round(float(roc_auc_score((te['n_fuegos_obs']>0).astype(int), te['p_iso'])), 4),
                      "stage2": round(float(roc_auc_score((te['n_fuegos_obs']>0).astype(int), te['p_stage2'])), 4)},
        "nota": "Poisson GLM + MEM res7, ajuste 2023, validación 2024",
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
