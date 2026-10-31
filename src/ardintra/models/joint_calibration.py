"""Dependencia espacial (frailty NB = copula gamma) + conformal prediction regional.

1. Frailty negativo-binomial: las celdas comparten el tiempo (mismo FWI) → los fuegos
   de un día están sobredispersos respecto al Poisson. Modelo: N_d | λ_d ~ Poisson(γ_d·λ_d),
   γ_d ~ Gamma(k, escala 1)  ⇒  N_d ~ NB con media λ_d y dispersión k.
   La gamma frailty ES el factor meteorológico compartido (equivalente a copula gaussiana
   de un factor común). Ajuste MLE de k en 2023, validación en 2024.

2. Split conformal regional: score = 1 - p_reg. Umbral cuantílico en 2023 → conjunto de
   regiones con garantía de cobertura marginal 1-α. Evaluación 2024: cobertura real
   y tamaño medio del conjunto vs top-K% fijo.

Uso: uv run python -m ardintra.models.joint_calibration
"""

from __future__ import annotations

import json
from math import lgamma, log, log1p
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REG_RAW = Path("reports/fullgrid_regional.parquet")
REG_CAL = Path("reports/fullgrid_regional_calibrado.parquet")
OUT = Path("reports/joint_calibration.json")


def nb_loglik_pmf(n: int, lam: float, k: float) -> float:
    """log P(N=n) para NB con media lam y frailty Gamma(k)."""
    lam = max(lam, 1e-9)  # floor: isotónica puede aplastar λ a 0 exacto
    return (lgamma(n + k) - lgamma(k) - lgamma(n + 1)
            + n * log(lam / (lam + k)) + k * log(k / (lam + k)))


def fit_k(dias: pd.DataFrame) -> float:
    """MLE de k sobre días con (lambda_honest, n_fuegos)."""
    lam = dias["lam_honest"].to_numpy()
    n = dias["n_fuegos"].to_numpy()

    def negll(log_k: float) -> float:
        k = np.exp(log_k)
        ll = sum(nb_loglog_pmf(int(ni), li, k) for ni, li in zip(n, lam))
        return -ll

    res = minimize_scalar(negll, bounds=(-5, 7), method="bounded")
    return float(np.exp(res.x))


def nb_loglog_pmf(n, lam, k):
    return nb_loglik_pmf(n, lam, k)


def main():
    raw = pd.read_parquet(REG_RAW)
    cal = pd.read_parquet(REG_CAL)  # tiene p_reg (recalibrada) para todas las regiones-día
    for d in (raw, cal):
        d["fecha"] = pd.to_datetime(d["fecha"])
    dts = pd.merge(raw[["fecha", "r7", "lam", "p_alguno", "n_fuegos_obs"]],
                   cal[["fecha", "r7", "p_reg"]], on=["fecha", "r7"], how="left")
    dts["year"] = dts["fecha"].dt.year
    # el parquet calibrado solo trae 2024 → recomputar p_reg de 2023
    # (isotónica regional ajustada en 2023; in-sample para 2023, honesta para 2024)
    if dts["p_reg"].isna().any():
        from sklearn.isotonic import IsotonicRegression
        m23 = dts["year"] == 2023
        iso_r = IsotonicRegression(out_of_bounds="clip").fit(
            dts.loc[m23, "p_alguno"], (dts.loc[m23, "n_fuegos_obs"] > 0).astype(int))
        dts.loc[m23, "p_reg"] = iso_r.predict(dts.loc[m23, "p_alguno"])
        print("p_reg de 2023 recomputada con isotónica regional (in-sample)")
    # lambda honesta por región: -log(1 - p_reg)
    dts["lam_reg_honest"] = -np.log1p(-dts["p_reg"].clip(0, 0.999))

    # --- 1) diario: lambda Galicia = suma de lambdas regionales honestas
    daily = dts.groupby(["fecha", "year"]).agg(
        lam_honest=("lam_reg_honest", "sum"),
        n_fuegos=("n_fuegos_obs", "sum"),
        n_fuegos_region=("n_fuegos_obs", lambda s: (s > 0).sum()),
    ).reset_index()

    tr = daily[daily["year"] == 2023]
    te = daily[daily["year"] == 2024]
    k_hat = fit_k(tr)
    print(f"k (frailty) ajustado en 2023: {k_hat:.3f}")
    print(f"  → Var[N] = λ + λ²/k  (k→∞ sería Poisson puro)")

    # comparación log-lik Poisson vs NB en train y test
    def ll(dias, k):
        return sum(nb_loglik_pmf(int(r.n_fuegos), r.lam_honest, k) for r in dias.itertuples())

    print(f"\nlog-lik 2023:  Poisson {ll(tr, 1e9):8.2f}  ·  NB(k={k_hat:.2f}) {ll(tr, k_hat):8.2f}")
    print(f"log-lik 2024:  Poisson {ll(te, 1e9):8.2f}  ·  NB(k={k_hat:.2f}) {ll(te, k_hat):8.2f}")

    # P(>=1) por día 2024: NB vs Poisson vs realidad (todos los días del parquet tienen fuego,
    # así que la validación de P(>=1) es solo sobre días activos — sesgo documentado)
    te = te.copy()
    te["p_nb_ge1"] = 1 - np.exp([nb_loglik_pmf(0, l, k_hat) for l in te["lam_honest"]])
    te["p_pois_ge1"] = 1 - np.exp(-te["lam_honest"])
    print("\nP(>=1 fuego) días activos 2024 — NB vs Poisson (media):")
    print(f"  NB: {te['p_nb_ge1'].mean():.3f} · Poisson: {te['p_pois_ge1'].mean():.3f} · realidad: 1.000 (solo días con fuego)")

    # distribución de conteos 2024 predicha vs real
    print("\nConteos diarios 2024 (27 días activos):")
    counts_real = te["n_fuegos"].value_counts().sort_index()
    for n in sorted(set(counts_real.index) | {1, 2, 3}):
        p_nb = np.mean([np.exp(nb_loglik_pmf(int(n), l, k_hat)) for l in te["lam_honest"]])
        p_po = np.mean([np.exp(nb_loglik_pmf(int(n), l, 1e9)) for l in te["lam_honest"]])
        real = counts_real.get(n, 0) / len(te)
        print(f"  N={n}: real {real:.3f} · NB {p_nb:.3f} · Poisson {p_po:.3f}")

    # --- 2) conformal regional (split temporal 2023 → 2024)
    # score = 1 - p_alguno CRUDO (conformal es distribution-free: vale cualquier score
    # monotónico con la incidencia; no requiere la isotónica regional)
    reg23 = dts[dts["year"] == 2023]
    reg24 = dts[dts["year"] == 2024].copy()
    n23 = len(reg23)
    alpha = 0.10
    scores23 = 1.0 - reg23["p_alguno"].to_numpy()
    qhat = np.quantile(scores23, min(np.ceil((n23 + 1) * (1 - alpha)) / n23, 1.0))
    reg24["in_set"] = (1.0 - reg24["p_alguno"]) <= qhat
    set_size = reg24.groupby("fecha")["in_set"].sum()
    hits = reg24[reg24["n_fuegos_obs"] > 0]
    cov = hits["in_set"].mean()
    print(f"\n=== CONFORMAL REGIONAL (α={alpha}, q̂={qhat:.6f}) ===")
    print(f"tamaño medio del conjunto: {set_size.mean():.0f} regiones "
          f"({set_size.mean() / reg24.groupby('fecha').size().mean():.1%} del territorio)")
    print(f"cobertura real de fuegos-region 2024: {cov:.1%} "
          f"({int(hits['in_set'].sum())}/{len(hits)} · garantía nominal ≥{1-alpha:.0%})")

    # comparación con top-K% fijo por p_reg
    for pct in (0.05, 0.10, 0.20):
        thr = reg24.groupby("fecha")["p_reg"].transform(lambda s: s.quantile(1 - pct))
        sel = reg24["p_reg"] >= thr
        cov_k = reg24[reg24["n_fuegos_obs"] > 0].loc[sel[reg24["n_fuegos_obs"] > 0]].shape[0] / len(hits)
        print(f"top {pct:.0%} fijo: cobertura {cov_k:.1%}")

    out = {
        "k_frailty": round(k_hat, 3),
        "ll_2023_poisson": round(ll(tr, 1e9), 2), "ll_2023_nb": round(ll(tr, k_hat), 2),
        "ll_2024_poisson": round(ll(te, 1e9), 2), "ll_2024_nb": round(ll(te, k_hat), 2),
        "conformal": {"alpha": alpha, "qhat": round(float(qhat), 6),
                      "tam_medio": round(float(set_size.mean()), 1),
                      "cobertura_2024": round(float(cov), 3),
                      "n_fuegos_region_2024": int(len(hits))},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
