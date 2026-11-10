# Experimentation log (2026-10-30/31)

Improvement cycle grounded in a literature review. Every experiment with its
result and decision. Full numbers in
[`reports/FULLGRID_REGIONAL.md`](../reports/FULLGRID_REGIONAL.md).

## Adopted ✅

| Experiment | Result | Source |
|---|---|---|
| **Isotonic calibration** (aux ≤2021, fitted on 2022) | Brier 0.0118→0.0065 (−45%), near-perfect reliability | Phelps & Woolford 2021 |
| **Regional Poisson aggregation + recalibration** | basis of the regional forecast | ours |
| **Stage 2 GLM + Moran Eigenvector Maps** | regional AUC 0.755 (vs 0.693 iso, 0.719 raw) · recall@10% region-days 39.4% (vs 27.3%) · α selected by CV within 2023 | "XGBoost meets INLA" arXiv 2508.09896 |
| **NGR-lite forecast correction** | forecast FWI RMSE −22% (3.03→2.37) · forecast cost −3pp→−1pp AUC · **removes the operational leakage** | Bohlmann & Laine 2024 |

## Refuted ❌ (all with metrics)

| Hypothesis | Result |
|---|---|
| Stratified negative sampling (hard negatives: doubled near-space + historical-ignition sites on FWI≥p90 days; Pourmohamad 2025) | **Hurts**: AUC 0.897→0.881, more conservative, less recall. Reverted; tiers behind `ARDENTIA_HARDNEG=1` |
| NB frailty (≈ gamma copula of the weather factor) | k=6.4 in train but **Poisson wins out-of-sample** (2024 log-lik) |
| Regional conformal prediction (α=0.10) | Honest but revealing: a 90% guarantee requires flagging **87.7% of the territory** — quantifies the signal ceiling |
| 14-day temporal sequences (LSTM 2×64 / flat / summaries) | **No gain** (LSTM 0.870 · tabular+summaries 0.864 vs tabular 0.897): FWI indices are already temporal integrators (DC/DMC accumulate months) — consistent with Di Giuseppe 2025 |
| Cause-separated models | Combined **−3.6pp** (0.861 vs 0.897): with 76% human ignitions the single model captures the dominant pattern |
| WUI/building density from OSM (446k buildings) | No gain: collinear with the existing road features |
| **TabPFN v2 distilled to full grid** | Sampled **AUC 0.9387 (vs 0.8878)** — but operational top-10% recall 18.0% vs 29.5%: **does not transfer to full grid** |
| **NVIDIA Kumo Tabular distilled** | Sampled **0.947 (best of all)** — operational top-10% 17.1%: **fails the same way** |

## The central lesson of the cycle

> **The sampled→full-grid gap is THE problem of this system** (demonstrated
> 3 times: the isotonic calibration is overconfident on the full grid; TabPFN
> and Kumo win in-sample and fail to transfer via distillation). Any
> improvement must be validated directly on the operational full-grid
> evaluation (82 days × 245k cells), not on the sample.

## Operational deliverables

- **Production pipeline**: NGR-corrected D-1 forecast → LGBM 48 feats →
  isotonic → Stage 2 GLM+MEM → ×N levels + λ (see [PREPROCESSING.md](PREPROCESSING.md))
- **Webapp** (`ardintra.api.mini_api`, port 8001): date selector → full
  pipeline → cell risk map + regional P(≥1) + real EGIF fires of the day (⭐)
- **Explainability (Phase 4)**: per-cell TreeSHAP in the viewer's cell card
  (up/down factors with original values and log-odds contributions) +
  day-level global importance for the bulletin. Optional local LLM writes the
  bulletin in prose (`scripts/serve_llm.sh`, any OpenAI-compatible server)
- **Video**: `reports/simulacion_2023_2panel.mp4` (2023 season with the full
  system, ×N labels and daily λ)
- Cycle artifacts: `fullgrid_eval*.json`, `fullgrid_regional*.parquet`,
  `stage2_model.npz`, `calibration.json`, `joint_calibration.json`,
  `kumo_teacher.json`, `tabpfn_results.json`, `distill_results.json`
