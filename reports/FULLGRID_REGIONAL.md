# Full-grid operational evaluation and probabilistic regional forecast

**Cycle 2026-10-30/31** — the master document of the experimentation cycle
adopted by the [main README](../README.md) and
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md). Reports from the previous
cycle (2026-10-29, Phase-2 gate) are in `archive/` (Spanish).

**Date:** 2026-10-30 · Model: LGBM (600 trees, lr 0.05, 48 features) · Target: ignition ≥10 ha

## 1. Full-grid operational evaluation (82 test days with ≥10 ha fires, 137 fires)

For the first time **all 245,172 cells** are predicted every test day (not
the 15-cell sample), emulating real operational use: every morning the top-K%
of Galicia by risk is flagged.

| Daily effort | Mean recall | Median recall |
|---|---|---|
| top 1% (2,451 cells) | 3.0% | 0% |
| top 5% (12,258 cells) | 14.2% | 0% |
| top 10% (24,517 cells) | **31.2%** | 0% |

**Honest reading:** flagging 10% of the territory every day captures ~31% of
the major fires (daily mean; the median is 0 because on most days no fire
falls in the top). It is a very hard problem: 137 fires spread over 82 days
on 245k cells.

## 2. Probability calibration (isotonic)

Auxiliary model trained ≤2021, isotonic fitted on 2022.

- **Brier: 0.0118 → 0.0065** (−45%)
- Near-perfect reliability on the 8 top bins (e.g.: calibrated p 3.16% vs
  real 3.08% on the top decile; 0.87% vs 0.84% at bin 0.0043).

With this the phrase *"this cell has a 3% probability of a ≥10 ha fire
today"* is honest.

## 3. Regional forecast (Poisson aggregation + recalibration)

Aggregation over res7 parent hexagons (5.2 km): λ = Σp_i → P(≥1) = 1−e^(−λ).

**Detected problem:** the raw P(≥1) is massively overconfident (regions with
predicted P(≥1) 50-100% had a fire 0.05% of the time). Causes: (a) the cell
isotonic was fitted with the 15-cells/day sample and the full grid
extrapolates outside its range (clipping to the max), (b) cells share the
same weather → perfect correlation that violates Poisson independence.

**Solution:** a second isotonic at region level, fitted on 2023 and evaluated
on 2024 (honest temporal split, 154,170 region-days, 33 with fire):

| Decile of p_reg | median p | real frequency |
|---|---|---|
| 1st | 0.0000 | 0.0000 |
| 5th | 0.0002 | 0.0001 |
| 8th | 0.0004 | 0.0004 |
| 10th (top) | 0.0005 | **0.0008** |

- **Regional AUC: 0.728** (predicting ≥1 fire ≥10 ha in the region that day)
- Top 10% region-days → 27.3% of region-fires · top 20% → 42.4%
- Monotone tendency in the upper deciles; absolute calibration compressed
  (rare events: base rate 0.02-0.03% per region-day)

## 4. Risk communication

Since honest probabilities are tiny, the final product communicates:

- **Risk level** by percentiles (green/yellow/orange/red) — cells and regions
- **Base-rate multiple** (×N): "this region today carries 85× the average risk"
- **Expected number of ≥10 ha fires in Galicia** (recalibrated λ) in the video footer

We never claim "there will definitely be a fire": it is a stochastic process,
we emit calibrated probability + an ordinal level (AEMET analogy).

## 5. Spatial dependence and guarantees (NB frailty + conformal)

`src/ardintra/models/joint_calibration.py` — two experiments on the regional
aggregation:

**a) Gamma frailty (≈ common-factor copula):** fires of the same day share
the weather (same FWI) → overdispersion relative to Poisson. Model
N_d ~ NB(mean λ_d, frailty k).

- k fitted on 2023: **6.4** (real overdispersion in train: days with 8-9 fires)
- **Out-of-sample (2024) Poisson wins**: log-lik −32.9 (Poisson) vs −34.5 (NB)
- With 55 training days, the extra parameter does not pay off. Poisson
  independence is adequate at the Galicia-day scale; overdispersion is not
  robustly estimable.

**b) Regional split conformal (α=0.10, fitted 2023 → evaluated 2024):**

- To guarantee ≥90% marginal coverage the conformal set must flag **87.7% of
  the territory** (mean 5,008 regions/day)
- Comparison: fixed top-20% by p_reg → 78.8% coverage with only 20% of the territory
- **Reading:** conformal quantifies the price of a guarantee — with AUC 0.73
  signal, a 90% guarantee demands flagging nearly everything. Useful as an
  honest system ceiling.

**Documented limitation:** the regional parquet only contains the 82 days
WITH fires (the 649 days without ≥10 ha fires were not evaluated), so
unconditional P(N≥1) calibration cannot be validated without reprocessing
the no-fire days.

## 6. Two-panel video (updated)

`reports/simulacion_2023_2panel.mp4` — left: res9 cells by calibrated-p
percentile; right: res7 regions by recalibrated P(≥1) with ×N labels on the
day's top-3; footer with expected Galicia λ and real fires (⭐ ≥10 ha, 🔵 <10 ha).

## Artifacts

- `reports/fullgrid_eval.json` — operational recall per day
- `reports/fullgrid_regional.parquet` — λ, raw P(≥1) and observed fires per region-day
- `reports/fullgrid_regional_calibrado.parquet` — recalibrated P(≥1) (2024 eval)
- `reports/regional_reliability_2024.csv` — regional reliability
- `reports/calibration.json`, `reports/reliability_diagram.csv` — cell calibration
- `reports/joint_calibration.json` — NB frailty + regional conformal

## 7. Phase 1 — improvement experiments (2026-10-31)

### 7.1 Stratified negative sampling — REFUTED

Hypothesis (Pourmohamad 2025): hard negatives (doubled near-space + historical
ignition sites on FWI≥p90 days) would fix the full-grid overconfidence.

| Config | AUC | AP | operational recall@10% |
|---|---|---|---|
| V1 original sampling | **0.897** | **0.036** | **31.2%** |
| V2 stratified | 0.881 | 0.031 | 27.2% |
| V2 without density features | 0.878 | 0.034 | — |

- Stratified sampling makes the model WORSE (more conservative, −1.9pp AUC)
- Regional overconfidence persists → it is structural (sample→full-grid
  calibration transfer + correlated cell sums), not a sampling problem
- The 2nd regional isotonic remains the correct pragmatic fix
- **Decision:** reverted to v1; new tiers behind `ARDENTIA_HARDNEG=1`
- Cause-density features: train/test shift (0→1.97), no gain — excluded

### 7.2 Ignition cause — propagated and available

- EGIF `idcausa` propagated to the dataset: 6,464 positives (92%) — 76%
  human (3xx+4xx), 3% lightning (1xx), rest unknown/re-ignition
- Columns `cause`, `es_humana`, `es_rayo` available for analysis and
  per-cause models

### 7.3 D-1 forecast — VIABLE with a small quantified cost

**CORRECTION NOTE:** the first version of this experiment used GFS Previous
Runs data (coverage only from 2024-01), which left the 2023 weather features
at zero → invalid results, retracted. The definitive version uses the
Historical Forecast API (HRES seamless, full 2023 coverage, ~24h lead) with
per-station NGR-lite bias correction (fitted 2023, verified on 2024: forecast
FWI RMSE 3.03→2.37, −22%).

| Metric | obs (V1) | NGR-corrected D-1 forecast (V3b) | diff |
|---|---|---|---|
| Aggregate AUC (sample) | 0.897 | 0.887 | −1.0pp |
| operational top-5 recall | 14.2% | 13.2% | −0.3 SE (noise) |
| operational top-10 recall | 31.2% | 29.5% | −0.4 SE (noise) |

- **Conclusion:** the system can operate on D-1 forecasts: the cost is −1pp
  aggregate AUC in-sample and undetectable in operational @K recall
  → removes the operational leakage (day-D weather is not available in the morning)
- Artifacts: `src/ardintra/features/build_met_fc.py` (NGR-lite),
  `data/raw/openmeteo/met_fc.parquet`, `reports/fullgrid_eval_v3.json`,
  flag `ARDENTIA_MET_FC=1` in fullgrid_eval

## 8. Stage 2 — hierarchical spatial model (GLM + Moran Eigenvector Maps)

Pure-python replica of the "XGBoost meets INLA" approach (arXiv 2508.09896), no R:

    log μ_{r,d} = α + β·log(1+λ_raw_{r,d}) + Σ_k γ_k·MEM_k(r)

- MEM: Moran eigenvectors of the res7 hex adjacency (5,710 regions, a discrete
  basis approximating the SPDE Matérn field; Griffith 1996)
- β recalibrates the Stage-1 intensity (fitted: 0.61 — corrects the overconfidence)
- u_r = MEM·γ: persistent residual spatial pattern (clustering not explained
  by dynamic features)
- Fit: PoissonRegressor (ridge), alpha=1e-6 **selected by CV within 2023
  (fit Jan-Jun → val Jul-Dec)** — 2024 never touched

### 2024 validation (temporal split, 154,170 region-days, 33 with fire)

| Model | AUC | recall@10% region-days |
|---|---|---|
| Raw Stage-1 λ | 0.719 | 30.3% |
| Regional isotonic (production) | 0.693 | 27.3% |
| **Stage 2 (GLM+MEM)** | **0.755** | **39.4%** |

Reliability by deciles: monotone and well calibrated (top decile: predicted
0.00052 vs real 0.00071 — 1.4× underconfidence, acceptable for rare events).

Alpha sensitivity: 1e-4 → 0.720 (spatial collapsed), 1e-6 → 0.755 (optimal,
selected by CV), 1e-8 → 0.717 (overfit). The 2023 internal CV selects 1e-6
independently → the gain is not a selection artifact.

**Conclusion:** Stage 2 genuinely improves the regional forecast (+3.6pp AUC,
+12pp recall@10% over the production baseline) with honest calibration. The
production isotonic recalibration is replaced by the GLM+MEM.

Artifacts: `src/ardintra/models/stage2_lgcp.py`, `reports/stage2_lgcp.json`,
`reports/stage2_reliability.csv`, `reports/stage2_predictions_2024.parquet`.

## 9. Stage 2 integrated into the video

`reports/simulacion_2023_2panel.mp4` regenerated: the right panel uses μ from
the GLM+MEM (`reports/stage2_model.npz`) — honest regional P(≥1), ×N labels
(risk vs 0.028% base rate) and daily λ = Σμ (expected number of fires).
Example Oct 12 2023: ×6-×7 on the top-3 (previously ×38 with the
overconfident model).

## 10. Phase 3 — temporal sequences (14 days) — NO GAIN

Explicit temporal context (14-day weather/FWI window per cell, IDW over 3
stations; `data/processed/features/v1/sequences.npz`, 267k × 14×8):

| Model | AUC (test) | AP |
|---|---|---|
| Day-D tabular LGBM (48 feats, obs) | **0.897** | **0.036** |
| Day-D tabular LGBM (48 feats, forecast test) | 0.867 | 0.019 |
| LGBM + 14d summaries | 0.864 | 0.030 |
| LGBM + flattened sequence (112) | 0.811 | 0.012 |
| LSTM 2×64 (weather sequence only) | 0.870 | 0.017 |

- **Weather sequences add no value**: the FWI indices (DC/DMC are accumulators
  with months of memory) already encode the relevant temporal history
- Consistent with Di Giuseppe 2025 (ECMWF): data quality > architecture
- Note: LSTM vs tabular is not apples-to-apples (LSTM has no static/spatial
  features); the fair test (tabular+summaries) confirms: −0.3pp AUC
- Additional finding (corrected after detecting invalid data): the NGR-corrected
  D-1 forecast costs −1pp aggregate AUC (0.897→0.887); operational @K recall
  stays within noise (−0.3/−0.4 SE)

## Final system state

Complete and validated operational pipeline:
1. D-1 forecast (NGR) → 2. LGBM 245k cells → 3. cell isotonic calibration →
4. Stage 2 GLM+MEM regional (AUC 0.755, recall@10% 39.4%) → 5. video/×N+λ levels

## 11. Literature round 2 (NGR, cause, WUI)

### 11.1 Forecast bias correction (NGR-lite) — ADOPTED ✅

Bohlmann & Laine 2024 (NHESS): statistical calibration of FWI forecasts.
Implementation: linear per-station correction (tmax/rh/wind) + multiplicative
precipitation factor, fitted with 2023 obs-forecast pairs, applied to the
whole forecast; the FWI chain recomputed over the corrected weather.

- Forecast FWI RMSE 2024 (holdout): 3.03 → **2.37 (−22%)** · bias +1.27 → +0.91
- Forecast cost in aggregate AUC: −3pp (uncorrected, invalid initial
  measurement) → **−1pp corrected** (0.897 → 0.887)
- Operational @K recall: within noise (−0.3/−0.4 SE)
- `src/ardintra/features/build_met_fc.py`

### 11.2 Cause-separated models — REFUTED ❌

Gelabert 2025 / Shmuel 2025 suggest per-cause models. With our data
(862 human ≥10 ha in train, 39 lightning, 458 other):

| Model | AUC |
|---|---|
| Single (reference) | **0.897** |
| Human-only | 0.867 |
| Natural+other-only | 0.704 |
| Combined 1−(1−p_h)(1−p_n) | 0.861 |

Splitting weakens the signal (fewer positives per model) and the combined
model loses 3.6pp. With 76% human ignitions the single model already captures
the dominant pattern. Cause stays as an analytical column, not a model split.

### 11.3 WUI / building density (OSM) — REFUTED ❌

Gelabert 2025: distance to WUI = main human driver. Population-free proxy:
446,494 OSM buildings → 1 km density + distance to nearest building
(`data/processed/h3_cells/res9_wui.parquet`).

- AUC: 0.8874 (without) vs 0.8857 (with) — no gain
- Minimal importance: we already have dist_to_road, road_density, road_class
  and road_fuel_interface — the human footprint is covered; the proxy is collinear

## 12. TabPFN v2 and distillation — in-sample gain, no grid transfer

### 12.1 TabPFN v2 (pretrained transformer, Nature 2025) — best in-sample model

Subsampling ensembles (10×10k, all positives), evaluated with 137 test
positives + 20k negatives (unbiased AUC):

| Model | AUC | AP |
|---|---|---|
| **TabPFN v2 (teacher)** | **0.9387** | **0.165** |
| Blend student+labels | 0.9320 | 0.138 |
| Distilled student (LGBM) | 0.9303 | 0.144 |
| LGBM labels (reference) | 0.8878 | 0.092 |

+5.1pp AUC and ×1.8 AP over LGBM on the sampled distribution.

### 12.2 Distillation to full grid — DOES NOT TRANSFER ❌

Direct TabPFN full-grid inference is infeasible (~30+ min/day).
Distillation: 420k full-grid train rows scored by the teacher
(`distill_set.parquet`) → LGBM student (retains 0.9303 in-sample).

But the full-grid operational evaluation (82 days) of the blend degrades:
top-10% 18.0% vs 29.5% (LGBM forecast) · regional AUC 0.593 · recall@10%
region 13.1%.

**Root cause:** the teacher was only calibrated/evaluated on the SAMPLED
distribution (15 cells/day); its ranking does not transfer to the full-grid
space — the same domain failure as the isotonic overconfidence. The
distillation faithfully copied the teacher, including its OOD weakness.

**Project lesson (3 pieces of evidence): the sample→full-grid gap is THE
central problem of this system.** Any improvement must be validated directly
on full grid, not on the sample.

### 12.3 Decision

Production pipeline = the operationally best-validated one:
NGR-corrected D-1 forecast → LGBM 48 feats → isotonic → Stage 2 GLM+MEM.

## 13. NVIDIA Kumo Tabular — second replication of the finding

Teacher 2 = Kumo Tabular large (NVIDIA, ICL, local HF weights, 215M params).

### On the sampled distribution — better than everything

| Model | AUC | AP |
|---|---|---|
| **Kumo teacher** | **0.947** | — |
| TabPFN teacher | 0.9387 | 0.165 |
| LGBM labels | 0.8887 | 0.092 |

### Operational full grid (Kumo-distilled student, 82 days) — fails the same way

| Metric | Kumo distilled | TabPFN distilled | LGBM forecast (V3b) |
|---|---|---|---|
| top-5 recall | 12.3% | 6.6% | **13.2%** |
| top-10 recall | 17.1% | 18.0% | **29.5%** |
| regional AUC | 0.623 | 0.593 | — |
| recall@10% region | 16.9% | 13.1% | — |

**Finding replicated with 2 independent foundation models:** winning on the
sampled distribution (more AUC than LGBM) implies being MORE committed to
that distribution → worse operational transfer. The classic LGBM, less
expressive, extrapolates better to the full grid.

Artifacts: `src/ardintra/models/kumo_teacher.py`, `reports/kumo_teacher.json`,
`reports/fullgrid_eval_kumo.json`, `reports/fullgrid_regional_kumo.parquet`,
`models/student_kumo.joblib`.

## 14. Webapp explanation semantics (viewer)

An important user-facing correction: painting "extreme" cells by
within-day percentile implies an absolute probability that does not exist at
105 m (cell P(≥10 ha) ≈ 10⁻⁴ even on the worst historical day — an honest
absolute map would always be green). The viewer therefore uses
**within-day risk concentration** language (Bottom/Low/Medium/High/Maximum,
FWI-gated palette: red only when the day's median FWI ≥15) and the magnitude
lives at REGION level (P(≥1), ×N) and in λ. See `ardintra/api/day_predictor.py`.
