# Diario de experimentación (2026-10-30/31)

Ciclo de mejoras basado en revisión de literatura. Cada experimento con su
resultado y decisión. Números completos en [`reports/FULLGRID_REGIONAL.md`](../reports/FULLGRID_REGIONAL.md).

## Adoptados ✅

| Experimento | Resultado | Fuente |
|---|---|---|
| **Calibración isotónica** (aux ≤2021, ajuste 2022) | Brier 0.0118→0.0065 (−45%), reliability casi perfecta | Phelps & Woolford 2021 |
| **Agregación regional Poisson + recalibración** | base del forecast regional | ours |
| **Stage 2 GLM + Moran Eigenvector Maps** | AUC regional 0.755 (vs 0.693 iso, 0.719 cruda) · rec@10% regiones 39.4% (vs 27.3%) · selección α por CV interna 2023 | "XGBoost meets INLA" arXiv 2508.09896 |
| **NGR-lite del forecast D-1** | RMSE del FWI forecast −22% (3.03→2.37) · coste del forecast −3pp→−1pp AUC · **elimina el leakage operativo** | Bohlmann & Laine 2024 |

## Refutados ❌ (todos con métricas)

| Hipótesis | Resultado |
|---|---|
| Muestreo estratificado de negativos (hard negatives: near-space doblado + sitios históricos de ignición en días FWI≥p90; Pourmohamad 2025) | **Empeora**: AUC 0.897→0.881, más conservativo, menos recall. Revertido; tiers detrás de `ARDENTIA_HARDNEG=1` |
| Frailty NB (≈ copula gamma del factor meteorológico) | k=6.4 en train pero **el Poisson gana out-of-sample** (log-lik 2024) |
| Conformal prediction regional (α=0.10) | Honesto pero indicado: garantía del 90% exige marcar **87.7% del territorio** — cuantifica el límite del señal |
| Secuencias temporales 14 días (LSTM 2×64 / flat / sumarios) | **Sin ganancia** (LSTM 0.870 · tabular+sumarios 0.864 vs tabular 0.897): los índices FWI ya son integradores temporales (DC/DMC acumulan meses) — consistente con Di Giuseppe 2025 |
| Modelos separados por causa | Combinado **−3.6pp** (0.861 vs 0.897): con 76% humanas, el modelo único captura el patrón dominante |
| WUI/densidad de edificios OSM (446k edificios) | Sin ganancia: colineal con las features viarias existentes |
| **TabPFN v2 destilado a full-grid** | En muestra **AUC 0.9387 (vs 0.8878)** — pero operativo top-10% 18.0% vs 29.5%: **no transfiere a full-grid** |
| **NVIDIA Kumo Tabular destilado** | En muestra **0.947 (mejor de todos)** — operativo top-10% 17.1%: **falla igual** |

## La lección central del ciclo

> **La brecha muestreado→full-grid es EL problema del sistema** (evidenciado
> 3 veces: calibración isotónica sobreenconfiada en full-grid; TabPFN y Kumo
> ganan en muestra y no transfieren vía destilación). Toda mejora debe
> validarse directamente en la evaluación operativa full-grid (82 días ×
> 245k celdas), no en la muestra.

## Entregables operativos

- **Pipeline de producción**: forecast D-1 corregido → LGBM 48 feats →
  isotónica → Stage 2 GLM+MEM → niveles ×N + λ (ver [PREPROCESSING.md](PREPROCESSING.md))
- **Webapp** (`ardintra.api.mini_api`, puerto 8001): selector de fecha →
  pipeline completo → mapa de celdas + P(≥1) regional + fuegos reales del día (⭐)
- **Vídeo**: `reports/simulacion_2023_2panel.mp4` (temporada 2023 con el
  sistema completo, etiquetas ×N y λ diario)
- Artefactos del ciclo: `fullgrid_eval*.json`, `fullgrid_regional*.parquet`,
  `stage2_model.npz`, `calibration.json`, `joint_calibration.json`,
  `kumo_teacher.json`, `tabpfn_results.json`, `distill_results.json`
