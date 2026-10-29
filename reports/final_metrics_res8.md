# Métricas finales (jerárquico + features completos, test 2023-24)

## Target: cualquier_ignicion (prev 3.63%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.3034 | 0.8532 | 0.391 | 0.2842 | 0.166 | 0.0896 |
| regional_solo | 0.3138 | 0.8545 | 0.401 | 0.2914 | 0.18 | 0.0891 |
| fusion_rank | 0.3116 | 0.8549 | 0.401 | 0.2914 | 0.166 | 0.308 |
| fwi_solo | 0.07 | 0.6903 | 0.095 | 0.0689 | 0.027 | 0.32 |

## Target: big10 (prev 0.82%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.1005 | 0.8698 | 0.511 | 0.0834 | 0.17 | 0.0193 |
| regional_solo | 0.1192 | 0.8711 | 0.519 | 0.0846 | 0.185 | 0.0193 |
| fusion_rank | 0.1105 | 0.8726 | 0.511 | 0.0834 | 0.193 | 0.3255 |
| fwi_solo | 0.0198 | 0.6818 | 0.141 | 0.023 | 0.037 | 0.3304 |
