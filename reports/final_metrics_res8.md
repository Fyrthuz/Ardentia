# Métricas finales (jerárquico + features completos, test 2023-24)

## Target: cualquier_ignicion (prev 3.63%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.3139 | 0.8556 | 0.394 | 0.2866 | 0.163 | 0.0783 |
| regional_solo | 0.3127 | 0.8558 | 0.408 | 0.2963 | 0.165 | 0.0778 |
| fusion_rank | 0.3175 | 0.857 | 0.401 | 0.2914 | 0.163 | 0.3077 |
| fwi_solo | 0.07 | 0.6903 | 0.095 | 0.0689 | 0.027 | 0.32 |

## Target: big10 (prev 0.82%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.1114 | 0.8738 | 0.437 | 0.0713 | 0.2 | 0.0151 |
| regional_solo | 0.1019 | 0.8617 | 0.444 | 0.0726 | 0.178 | 0.015 |
| fusion_rank | 0.1107 | 0.8697 | 0.444 | 0.0726 | 0.193 | 0.3257 |
| fwi_solo | 0.0198 | 0.6818 | 0.141 | 0.023 | 0.037 | 0.3304 |
