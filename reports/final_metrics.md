# Métricas finales (jerárquico + features completos, test 2023-24)

## Target: cualquier_ignicion (prev 0.97%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.1731 | 0.8728 | 0.485 | 0.0939 | 0.234 | 0.058 |
| regional_solo | 0.1724 | 0.8734 | 0.477 | 0.0923 | 0.219 | 0.0589 |
| fusion_rank | 0.1752 | 0.8745 | 0.48 | 0.0929 | 0.227 | 0.3252 |
| fwi_solo | 0.0167 | 0.6713 | 0.092 | 0.0179 | 0.026 | 0.3301 |

## Target: big10 (prev 0.23%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.0255 | 0.8759 | 0.467 | 0.0212 | 0.212 | 0.0129 |
| regional_solo | 0.0278 | 0.8696 | 0.453 | 0.0205 | 0.219 | 0.012 |
| fusion_rank | 0.0284 | 0.8809 | 0.504 | 0.0228 | 0.234 | 0.3263 |
| fwi_solo | 0.0053 | 0.6679 | 0.139 | 0.0063 | 0.044 | 0.3326 |
