# Métricas finales (jerárquico + features completos, test 2023-24)

## Target: cualquier_ignicion (prev 0.97%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.1829 | 0.8765 | 0.516 | 0.0999 | 0.243 | 0.0496 |
| regional_solo | 0.1701 | 0.8712 | 0.489 | 0.0946 | 0.232 | 0.0507 |
| fusion_rank | 0.1795 | 0.8756 | 0.503 | 0.0972 | 0.248 | 0.325 |
| fwi_solo | 0.0167 | 0.6713 | 0.092 | 0.0179 | 0.026 | 0.3301 |

## Target: big10 (prev 0.23%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.0278 | 0.8732 | 0.423 | 0.0192 | 0.226 | 0.0087 |
| regional_solo | 0.0299 | 0.8783 | 0.423 | 0.0192 | 0.226 | 0.0074 |
| fusion_rank | 0.032 | 0.8847 | 0.474 | 0.0215 | 0.212 | 0.3264 |
| fwi_solo | 0.0053 | 0.6679 | 0.139 | 0.0063 | 0.044 | 0.3326 |
