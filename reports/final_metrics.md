# Métricas finales (jerárquico + features completos, test 2023-24)

## Target: cualquier_ignicion (prev 0.97%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.1792 | 0.8752 | 0.484 | 0.0936 | 0.229 | 0.05 |
| regional_solo | 0.1701 | 0.8712 | 0.489 | 0.0946 | 0.232 | 0.0507 |
| fusion_rank | 0.1794 | 0.8749 | 0.494 | 0.0956 | 0.241 | 0.3251 |
| fwi_solo | 0.0167 | 0.6713 | 0.092 | 0.0179 | 0.026 | 0.3301 |

## Target: big10 (prev 0.23%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.0256 | 0.875 | 0.431 | 0.0195 | 0.197 | 0.0087 |
| regional_solo | 0.0299 | 0.8783 | 0.423 | 0.0192 | 0.226 | 0.0074 |
| fusion_rank | 0.0344 | 0.8855 | 0.46 | 0.0208 | 0.226 | 0.3263 |
| fwi_solo | 0.0053 | 0.6679 | 0.139 | 0.0063 | 0.044 | 0.3326 |
