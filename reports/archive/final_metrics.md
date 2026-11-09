# Métricas finales (jerárquico + features completos, test 2023-24)

## Target: cualquier_ignicion (prev 0.97%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.1774 | 0.8724 | 0.468 | 0.0906 | 0.238 | 0.0494 |
| regional_solo | 0.1701 | 0.8712 | 0.489 | 0.0946 | 0.232 | 0.0507 |
| fusion_rank | 0.1775 | 0.8735 | 0.491 | 0.0949 | 0.239 | 0.3251 |
| fwi_solo | 0.0167 | 0.6713 | 0.092 | 0.0179 | 0.026 | 0.3301 |

## Target: big10 (prev 0.23%)
| Modelo | PR-AUC | ROC-AUC | recall@5% | precision@5% | recall@1% | Brier |
|---|---|---|---|---|---|---|
| fino_solo | 0.0279 | 0.8777 | 0.445 | 0.0202 | 0.219 | 0.0084 |
| regional_solo | 0.0299 | 0.8783 | 0.423 | 0.0192 | 0.226 | 0.0074 |
| fusion_rank | 0.0317 | 0.8868 | 0.482 | 0.0218 | 0.204 | 0.3262 |
| fwi_solo | 0.0053 | 0.6679 | 0.139 | 0.0063 | 0.044 | 0.3326 |
