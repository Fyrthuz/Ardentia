# Bayesiano: prior climático × evidencia del modelo

## cualquier_ignicion (prev 0.97%)
| Variante | PR-AUC | Lift | ROC | Brier |
|---|---|---|---|---|
| modelo_raw_sin_calibrar | 0.1792 | 18.5× | 0.8752 | 0.05 |
| bayes_posterior | 0.1792 | 18.5× | 0.8752 | 0.0265 |
| prior_como_feature | 0.1051 | 10.9× | 0.7786 | 0.0398 |
| fwi_solo | 0.0167 | None× | None | None |

## big10 (prev 0.23%)
| Variante | PR-AUC | Lift | ROC | Brier |
|---|---|---|---|---|
| modelo_raw_sin_calibrar | 0.0256 | 11.3× | 0.875 | 0.0087 |
| bayes_posterior | 0.0256 | 11.3× | 0.875 | 0.0049 |
| prior_como_feature | 0.024 | 10.6× | 0.8277 | 0.0063 |
| fwi_solo | 0.0053 | None× | None | None |
