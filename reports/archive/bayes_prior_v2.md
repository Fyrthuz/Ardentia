# Prior v2: susceptibilidad con vecindario × evidencia dinámica

## cualquier_ignicion (prev 0.97%)
| Variante | PR-AUC | Lift | ROC | Brier |
|---|---|---|---|---|
| dinamico_solo | 0.1792 | 18.5× | 0.8752 | 0.05 |
| posterior_prior_v1_conteo | 0.1789 | 18.5× | 0.8752 | 0.0265 |
| posterior_prior_v2_susceptibilidad | 0.1408 | 14.6× | 0.8855 | 0.0475 |
| prior_v2_solo | 0.0854 | 8.8× | 0.7235 | 0.0922 |

## big10 (prev 0.23%)
| Variante | PR-AUC | Lift | ROC | Brier |
|---|---|---|---|---|
| dinamico_solo | 0.0256 | 11.3× | 0.875 | 0.0087 |
| posterior_prior_v1_conteo | 0.0256 | 11.3× | 0.875 | 0.0049 |
| posterior_prior_v2_susceptibilidad | 0.036 | 15.9× | 0.8856 | 0.0073 |
| prior_v2_solo | 0.0114 | 5.0× | 0.6956 | 0.0333 |
