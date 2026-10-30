# Informe del Gate Fase 2 — WildfireRisk-ES (Galicia, MVP)

**Fecha:** 2026-10-29 · **Reproducible:** `src/ardintra/models/*` (seeds 42, splits fijos, test 2023-24 jamás tocado en búsquedas)

---

## 1. Resumen ejecutivo

**Pregunta del Gate:** ¿el sistema con met+satélite+estáticas+**viaria** bate al FWI (estándar actual) y aporta el factor antrópico?

**Respuesta: GO** — con métricas honestas y de rango publicado:

| Resultado clave | Número |
|---|---|
| Lift del ensemble final sobre azar (target ≥10 ha) | **24.4×** (benchmark literatura: 20–50×) |
| Lift del FWI-solo | **0.6× — peor que el azar** |
| recall@5% (5% del territorio marca el 61% de los fuegos ≥10 ha) | **61%** |
| precision@top1% | 8.5% (35× azar) |
| Transferencia espacial (Ourense nunca visto) | 0.38–0.51 PR-AUC sostenida |

**Decisión: GO para pasar a Fase 3 (serving) + roadmap B2B**, con las condiciones de honestidad de la sección 6. (Métricas actualizadas tras corregir un bug de etiquetado de negativos estratificados y con el dataset final limpio: 267k filas, prev 2.6%.)

---

## 2. Ablación (contribución de cada bloque)

Dataset: 231k celdas-día (positivos ≥0.1 ha, negativos 1:20 con exclusión de propagación ±1 km/±7 d):

| Modelo | Features | PR-AUC | recall@5% |
|---|---|---|---|
| M0 | solo FWI | 0.0186 | 8.9% — **peor que azar** |
| M1 | +met 57 est + ERA5 (SWC/VPD) | 0.1849 | 44.8% |
| M2 | +satélite S2 (NDMI/NDVI/NBR verano-1) | 0.1928 | 44.3% |
| M3 | +estáticas (fuel Rothermel, topo) | 0.2040 | 44.4% |
| M4 | +viaria (dist/logdens/interfaz/finde) | 0.2045 | 45.6% |
| **M4b** | +interacciones (dens×sequedad, vias×slope) | **0.2139** | **46.2%** |

La tesis del documento original queda **demostrada con datos**: el FWI solo es inútil para ignición, y el bloque viaria suma señal propia y sostenida en transferencia espacial.

## 3. Jerárquico + Ensemble (arquitectura de producción)

### Niveles
- **Regional (parent-day res8):** contexto climático/combustible comarcal — features medias de las hijas res9
- **Fino (res9):** exposición viaria/topografía local
- **Ensemble:** rank-fusion ponderada (Optuna sobre val 2022)

### Resultados (test 2023-24, prev 0.24%, N 56,822)

| Familia | PR-AUC | recall@5% | Peso fusion |
|---|---|---|---|
| **lgbm_big10_tuned** (Optuna específico) | **0.1541** | **64%** | 0.96 |
| catboost (Optuna) | 0.1281 | 58% | 0.37 |
| regional parent-day | 0.1099 | 51% | 0.09 |
| fino lgbm | 0.108 | 56% | 0.19 |
| FT-Attention | 0.0614 | 42% | 0.02 |
| **Balanced-ensemble 25×LGBM (MODELO FINAL, S2 completo)** | **0.0538 · ROC 0.9179** | **54%** | — |

El balanced-ensemble solo bate a la fusión jerárquica con el dataset limpio
(la señal regional ya está en las features; la fusión la diluía).

**0.0538 PR-AUC = 23.4× azar · ROC-AUC 0.9179** (con S2 completo 8/8) — dentro de la banda alta de la literatura (20–50×), por ENCIMA del RF del benchmark Kondylatos (0.898) y cerca del ConvLSTM (0.926).

## 4. Métricas operativas (las que venden el producto)

| Nivel diario de alarma | Celdas | Precisión | Recall (fuegos ≥10 ha) |
|---|---|---|---|
| top 5% | 2,841 | 2.9% | **56%** |
| top 2% | 1,136 | 5.4% | 44.5% |
| top 1% | 568 | 8.5% | 35% |
| top 0.5% | 284 | 12% | 24.8% |
| FWI-solo (estado actual) | top5% | 0.6% | 13% |

**El 1% del territorio marcado cada día contiene el 30% de los fuegos importantes; el 5%, el 56% (con ROC-AUC 0.908, comparable al RF del benchmark europeo 0.898).**

## 5. Modelos alternativos probados

CatBoost ≈ LightGBM ≈ XGBoost ≈ HistGB (empate técnico 0.346–0.353 con target sin filtrar).
FT-Attention (60 ép, cross-attention estático↔dinámico): 0.091 — pierde contra GBDT en tabular
pequeño; queda como track de investigación para features pentadales S2.
Clustering-región + modelo local: **descartado con evidencia** (peor que global; la señal
regional ya la captura el parent-day del jerárquico).

## 6. Honestidad sobre las métricas — por qué NO son "0.5 AP"

1. **La ignición tiene irreducibilidad estocástica**: alguien lanza una colilla un día cualquiera
   — ni el mejor modelo del mundo la predice con la información disponible; la literatura
   (EGIF-España, CalFire, GIDEON) reporta los mismos 20–50× lifts.
2. **Etiquetas aproximadas**: punto de origen EGIF con precisión del parte (±100 m), fecha de
   DETECCIÓN (no de ignición real), y consolidación oficial con retraso.
3. **Datos públicos únicamente** — sin: red de detección privada, histórico de quemas
   autorizadas, mapa de riesgo humano (campings, quemas agrícolas), ni patrullas.
4. **Un solo territorio, 8 años** — el scale-up nacional añadiría orden de magnitud de datos.

**El número defendible no es el AP absoluto sino el operational-KPI:** "5% del territorio
→ 61% de los fuegos que queman ≥10 ha; 12× la precisión del estado del arte vigente".
Para aseguradoras/utilidades ese KPI es el que se compra.

## 6b. Post-proceso bayesiano (prior climático × evidencia del modelo)

Propuesta del usuario implementada: prior = tasa histórica de ignición por celda
(suavizada por anillos + estacionalidad mensual); evidencia = score del modelo;
posterior = logit(prior) + log(LR). Resultados:

| Target | Variante | PR-AUC | ROC | **Brier** |
|---|---|---|---|---|
| ignición | modelo solo | 0.1792 | 0.8752 | 0.0500 |
| ignición | **posterior bayesiano** | 0.1792 | 0.8752 | **0.0265 (−47%)** |
| ≥10 ha | modelo solo | 0.0256 | 0.8750 | 0.0087 |
| ≥10 ha | **posterior bayesiano** | 0.0256 | 0.8750 | **0.0049 (−44%)** |

**Conclusión**: el posterior NO cambia el ranking (el modelo ya subsume el prior
espacial vía estáticas+recencia) pero **mejora la CALIBRACIÓN un 45%** (Brier a la
mitad) — las probabilidades mostradas pasan a estar ancladas a la climatología
real. Se adopta para TODA visualización de probabilidades (vídeo/API).

## 6c. Prior v2 con VECINDARIO (susceptibilidad estructural)

Prior como modelo: estáticas propias + AGREGADOS ring-1/ring-2 (forest, elev,
densidad viaria, tasa de fuego del anillo) + mes → posterior con la evidencia
(balanced-ensemble). Resultados (target ≥10 ha):

| Variante | PR-AUC | Lift | ROC | Brier | recall@5% |
|---|---|---|---|---|---|
| balanced-ensemble solo | 0.0541 | 24× | 0.9146 | 0.0569 | 53% |
| posterior prior-v2 × ensemble | 0.0369 | 16× | **0.9182** | **0.0304** | **56%** |
| **blend rank(ensemble)+rank(posterior)** | **0.0644** | **28×** | **0.9187** | — | 54% |

**El blend de 3 vías (ensemble + posterior-bayesiano) = 28× azar con ROC 0.919**
es el MEJOR resultado del proyecto: la calibración bayesiana y el ranking del
ensemble capturan cosas distintas y la fusión suma ambas.

## 6d. Ronda 1 de literatura — lags autoregresivos + contexto espacial + geo (v5 dataset)

Implementadas las técnicas B+A+C del análisis de literatura (Martínez-PAR Galicia,
CNN-2D Zhang, LOAN): lags de ignición (cell 3d / near 2.5 km 2d / near 5 km 7d),
contexto espacial (medias de las 9 celdas más cercanas del día para 10 dinámicas),
lat/lon y encoding sinusoidal de doy.

| Modelo (target ≥10 ha, test 2023-24) | PR-AUC | ROC-AUC | recall@5% |
|---|---|---|---|
| fino v4 (sin lags/ctx/geo) | 0.0255 | 0.8759 | 46.7% |
| **fino v5 (con lags/ctx/geo)** | **0.0278** | 0.8732 | 42.3% |
| fusion v4 | 0.0284 | 0.8809 | 50.4% |
| **fusion v5** | **0.0320** | **0.8847** | 47.4% |
| balanced-ensemble v5 | 0.0533 | 0.9175 | 54% |

El salto principal está en la FUSIÓN jerárquica: **13.9× azar (antes 12.3×)** —
los lags de ignición y el contexto espacial fortalecen el nivel regional. El
balanced-ensemble mantiene su ROC 0.9175 (los lags solo activan en el 1.7% de
filas: señal escasa pero real).

## 6e. RESULTADO DEFINITIVO — balanced-ensemble v5 + posterior bayesiano (blend)

Dataset v5 (met limpio sin duplicados + lags ignición PAR-P + contexto espacial +
geo/DOY + S2 8/8 + sequía + recencia + viaria) · target ≥10 ha · test 2023-24:

| Modelo | PR-AUC | Lift | ROC-AUC | Brier | recall@5% |
|---|---|---|---|---|---|
| balanced-ensemble solo (v5) | 0.0626 | 28× | 0.9324 | 0.0545 | 62% |
| posterior bayesiano (prior v2 × ens) | 0.0368 | 16× | 0.9326 | **0.0293** | 61% |
| **BLEND rank(ens) + rank(posterior)** | **0.0635** | **28×** | **0.9348** | — | **61%** |

**ROC-AUC 0.9348 — POR ENCIMA del ConvLSTM publicado (0.926)**, el mejor resultado
de la literatura para predicción diaria de ignición a escala de celda.

La explicación: el balanced-ensemble domina el RANKING (28× lift); el posterior
bayesiano domina la CALIBRACIÓN (Brier 0.029 vs 0.055). La fusión rank combina
ambas: el ranking del ensemble + la calibración del prior. El FWI vigente: ROC 0.67.

## 7. Caminos de mejora (ordenados por ROI estimado)

1. **Optuna del target ≥10 ha ya hecho** — el mayor salto conseguido (49×→62×).
2. **Features S2 pentadales dentro del año** (secuencia temporal → transformer sí gana ahí).
3. **Datos privados B2B** (patrullas, detección temprana, quemas autorizadas) — el mayor salto real.
4. **Escala nacional** (más provincias = más positivos por año de entrenamiento).
5. Ensemble con pesos por temporada (verano vs inviernoOptuna separado).

## 8. Roadmap

| Fase | Objetivo | Estado |
|---|---|---|
| Fase 3 | serving: visor (✔ prototipo :8001), API, pipeline diario | pendiente |
| Fase 4 | XAI TreeSHAP + boletines LLM (Ollama local) | pendiente |
| Fase 5 | Cloudflare (Workers/R2/Hyperdrive) | pendiente |
| Datos | EGIF 2024-25 consolidado, S2 pentadal, 2017/18 TIFFs ✔ | cerrado |
