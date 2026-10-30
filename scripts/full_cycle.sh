#!/usr/bin/env bash
# Cadena completa: reentrenar todas las resoluciones + gráficas + vídeos
set -u
cd /home/fyrthuz/Desktop/Ardentia
set -a; source .env; set +a

for R in 7 6 5 4; do
  echo "[$(date '+%H:%M')] rebuild dataset res$R"
  ARDENTIA_RES=$R ARDENTIA_MIN_HA=0.1 .venv/bin/python -u -m ardintra.features.build_dataset > /tmp/opencode/rebuild_res$R.log 2>&1
done

echo "[$(date '+%H:%M')] balanced-ensemble en las 6 resoluciones"
for R in 9 8 7 6 5 4; do
  ARDENTIA_RES=$R .venv/bin/python -m ardintra.models.balanced_ensemble > /tmp/opencode/be_$R.log 2>&1
done

echo "[$(date '+%H:%M')] ablaciones res9 + res8"
.venv/bin/python -m ardintra.models.ablation_run > /tmp/opencode/abl9.log 2>&1
ARDENTIA_RES=8 .venv/bin/python -m ardintra.models.ablation_run > /tmp/opencode/abl8.log 2>&1

echo "[$(date '+%H:%M')] métricas finales res9 + res8"
.venv/bin/python -m ardintra.models.final_metrics > /tmp/opencode/fm9.log 2>&1
ARDENTIA_RES=8 .venv/bin/python -m ardintra.models.final_metrics > /tmp/opencode/fm8.log 2>&1

echo "[$(date '+%H:%M')] gráficas"
.venv/bin/python -m ardintra.viz.metric_charts > /tmp/opencode/charts.log 2>&1

echo "[$(date '+%H:%M')] vídeos de las 6 resoluciones"
rm -rf data/processed/video_frames*
for R in 9 8 7 6 5 4; do
  ARDENTIA_RES=$R .venv/bin/python -u -m ardintra.viz.day_simulation 2023 > /tmp/opencode/video_$R.log 2>&1
done

echo "[$(date '+%H:%M')] FULL-CYCLE-DONE"
