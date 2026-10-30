#!/usr/bin/env bash
# Cadena final: reentrenar con S2 completo → evaluar → vídeos con el blend → docs
set -u
cd /home/fyrthuz/Desktop/Ardentia
set -a; source .env; set +a

echo "[$(date '+%H:%M')] 1/6 rebuild datasets res9 + res8 (S2 8/8 completo)"
ARDENTIA_MIN_HA=0.1 .venv/bin/python -u -m ardintra.features.build_dataset > /tmp/opencode/build_s2full_res9.log 2>&1
ARDENTIA_RES=8 ARDENTIA_MIN_HA=0.1 ARDENTIA_NEG=40 .venv/bin/python -u -m ardintra.features.build_dataset > /tmp/opencode/build_s2full_res8.log 2>&1

echo "[$(date '+%H:%M')] 2/6 métricas finales res9 + res8"
.venv/bin/python -m ardintra.models.final_metrics > /tmp/opencode/fm9_final.log 2>&1
ARDENTIA_RES=8 .venv/bin/python -m ardintra.models.final_metrics > /tmp/opencode/fm8_final.log 2>&1

echo "[$(date '+%H:%M')] 3/6 balanced-ensembles res9 + res8"
ARDENTIA_RES=9 .venv/bin/python -m ardintra.models.balanced_ensemble > /tmp/opencode/be9_s2.log 2>&1
ARDENTIA_RES=8 .venv/bin/python -m ardintra.models.balanced_ensemble > /tmp/opencode/be8_s2.log 2>&1

echo "[$(date '+%H:%M')] 4/6 ablaciones res9 + res8"
.venv/bin/python -m ardintra.models.ablation_run > /tmp/opencode/abl9_s2.log 2>&1
ARDENTIA_RES=8 .venv/bin/python -m ardintra.models.ablation_run > /tmp/opencode/abl8_s2.log 2>&1

echo "[$(date '+%H:%M')] 5/6 gráficas"
.venv/bin/python -m ardintra.viz.metric_charts > /tmp/opencode/charts.log 2>&1

echo "[$(date '+%H:%M')] 6/6 vídeos res9 + res8 (2 s/frame, blend modelo)"
rm -rf data/processed/video_frames*
ARDENTIA_RES=9 .venv/bin/python -u -m ardintra.viz.video_blend 2023 > /tmp/opencode/video9_blend.log 2>&1
ARDENTIA_RES=8 .venv/bin/python -u -m ardintra.viz.video_blend 2023 > /tmp/opencode/video8_blend.log 2>&1

echo "[$(date '+%H:%M')] FULL-CYCLE-S2-DONE"
