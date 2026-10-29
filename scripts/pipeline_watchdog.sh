#!/usr/bin/env bash
# Watchdog de pipelines largos: relanza jobs muertos o con log congelado.
# Cada módulo es idempotente (resume por checkpoints: parquet/nc/tif).
set -u
ROOT="/home/fyrthuz/Desktop/Ardentia"

check() {
  local name="$1" pattern="$2" stale_min="$3" logfile="$4" module="$5" done_cmd="$6"
  eval "$done_cmd" && { echo "$(date '+%H:%M') $name: COMPLETO — nada que vigilar"; return; }
  local nproc; nproc=$(pgrep -f "$pattern" | wc -l)
  local stale; stale=$(( stale_min * 60 ))
  if [ "$nproc" -eq 0 ]; then
    echo "$(date '+%H:%M') $name: muerto → relanzando"
    setsid nohup bash -c "cd $ROOT && set -a && source .env && set +a && .venv/bin/python -u -m $module >> $logfile 2>&1" >/dev/null 2>&1 &
    return
  fi
  local mtime; mtime=$(stat -c %Y "$logfile" 2>/dev/null || echo 0)
  if [ "$mtime" != "0" ] && [ $(( $(date +%s) - mtime )) -gt $stale ]; then
    echo "$(date '+%H:%M') $name: log congelado ${stale_min}min → kill+relanza"
    pgrep -f "$pattern" | xargs -r kill
    sleep 2
    setsid nohup bash -c "cd $ROOT && set -a && source .env && set +a && .venv/bin/python -u -m $module >> $logfile 2>&1" >/dev/null 2>&1 &
  fi
}

check aemet "ardintra.ingest.aemet.backfill_missing" 30 /tmp/opencode/aemet_missing.log \
  ardintra.ingest.aemet.backfill_missing \
  '[ "$(ls data/raw/aemet/missing_backfill/ 2>/dev/null | wc -l)" -ge 18 ]'

check era5 "ardintra.ingest.era5.era5land_cds" 45 /tmp/opencode/era5.log \
  ardintra.ingest.era5.era5land_cds \
  '[ "$(ls data/raw/era5land/*.nc 2>/dev/null | wc -l)" -ge 32 ]'

check s2 "ardintra.ingest.s2_stac" 60 data/logs/s2_stac.log \
  "ardintra.ingest.s2_stac" \
  '[ "$(ls data/raw/s2/*.tif 2>/dev/null | wc -l)" -ge 8 ]'


# auto-rerun del gate: cuando AEMET+S2/ERA5 estén completos reconstruye y re-abla una sola vez
if [ -f "data/raw/aemet/stations.parquet" ] && \
   [ "$(ls data/raw/aemet/missing_backfill/ 2>/dev/null | wc -l)" -ge 18 ] && \
   [ "$(ls data/raw/s2/*.tif 2>/dev/null | wc -l)" -ge 8 ] && \
   [ ! -f "data/processed/features/v1/.gate_rerun_done" ]; then
    echo "$(date '+%H:%M') auto-rerun del gate (dataset+ablation)"
    cd $ROOT
    .venv/bin/python -m ardintra.features.build_dataset > /tmp/opencode/build_dataset.log 2>&1
    .venv/bin/python -m ardintra.models.ablation_run > /tmp/opencode/ablation.log 2>&1
    touch data/processed/features/v1/.gate_rerun_done
    echo "$(date '+%H:%M') auto-rerun del gate finalizado"
fi
