#!/bin/bash
# Sequential queue of co-training cells on the real50_mimic376 dataset (made by make_cells.py).
#   nohup franka_lerobot/scripts/queue_cells.sh <dir with cells.txt> > <dir>/queue.log 2>&1 &
# Line: "<job_name> <alpha_sim> <exclude file>". Re-read before every run; done lines go to cells.txt.done; a failed run
# stops the queue. 100k steps, ONLY the 100k checkpoint is saved (SAVE_FREQ=100000, disk). START_AT="00:00" waits.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=$(realpath "${1:?usage: $0 <cells dir>}"); Q=$D/cells.txt; DONE=$Q.done; touch "$DONE"
export PY=${PY:-/home/cir/miniconda3/envs/franka_il/bin/python}
export STEPS=${STEPS:-100000} SAVE_FREQ=${SAVE_FREQ:-100000}
export OUT_ROOT=${OUT_ROOT:-/home/cir/jello/JELLO_ARM_FRANKA/outputs/train}
export DS_ROOT=${DS_ROOT:-/home/cir/bc_data/lerobot/franka/cubes_20261014_30hz_real50_mimic376}
if [ -n "${START_AT:-}" ]; then
  T=$(date -d "$START_AT" +%s); [ "$T" -lt "$(date +%s)" ] && T=$((T+86400))
  echo "=== waiting until $(date -d @$T)"; sleep $((T - $(date +%s)))
fi
while true; do
  NEXT=$(grep -v '^\s*#' "$Q" | sed '/^\s*$/d' | while read -r l; do grep -qxF "$l" "$DONE" || { echo "$l"; break; }; done)
  [ -z "$NEXT" ] && { echo "=== queue empty $(date)"; exit 0; }
  read -r JOB ALPHA EXF <<< "$NEXT"
  echo "=== start $JOB alpha=$ALPHA excl=$EXF $(date)  df $(df -h --output=avail /home | tail -1)"
  JOB_NAME=$JOB EXTRA="--exclude-episodes $(cat "$D/$EXF")" bash "$HERE/train_weighted.sh" real50_mimic376 "$ALPHA" \
    > "$D/$JOB.log" 2>&1
  RC=$?
  echo "=== end $JOB rc=$RC $(date)"
  [ $RC -ne 0 ] && { echo "=== queue stopped: $JOB failed"; exit $RC; }
  echo "$NEXT" >> "$DONE"
done
