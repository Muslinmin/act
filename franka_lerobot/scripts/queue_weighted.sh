#!/bin/bash
# Sequential queue of train_weighted.sh runs: each starts only after the previous one exited 0.
#
#   nohup franka_lerobot/scripts/queue_weighted.sh <queue_file> > <queue_file>.log 2>&1 &
#
# <queue_file>: one run per line, "<preset> <alpha_sim>" (e.g. "real50_mimic47 0.2"); '#' comments. The file is re-read
# before every run, so a line that has not started yet can be deleted or commented out at any time to drop it.
# Finished lines go to <queue_file>.done; a failed run stops the queue. Per-run log: $LOG_DIR/<job_name>.log
# (default: next to the queue file). Every env var of train_weighted.sh (PY, DATA_ROOT, DS_ROOT_<preset>, OUT_ROOT,
# WANDB, ...) is passed through; DS_ROOT_<preset> (e.g. DS_ROOT_real50_mimic47) sets one preset's dataset root.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
Q=$(realpath "${1:?usage: $0 <queue_file>}")
DONE=$Q.done; touch "$DONE"
LOG_DIR=${LOG_DIR:-$(dirname "$Q")}
while true; do
  NEXT=$(grep -v '^\s*#' "$Q" | sed '/^\s*$/d' | while read -r l; do grep -qxF "$l" "$DONE" || { echo "$l"; break; }; done)
  [ -z "$NEXT" ] && { echo "=== queue empty $(date)"; exit 0; }
  read -r PRESET ALPHA <<< "$NEXT"
  TAG=$(python3 -c "print('a%03d' % round(float('$ALPHA')*100))")
  JOB=act_cubes_${PRESET}_${TAG}_b40
  DSV=DS_ROOT_$PRESET
  echo "=== start $JOB $(date)  log $LOG_DIR/$JOB.log"
  DS_ROOT=${!DSV:-${DS_ROOT:-}} JOB_NAME=$JOB bash "$HERE/train_weighted.sh" "$PRESET" "$ALPHA" > "$LOG_DIR/$JOB.log" 2>&1
  RC=$?
  echo "=== end $JOB rc=$RC $(date)"
  [ $RC -ne 0 ] && { echo "=== queue stopped: $JOB failed"; exit $RC; }
  echo "$NEXT" >> "$DONE"
done
