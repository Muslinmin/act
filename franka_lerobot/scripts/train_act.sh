#!/bin/bash
# Train LeRobot ACT (lerobot 0.4.1) on a Franka LeRobot dataset with held-out episodes.
#
# Parameterised copy of ~/bc_data/lerobot/run_act_cubes_c100.sh (the script that trained act_cubes_c100,
# 2026-10-14). With the default preset and WANDB=true it runs the same lerobot-train command line.
#
#   conda activate franka_il
#   franka_lerobot/scripts/train_act.sh c100                 # real data, 47 training eps
#   franka_lerobot/scripts/train_act.sh simtele24            # 24 sim-teleop + 23 real training eps
#   RESUME=1 franka_lerobot/scripts/train_act.sh c100        # continue from <out>/checkpoints/last
#   STEPS=200 SAVE_FREQ=100 OUT_DIR=/tmp/act_smoke franka_lerobot/scripts/train_act.sh c100   # smoke test
#
# Every setting below can be overridden from the environment.
set -euo pipefail

PRESET=${1:-c100}
DATA_ROOT=${DATA_ROOT:-$HOME/bc_data/lerobot}       # parent of <repo_id>/ dirs written by convert_to_lerobot.py
OUT_ROOT=${OUT_ROOT:-$PWD/outputs/train}             # original runs: JELLO_ARM_FRANKA/outputs/train
case "$PRESET" in
  c100)
    : "${REPO_ID:=franka/cubes_20261014_30hz}"
    : "${JOB_NAME:=act_cubes_c100}"
    : "${NOTES:=LeRobot ACT chunk100, 640x480 wrist+base, 47 eps of 20261014 cubes (held out 10,25,40)}" ;;
  simtele24)
    # Dataset built on JELLO_ARM_FRANKA branch sim-stack (docs/sim/experiments/COTRAIN.md, "simtele24"):
    # 24 sim-teleop + 23 real training slots, held-out slots 10/25/40 real. Same training as c100.
    : "${REPO_ID:=franka/cubes_20261014_30hz_simtele24}"
    : "${JOB_NAME:=act_cubes_c100_simtele24}"
    : "${NOTES:=LeRobot ACT chunk100, 640x480 wrist+base, 20261014 cubes simtele24 mix (24 sim teleop + 23 real; held out 10,25,40)}" ;;
  custom)
    : "${REPO_ID:?set REPO_ID}" "${JOB_NAME:?set JOB_NAME}"
    : "${NOTES:=}" ;;
  *) echo "usage: $0 [c100|simtele24|custom]" >&2; exit 2 ;;
esac
DS_ROOT=${DS_ROOT:-$DATA_ROOT/$REPO_ID}
OUT_DIR=${OUT_DIR:-$OUT_ROOT/$JOB_NAME}

# Hyperparameters (values of act_cubes_c100)
HELDOUT=${HELDOUT:-10,25,40}         # dataset episode indices left out of training (offline eval)
CHUNK=${CHUNK:-100}                  # policy.chunk_size = policy.n_action_steps (100 steps = 3.3 s at 30 Hz)
LR=${LR:-3e-5}                       # policy.optimizer_lr (backbone keeps lerobot's 1e-5 default)
BATCH=${BATCH:-32}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-10000}
LOG_FREQ=${LOG_FREQ:-200}
NUM_WORKERS=${NUM_WORKERS:-16}
DEVICE=${DEVICE:-cuda}
WANDB=${WANDB:-false}                # the original runs used true (project franka_act); needs `wandb login`
WANDB_PROJECT=${WANDB_PROJECT:-franka_act}
RESUME=${RESUME:-0}

PY=${PY:-python}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # local dataset only; nothing goes to the Hub during training

if [ ! -f "$DS_ROOT/meta/info.json" ]; then
  if [ -n "${RAW_SRC:-}" ] && [ -n "${JELLO_REPO:-}" ]; then
    echo "=== convert $(date)  $RAW_SRC -> $DS_ROOT"
    (cd "$JELLO_REPO" && $PY scripts/convert_to_lerobot.py --src "$RAW_SRC" --out "$DATA_ROOT" --repo-id "$REPO_ID" --fps 30)
  else
    echo "no dataset at $DS_ROOT. Convert it first (README 'Data'), or set RAW_SRC=<raw session dir> and" >&2
    echo "JELLO_REPO=<JELLO_ARM_FRANKA checkout> to convert here." >&2
    exit 1
  fi
fi

if [ "$RESUME" = 1 ]; then
  CFG=$OUT_DIR/checkpoints/last/pretrained_model/train_config.json
  echo "=== resume $(date)  from $CFG"
  # lerobot restores model, optimizer, scheduler, step and RNG from checkpoints/last/training_state.
  # STEPS may be raised here to train longer than the original run.
  exec lerobot-train --resume=true --config_path="$CFG" --steps="$STEPS"
fi

TRAIN_EPS=$($PY -c "
import json, sys
n = json.load(open('$DS_ROOT/meta/info.json'))['total_episodes']
held = {int(x) for x in '$HELDOUT'.split(',') if x}
print('[' + ','.join(str(i) for i in range(n) if i not in held) + ']')")
echo "=== train $(date)  dataset=$REPO_ID  out=$OUT_DIR  episodes=$TRAIN_EPS"

lerobot-train \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DS_ROOT" --dataset.episodes="$TRAIN_EPS" \
  --policy.type=act --policy.device="$DEVICE" --policy.push_to_hub=false \
  --policy.chunk_size="$CHUNK" --policy.n_action_steps="$CHUNK" --policy.optimizer_lr="$LR" \
  --batch_size="$BATCH" --num_workers="$NUM_WORKERS" --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ" \
  --output_dir="$OUT_DIR" --job_name="$JOB_NAME" \
  --wandb.enable="$WANDB" --wandb.project="$WANDB_PROJECT" --wandb.disable_artifact=true --wandb.notes="$NOTES"
echo "=== done $(date)"
