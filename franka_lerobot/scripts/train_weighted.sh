#!/bin/bash
# Weighted co-training run (lerobot 0.4.1 ACT): stock lerobot-train + the per-frame sampler of train_weighted.py.
#
#   franka_lerobot/scripts/train_weighted.sh <preset> <alpha_sim>
#     preset: simtele24 | real50_mimic47 | real50_mimic176 | real50_mimic376 | c100 (real only; alpha must be 0)
#     real50_mimic376: nothing held out (--dataset.episodes = all 426)
#   e.g.  WANDB=true DS_ROOT=~/data/cubes_20261014_30hz_real50_mimic47 franka_lerobot/scripts/train_weighted.sh real50_mimic47 0.2
#         -> $OUT_ROOT/act_cubes_real50_mimic47_a020_b40
#
# The ONLY difference to the reference runs (run_act_cubes_real50_mimic176_b40.sh, run_act_cubes_c100.sh) is the
# sampler: same --dataset.episodes (all but 10,25,40; lerobot 0.4.1 still samples those, kept for comparability), same
# stock merged-dataset normalisation stats, and the hyperparameters of act_cubes_real50_mimic176_b40 (= c100 with
# batch 40, save_freq 8000).
#
# Env: PY (python of the franka_il env, default `python`), DATA_ROOT (default ~/bc_data/lerobot/franka), DS_ROOT,
# OUT_ROOT (default $PWD/outputs/train), OUT_DIR, JOB_NAME, WANDB (default true, as the reference runs), DEVICE,
# NUM_WORKERS, STEPS, SAVE_FREQ, RESUME=1 (continue <out>/checkpoints/last), DRY_RUN=1 (200 steps, save at 200,
# wandb off, output <out>_dryrun; any device, set DEVICE=cpu for CPU), EXTRA (extra train_weighted.py args, e.g.
# "--exclude-episodes 10 25 40"). If DS_ROOT does not exist and the preset has a Hub copy, it is downloaded there.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRESET=${1:?usage: $0 <simtele24|real50_mimic47|real50_mimic176|real50_mimic376|c100> <alpha_sim>}
ALPHA=${2:?usage: $0 <preset> <alpha_sim>}
PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-$HOME/bc_data/lerobot/franka}
OUT_ROOT=${OUT_ROOT:-$PWD/outputs/train}
case "$PRESET" in
  simtele24)       NAME=cubes_20261014_30hz_simtele24;       N_EP=50;  MAP=simtele24_mapping.json;       HUB=Lusmse/franka_cubes_20261014_30hz_simtele24 ;;
  real50_mimic47)  NAME=cubes_20261014_30hz_real50_mimic47;  N_EP=97;  MAP=real50_mimic47_mapping.json;  HUB= ;;
  real50_mimic176) NAME=cubes_20261014_30hz_real50_mimic176; N_EP=226; MAP=real50_mimic176_mapping.json; HUB= ;;
  c100)            NAME=cubes_20261014_30hz;                 N_EP=50;  MAP=;                             HUB= ;;
  real50_mimic376) NAME=cubes_20261014_30hz_real50_mimic376; N_EP=426; MAP=real50_mimic376_mapping.json; HUB=Lusmse/franka_cubes_20261014_30hz_real50_mimic376
                   HELDOUT="-1" ;;  # all 50 real trained on (user 2026-10-19: robot eval is the test) + 376 mimic
  *) echo "unknown preset $PRESET" >&2; exit 2 ;;
esac
HELDOUT=${HELDOUT:-10,25,40}  # dataset indices left out of --dataset.episodes (all presets but real50_mimic376: 10,25,40; -1 = none)
DS_ROOT=${DS_ROOT:-$DATA_ROOT/$NAME}
TAG=$($PY -c "a=float('$ALPHA'); assert 0<=a<=1, a; print('a%03d' % round(a*100))")
JOB_NAME=${JOB_NAME:-act_cubes_${PRESET}_${TAG}_b40}
OUT_DIR=${OUT_DIR:-$OUT_ROOT/$JOB_NAME}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-8000}
WANDB=${WANDB:-true}
if [ "${DRY_RUN:-0}" = 1 ]; then STEPS=200; SAVE_FREQ=200; WANDB=false; OUT_DIR=${OUT_DIR}_dryrun; fi
DEVICE=${DEVICE:-cuda}
NUM_WORKERS=${NUM_WORKERS:-16}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
MAP_ARG=(); [ -n "$MAP" ] && MAP_ARG=(--slot-mapping "$HERE/../mappings/$MAP")
OWN=(--alpha-sim "$ALPHA" "${MAP_ARG[@]}" ${EXTRA:-})

if [ ! -f "$DS_ROOT/meta/info.json" ]; then
  if [ -n "$HUB" ]; then
    echo "=== download $HUB -> $DS_ROOT"
    HF_HUB_OFFLINE=0 hf download "$HUB" --repo-type dataset --local-dir "$DS_ROOT"
  else
    echo "no dataset at $DS_ROOT (set DS_ROOT)" >&2; exit 1
  fi
fi

if [ "${RESUME:-0}" = 1 ]; then
  CFG=$OUT_DIR/checkpoints/last/pretrained_model/train_config.json
  echo "=== resume $(date)  $CFG  ${OWN[*]}"
  exec $PY "$HERE/train_weighted.py" "${OWN[@]}" --resume=true --config_path="$CFG" --steps="$STEPS"
fi

TRAIN_EPS=$($PY -c "print('['+','.join(str(i) for i in range($N_EP) if i not in ($HELDOUT,))+']')")
echo "=== train $(date)  preset=$PRESET alpha_sim=$ALPHA  ds=$DS_ROOT  out=$OUT_DIR  git=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
$PY "$HERE/train_weighted.py" "${OWN[@]}" \
  --dataset.repo_id=franka/$NAME --dataset.root="$DS_ROOT" --dataset.episodes="$TRAIN_EPS" \
  --policy.type=act --policy.device="$DEVICE" --policy.push_to_hub=false \
  --policy.chunk_size=100 --policy.n_action_steps=100 --policy.optimizer_lr=3e-5 \
  --batch_size=40 --num_workers="$NUM_WORKERS" --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq=200 \
  --output_dir="$OUT_DIR" --job_name="$JOB_NAME" \
  --wandb.enable="$WANDB" --wandb.project=franka_act --wandb.disable_artifact=true \
  --wandb.notes="LeRobot ACT chunk100, $NAME, weighted co-training sampler alpha_sim=$ALPHA (only change vs the b40 reference runs; stock stats and --dataset.episodes); batch 40, save 8000"
echo "=== done $(date)"
