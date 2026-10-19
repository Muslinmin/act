#!/bin/bash
# Push selected checkpoints of a RUNNING lerobot training to a Hugging Face model repo as soon as they are written,
# so a partially trained model can be evaluated before the run ends.
#
#   nohup franka_lerobot/scripts/push_checkpoints_loop.sh <output_dir> <hf_repo_id> "<steps>" > push.log 2>&1 &
#   e.g. ... outputs/train/act_cubes_real50_mimic376_a090_b40 Lusmse/act_franka_cubes_real50_mimic376_a090_b40 \
#            "40000 80000 96000 120000 160000 200000"
#
# Each step S goes to <repo>/checkpoint_<S:06d>/ (the pretrained_model files only: config, weights, pre/post
# processors + their normaliser stats, train_config; never the optimizer state). A checkpoint dir is uploaded only
# once it is complete (model.safetensors present and its size unchanged for 60 s). The repo is created private.
# The LAST step of the list is also pushed to the repo ROOT with the model card (push_model_to_hub.py), so the repo
# id alone loads the final model. Ends after the last step. Env: PY (python with huggingface_hub, default python).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?usage: $0 <output_dir> <hf_repo_id> "<steps>"}
REPO=${2:?}
STEPS=(${3:?})
PY=${PY:-python}
LAST=${STEPS[-1]}
hf repo create "$REPO" --repo-type model --private --exist-ok || exit 1
for S in "${STEPS[@]}"; do
  D=$OUT/checkpoints/$(printf %06d "$S")/pretrained_model
  echo "=== $(date '+%F %T') waiting for $D"
  until [ -f "$D/model.safetensors" ]; do sleep 120; done
  while :; do a=$(stat -c %s "$D/model.safetensors"); sleep 60; [ "$a" = "$(stat -c %s "$D/model.safetensors")" ] && break; done
  for i in 1 2 3 4; do
    hf upload "$REPO" "$D" "checkpoint_$(printf %06d "$S")" --repo-type model \
      --commit-message "checkpoint $S (pushed during training)" && break
    echo "upload failed, retry $i"; sleep $((2 ** i * 15))
  done
  echo "=== $(date '+%F %T') pushed step $S -> $REPO/checkpoint_$(printf %06d "$S")"
done
$PY "$HERE/push_model_to_hub.py" --ckpt "$OUT/checkpoints/$(printf %06d "$LAST")/pretrained_model" --repo-id "$REPO" \
  --heldout --commit-message "final model (step $LAST) at the repo root + card"
echo "=== $(date '+%F %T') done"
