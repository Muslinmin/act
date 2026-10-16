#!/bin/bash
# Build the `franka_il` conda env from scratch: Python 3.11, torch 2.7.0+cu128, numpy 1.26, lerobot 0.4.1
# (huggingface/lerobot a5b29d43) and the exact versions of requirements-franka_il.txt.
#
#   bash franka_lerobot/env/setup_env.sh                       # env franka_il, lerobot cloned to ~/src/lerobot
#   ENV_NAME=myenv LEROBOT_DIR=/opt/lerobot APPLY_PATCH=0 bash franka_lerobot/env/setup_env.sh
#
# APPLY_PATCH=1 (default) applies patches/0001-lerobot-embed_images-skip-per-row-map.patch to lerobot: a
# speed-up of dataset writing (identical output) that the original env had. Training does not depend on it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENV_NAME=${ENV_NAME:-franka_il}
LEROBOT_DIR=${LEROBOT_DIR:-$HOME/src/lerobot}
LEROBOT_COMMIT=a5b29d430105f5235eb05bbf2db5a0d747a869d6   # "chore(installation): remove libero installation patch (#2416)"
APPLY_PATCH=${APPLY_PATCH:-1}

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env $ENV_NAME already exists; remove it or set ENV_NAME" >&2; exit 1
fi
conda create -n "$ENV_NAME" python=3.11 -y
PIP="conda run --no-capture-output -n $ENV_NAME python -m pip"

$PIP install --no-deps -r "$HERE/requirements-franka_il.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128

if [ ! -d "$LEROBOT_DIR/.git" ]; then
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
git -C "$LEROBOT_DIR" checkout --detach "$LEROBOT_COMMIT"
if [ "$APPLY_PATCH" = 1 ]; then
  git -C "$LEROBOT_DIR" apply --check "$HERE/../patches/0001-lerobot-embed_images-skip-per-row-map.patch" \
    && git -C "$LEROBOT_DIR" apply "$HERE/../patches/0001-lerobot-embed_images-skip-per-row-map.patch" \
    || echo "patch not applied (already applied?)"
fi
$PIP install --no-deps -e "$LEROBOT_DIR"

conda run -n "$ENV_NAME" python -c "
import torch, numpy, lerobot
from lerobot.policies.act.modeling_act import ACTPolicy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| numpy', numpy.__version__, '| lerobot ok')"
echo "done: conda activate $ENV_NAME"
