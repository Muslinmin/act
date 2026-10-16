# Franka ACT with LeRobot

This folder trains, evaluates and publishes the ACT policies for the Franka Panda **cubes** task
("pick the coloured cubes and place them in the box"). The demonstrations were recorded with GELLO teleoperation.

The policies are trained with the ACT implementation in [LeRobot](https://github.com/huggingface/lerobot)
(lerobot 0.4.1), not with `imitate_episodes.py` at the root of this repository. The root code (tonyzhaozh/act) is
unchanged on this branch.

| Model | Data | Held-out joint L1 | Held-out gripper L1 | Hub |
|---|---|---|---|---|
| `act_cubes_c100` | 47 real episodes, `franka/cubes_20261014_30hz` | **7.01 mrad** | **0.0103** | [Lusmse/act_franka_cubes_c100](https://huggingface.co/Lusmse/act_franka_cubes_c100) (private) |
| `act_cubes_c100_simtele24` | 24 sim-teleop + 23 real episodes | not trained yet | | |

## Why LeRobot ACT and not the original code

- **Data format.** The robot stack ([Muslinmin/JELLO_ARM_FRANKA](https://github.com/Muslinmin/JELLO_ARM_FRANKA),
  branch `il-recording`) converts its recordings to a LeRobot v3 dataset. That format holds AV1 video for two
  640x480 cameras, the joint/gripper state and actions, the flange pose, and the per-episode metadata.
  `imitate_episodes.py` reads the ALOHA hdf5 layout (`/observations/qpos`, `/action`, 14-DoF bimanual) with
  hard-coded task configs in `constants.py`.
- **Deployment.** The checkpoint contains the policy, its normaliser statistics (pre/post-processors) and the
  training config in one `pretrained_model/` folder. The robot's ZMQ policy server loads exactly that folder with
  `ACTPolicy.from_pretrained`, and the same folder goes to the Hugging Face Hub as is.
- **Tooling.** Checkpointing, resume, wandb logging, episode selection (held-out episodes) and video decoding are
  built in.
- **Same model.** LeRobot's `act` policy is a port of this repository's model: ResNet18 backbone, transformer
  encoder/decoder, CVAE with `kl_weight=10`, L1 loss, chunked actions, optional temporal ensembling.

## Contents

| Path | What |
|---|---|
| `env/setup_env.sh` | builds the conda env `franka_il` from scratch |
| `env/requirements-franka_il.txt` | exact pinned package set of `franka_il` (96 packages, installed with `--no-deps`) |
| `env/pip-freeze-franka_il-2026-10-16.txt` | full `pip freeze` of the original env, for reference only |
| `patches/0001-lerobot-embed_images-skip-per-row-map.patch` | optional lerobot speed-up for dataset writing (`git format-patch` of commit 56d676d) |
| `scripts/train_act.sh` | training: presets `c100`, `simtele24`, `custom`; resume |
| `scripts/eval_heldout.py` | offline open-loop eval on the held-out real episodes (copied from JELLO_ARM_FRANKA `sim-stack`) |
| `scripts/push_model_to_hub.py` | uploads a `pretrained_model` dir and a generated model card to the Hub |

## 1. Environment (from scratch)

Requirements: Linux x86-64, an NVIDIA driver for CUDA 12.8 (R570 or newer), conda, git.

```bash
git clone -b franka-lerobot-act https://github.com/Muslinmin/act.git && cd act
bash franka_lerobot/env/setup_env.sh            # creates conda env franka_il, clones lerobot to ~/src/lerobot
conda activate franka_il
```

`setup_env.sh` does the following:

```bash
conda create -n franka_il python=3.11 -y
conda activate franka_il
pip install --no-deps -r franka_lerobot/env/requirements-franka_il.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128          # torch 2.7.0+cu128, numpy 1.26.0, ...
git clone https://github.com/huggingface/lerobot.git ~/src/lerobot
git -C ~/src/lerobot checkout a5b29d430105f5235eb05bbf2db5a0d747a869d6   # lerobot 0.4.1
git -C ~/src/lerobot apply $PWD/franka_lerobot/patches/0001-lerobot-embed_images-skip-per-row-map.patch  # optional
pip install --no-deps -e ~/src/lerobot
```

Notes:

- **`--no-deps` is intentional.** The requirements file is the complete dependency closure of the env that trained
  the models, at its exact versions. A few of those versions are outside ranges that packages declare but do not
  need: numpy 1.26 with rerun-sdk (which declares numpy>=2), and wandb 0.28 with lerobot (which declares
  wandb<0.22). The resolver would refuse these, so the file is installed without it.
- `torchcodec` is not installed, so lerobot decodes video with **pyav**. lerobot prints a warning about this on
  every start. The warning is harmless.
- The patch (env var `APPLY_PATCH=1`, the default) skips a per-row `datasets.map` in `embed_images` when no feature
  needs embedding, so writing a video-only dataset is faster. The output is byte-identical. Training and inference
  do not use that code path. `APPLY_PATCH=0` gives plain upstream a5b29d4.
- The recipe was tested from scratch on 2026-10-16. The test built the env, imported the ACT and Diffusion
  policies, ran a 20-step training run plus a resume on the real dataset, and loaded the Hub model.
- Optional extras the JELLO repo uses: `pip install mujoco==3.12.0 pytest` (MuJoCo replay, tests).

Check:

```bash
python -c "import torch, lerobot; from lerobot.policies.act.modeling_act import ACTPolicy; \
           print(torch.__version__, torch.cuda.is_available())"     # 2.7.0+cu128 True
```

## 2. Data

Recording and conversion are part of the robot repo, not this one:
[Muslinmin/JELLO_ARM_FRANKA](https://github.com/Muslinmin/JELLO_ARM_FRANKA), branch `il-recording`,
`docs/IL_DATA_PIPELINE.md` (QC, conversion, conventions).

```bash
cd JELLO_ARM_FRANKA            # branch il-recording, run in franka_il
python scripts/convert_to_lerobot.py --src ~/bc_data/20261014 --out ~/bc_data/lerobot \
    --repo-id franka/cubes_20261014_30hz --fps 30          # 30 Hz, full 640x480 images
```

The resulting dataset `franka/cubes_20261014_30hz` is local and not on the Hub:

- 50 episodes, 66,930 frames, 30 Hz, LeRobot format v3.0, AV1 video.
- `observation.state` (8): 7 joints [rad] + gripper **closed fraction** (1 = closed).
- `action` (8): 7 commanded joints, absolute, + gripper closed fraction.
- `observation.images.wrist`, `observation.images.base`: 480x640 RGB.
- `observation.ee_pose` (7): stored, but ACT does not read it.
- **Held out: episodes 10, 25, 40** (dataset indices). They are never trained on and are used for the offline
  eval. The other 47 episodes are for training.

`simtele24` dataset: `franka/cubes_20261014_30hz_simtele24` keeps the same 50 slots. 24 training slots hold the
user's sim-teleop episodes from the same cube/box layouts; 23 training slots and the 3 held-out slots are real. It is
built on JELLO_ARM_FRANKA branch `sim-stack`, see `docs/sim/experiments/COTRAIN.md` ("simtele24") there.

## 3. Training

```bash
conda activate franka_il
cd /path/to/JELLO_ARM_FRANKA         # outputs go to $PWD/outputs/train/<job> (or set OUT_ROOT / OUT_DIR)
WANDB=true /path/to/act/franka_lerobot/scripts/train_act.sh c100         # -> outputs/train/act_cubes_c100
WANDB=true /path/to/act/franka_lerobot/scripts/train_act.sh simtele24    # -> outputs/train/act_cubes_c100_simtele24
```

`train_act.sh` is a parameterised copy of the scripts that ran the original trainings:
`~/bc_data/lerobot/run_act_cubes_c100.sh` and `run_act_cubes_c100_simtele24.sh` on the training workstation.
`c100` with `WANDB=true` produces the same `train_config.json` as the original run, except `output_dir`. The two
presets differ only in dataset, output and job names (COTRAIN.md). Hyperparameters (every value is in
`train_config.json`):

| Setting | Value | How it is set |
|---|---|---|
| policy | `act` | `--policy.type=act` |
| chunk_size / n_action_steps | 100 / 100 (3.3 s at 30 Hz; all 100 actions are executed per query) | `CHUNK` |
| optimizer | AdamW, lr **3e-5**, backbone lr 1e-5, weight_decay 1e-4, grad clip 10, no scheduler | `LR` |
| batch size | **32** | `BATCH` |
| steps | **100,000** (about 48 epochs) | `STEPS` |
| checkpoints | every 10,000 steps, plus `last` | `SAVE_FREQ` |
| backbone | ResNet18, ImageNet weights (`IMAGENET1K_V1`) | lerobot default |
| transformer | dim 512, 8 heads, FFN 3200, 4 encoder / 1 decoder layers, dropout 0.1 | lerobot default |
| CVAE | latent 32, 4 VAE encoder layers, kl_weight 10 | lerobot default |
| normalisation | mean/std for images, state and action | lerobot default |
| image augmentation | none | lerobot default |
| n_obs_steps | 1 | lerobot default |
| seed | 1000 | lerobot default |
| episodes | all except `HELDOUT=10,25,40` | `HELDOUT` |
| data loader | 16 workers, pyav decoding | `NUM_WORKERS` |
| logging | wandb project `franka_act` (`WANDB=true`), every 200 steps | `WANDB`, `LOG_FREQ` |

On one RTX 6000 Ada, c100 took about 8 h 20 min. The final training loss was 0.032.

Other knobs: `DATA_ROOT` (default `~/bc_data/lerobot`), `DS_ROOT`, `OUT_ROOT`, `OUT_DIR`, `DEVICE`,
`HF_HUB_OFFLINE` (default 1). Preset `custom` needs `REPO_ID` and `JOB_NAME`. If the dataset is missing and both
`RAW_SRC` and `JELLO_REPO` are set, the script runs the conversion first.

Smoke test (2 minutes): `STEPS=20 SAVE_FREQ=20 OUT_DIR=/tmp/act_smoke franka_lerobot/scripts/train_act.sh c100`

### Resuming

```bash
RESUME=1 franka_lerobot/scripts/train_act.sh c100                 # continue to STEPS (100k)
RESUME=1 STEPS=150000 franka_lerobot/scripts/train_act.sh c100    # train longer
```

This runs `lerobot-train --resume=true --config_path=<out>/checkpoints/last/pretrained_model/train_config.json`.
lerobot restores the model, optimizer, step and RNG from `checkpoints/last/training_state`.

## 4. Held-out evaluation

```bash
HF_HUB_OFFLINE=1 python franka_lerobot/scripts/eval_heldout.py \
    --jello-repo /path/to/JELLO_ARM_FRANKA \
    --ckpt /path/to/JELLO_ARM_FRANKA/outputs/train/act_cubes_c100/checkpoints/last/pretrained_model \
    --out results/eval_act_cubes_c100.json
# a Hub model works too:  --ckpt Lusmse/act_franka_cubes_c100   (without HF_HUB_OFFLINE)
```

The metric is offline and open-loop, on the **real** held-out episodes 10/25/40 of `franka/cubes_20261014_30hz`:

1. Every 3rd frame is a query point.
2. The policy sees the stored observation and predicts its 100-step chunk.
3. The chunk is compared with the recorded actions of the next 100 frames. Steps past the episode end are skipped.

Reported values: joint L1 (mean over the 7 joints, in mrad) and gripper L1 (closed fraction). Each is averaged per
episode, then over the 3 episodes. Inference goes through the robot's own `scripts/act_policy_server.py::LeRobotACT`,
the deployed code path, so `--jello-repo` must point to a JELLO_ARM_FRANKA checkout (`il-recording` or later).
On one GPU it takes about a minute (1,350 queries; 42 s on an RTX 6000 Ada).

Baseline `act_cubes_c100` (step 100k = `last`):

| | joint L1 [mrad] | gripper L1 |
|---|---|---|
| mean of episodes 10/25/40 | **7.01** | **0.0103** |
| ep 10 / 25 / 40 | 6.70 / 6.98 / 7.35 | 0.0104 / 0.0114 / 0.0091 |
| chunk step 0 / 49 / 99 | 5.63 / 7.13 / 8.75 | |

This is an imitation metric. It does not measure success on the robot.

## 5. Pushing a model to the Hub

```bash
hf auth login                                  # once, with a write token
python franka_lerobot/scripts/push_model_to_hub.py \
    --ckpt /path/to/JELLO_ARM_FRANKA/outputs/train/act_cubes_c100/checkpoints/last/pretrained_model \
    --repo-id Lusmse/act_franka_cubes_c100 \
    --eval-json results/eval_act_cubes_c100.json \
    --jello-repo /path/to/JELLO_ARM_FRANKA
```

The script creates the repo **private** (`--public` overrides) and makes one commit with:

- the `pretrained_model` files: `config.json`, `model.safetensors`, the pre/post-processor JSON and normaliser
  safetensors, `train_config.json`;
- `README.md`, the generated model card: dataset, train/held-out episodes, every hyperparameter, held-out metrics,
  usage, deployment, and the git commits of this repo, lerobot and JELLO_ARM_FRANKA;
- `eval_heldout.json`.

The optimizer state is never uploaded. `--dry-run --out-card card.md` previews the card.

Load it back:

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
policy = ACTPolicy.from_pretrained("Lusmse/act_franka_cubes_c100").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path="Lusmse/act_franka_cubes_c100")
```

## 6. Deploying on the robot

Deployment is in JELLO_ARM_FRANKA (branch `il-recording`), `docs/IL_DATA_PIPELINE.md` section 7. In short:

```bash
# franka_il: ACT on the GPU behind ZMQ (owns gripper flip, FK for ee_pose, image resize)
hf download Lusmse/act_franka_cubes_c100 --local-dir ~/models/act_franka_cubes_c100     # or use the local checkpoint
HF_HUB_OFFLINE=1 python scripts/act_policy_server.py --ckpt ~/models/act_franka_cubes_c100 --fps 30
# .venv (T4): the 100 Hz control loop plays the chunks at 30 Hz with temporal ensembling; hand on the e-stop
python experiments/run_env.py --agent policy --policy-gui \
    --wrist-camera-serial 260422273008 --base-camera-serial 335622071698
```

`--fps 30` is needed for a downloaded checkpoint. Without it, the server reads the rate from the training dataset's
`meta/info.json` through the absolute path in `train_config.json`.
