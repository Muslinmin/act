#!/usr/bin/env python
"""Upload a LeRobot ACT checkpoint (a `.../checkpoints/<step>/pretrained_model` dir) to a Hugging Face model repo,
together with a generated model card (README.md) and, optionally, the held-out eval JSON of eval_heldout.py.

    conda activate franka_il            # needs huggingface_hub; `hf auth login` once (write token)
    python franka_lerobot/scripts/push_model_to_hub.py \
        --ckpt /home/cir/jello/JELLO_ARM_FRANKA/outputs/train/act_cubes_c100/checkpoints/last/pretrained_model \
        --repo-id Lusmse/act_franka_cubes_c100 \
        --eval-json results/eval_act_cubes_c100.json \
        --jello-repo /home/cir/jello/JELLO_ARM_FRANKA
    # --dry-run writes the card to <out-card> and uploads nothing.

What is uploaded: every file of the pretrained_model dir (config.json, model.safetensors, the pre/post-processor
JSON + normaliser-stat safetensors, train_config.json), README.md, and eval_heldout.json when --eval-json is given.
The training_state (optimizer) is never uploaded. The repo is created PRIVATE unless --public is given.

The card is built from train_config.json (dataset, episodes, every hyperparameter), the training dataset's
meta/info.json when it exists locally (fps, features, episode/frame counts), the eval JSON and the git commits of
this repo, the lerobot install and the JELLO_ARM_FRANKA checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPLOAD_FILES = (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def git(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(cwd), *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_desc(cwd: Path) -> str | None:
    sha = git(cwd, "rev-parse", "HEAD")
    if not sha:
        return None
    branch = git(cwd, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    url = git(cwd, "remote", "get-url", "origin") or "?"
    dirty = " (uncommitted changes)" if git(cwd, "status", "--porcelain", "--untracked-files=no") else ""
    return f"{url} @ `{sha[:10]}` (branch `{branch}`){dirty}"


def lerobot_desc() -> str:
    try:
        import importlib.metadata as md

        import lerobot

        ver = md.version("lerobot")
        src = Path(lerobot.__file__).resolve().parent
        sha = git(src, "rev-parse", "HEAD")
        if sha:
            log = git(src, "log", "--format=%h %s", "-3") or ""
            return f"lerobot {ver}, source `{sha[:10]}`; recent commits:\n\n  ```\n  " + log.replace("\n", "\n  ") + "\n  ```\n"
        return f"lerobot {ver}"
    except Exception as e:  # noqa: BLE001
        return f"(could not inspect lerobot: {e})"


def fmt_eps(eps: list[int]) -> str:
    """[0,1,2,3,5,6] -> '0-3, 5-6'"""
    out, start, prev = [], None, None
    for e in sorted(eps) + [None]:
        if start is None:
            start = prev = e
        elif e is not None and e == prev + 1:
            prev = e
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = e
    return ", ".join(out)


def build_card(a, ckpt: Path) -> str:
    tc = json.loads((ckpt / "train_config.json").read_text())
    pol, ds, opt = tc["policy"], tc["dataset"], tc["optimizer"]
    info = {}
    info_path = Path(ds["root"]) / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    n_total = info.get("total_episodes")
    train_eps = ds.get("episodes")
    held = sorted(set(range(n_total)) - set(train_eps)) if (n_total and train_eps) else a.heldout
    fps = info.get("fps", "?")
    ev = json.loads(Path(a.eval_json).read_text()) if a.eval_json else None
    name = a.repo_id.split("/")[-1]

    feats_in = "\n".join(f"| `{k}` | {v['type']} | {v['shape']} |" for k, v in pol["input_features"].items())
    feats_out = "\n".join(f"| `{k}` | {v['type']} | {v['shape']} |" for k, v in pol["output_features"].items())

    hp = [
        ("policy", "LeRobot `act` (ACT, Zhao et al. 2023)"),
        ("chunk_size / n_action_steps", f"{pol['chunk_size']} / {pol['n_action_steps']}"
         + (f" ({pol['chunk_size'] / fps:.2f} s at {fps} Hz)" if isinstance(fps, (int, float)) else "")),
        ("n_obs_steps", pol["n_obs_steps"]),
        ("vision backbone", f"{pol['vision_backbone']} ({pol['pretrained_backbone_weights']})"),
        ("dim_model / n_heads / dim_feedforward", f"{pol['dim_model']} / {pol['n_heads']} / {pol['dim_feedforward']}"),
        ("encoder / decoder layers", f"{pol['n_encoder_layers']} / {pol['n_decoder_layers']}"),
        ("VAE", f"use_vae={pol['use_vae']}, latent_dim={pol['latent_dim']}, "
                f"n_vae_encoder_layers={pol['n_vae_encoder_layers']}, kl_weight={pol['kl_weight']}"),
        ("dropout", pol["dropout"]),
        ("normalisation", ", ".join(f"{k}: {v}" for k, v in pol["normalization_mapping"].items())),
        ("optimizer", f"{opt['type']}, lr {opt['lr']}, backbone lr {pol['optimizer_lr_backbone']}, "
                      f"weight_decay {opt['weight_decay']}, betas {opt['betas']}, grad_clip_norm {opt['grad_clip_norm']}"),
        ("scheduler", tc.get("scheduler")),
        ("batch size", tc["batch_size"]),
        ("steps", f"{tc['steps']:,}"),
        ("seed", tc["seed"]),
        ("image augmentation", "enabled" if ds["image_transforms"]["enable"] else "none"),
        ("temporal_ensemble_coeff", pol["temporal_ensemble_coeff"]),
        ("video backend", ds.get("video_backend")),
    ]
    hp_rows = "\n".join(f"| {k} | {v} |" for k, v in hp)

    if ev:
        per_ep = "\n".join(
            f"| {e} | {v['frames']} | {v['query_points']} | {v['joint_l1_mrad']:.2f} | {v['gripper_l1']:.4f} |"
            for e, v in ev["per_episode"].items())
        hz = ev.get("per_horizon", {})
        hz_rows = "\n".join(f"| {s} | {j:.2f} | {g:.4f} |"
                            for s, j, g in zip(hz.get("step", []), hz.get("joint_l1_mrad", []), hz.get("gripper_l1", [])))
        eval_md = f"""Offline, open-loop, on the **real held-out episodes {', '.join(map(str, ev['episodes']))}** of
`{ev['dataset']['repo_id']}` (never seen in training), with
[`franka_lerobot/scripts/eval_heldout.py`]({a.code_url}/franka_lerobot/scripts/eval_heldout.py):
query points are frames 0, {ev['stride']}, {2 * ev['stride']}, ... of each episode; at each one the policy sees the stored observation and predicts its
{ev['chunk']}-step chunk, which is compared with the recorded actions of the next {ev['chunk']} frames (steps past
the episode end are skipped, like the training loss's `action_is_pad`). Joint L1 = mean |error| over the 7 joints
in mrad; gripper L1 = mean |error| of the closed fraction (0..1). Episodes are averaged with equal weight.

| metric | value |
|---|---|
| **joint L1** | **{ev['joint_l1_mrad']:.2f} mrad** |
| **gripper L1** | **{ev['gripper_l1']:.4f}** |
| joint L1, pooled over all compared steps | {ev['pooled']['joint_l1_mrad']:.2f} mrad |
| gripper L1, pooled | {ev['pooled']['gripper_l1']:.4f} |

| held-out episode | frames | queries | joint L1 [mrad] | gripper L1 |
|---|---|---|---|---|
{per_ep}

| chunk step | joint L1 [mrad] | gripper L1 |
|---|---|---|
{hz_rows}

Evaluated checkpoint: `{Path(ev['ckpt']).parent.name}/pretrained_model` (the uploaded weights; full JSON in
`eval_heldout.json`). This is an offline imitation metric, not a robot success rate."""
    else:
        eval_md = "No held-out evaluation was attached (`--eval-json`)."

    prov = [("this code", git_desc(HERE) or "?"), ("lerobot", lerobot_desc())]
    if a.jello_repo:
        prov.append(("JELLO_ARM_FRANKA (conversion, deployment)", git_desc(Path(a.jello_repo)) or a.jello_repo))
    prov += [tuple(p.split("=", 1)) for p in a.provenance]
    prov_md = "\n".join(f"- **{k}**: {v}" for k, v in prov)

    data_md = f"""- LeRobot dataset `{ds['repo_id']}` (LeRobot format {info.get('codebase_version', '?')}, not on the Hub;
  local root at training time `{ds['root']}`), robot `{info.get('robot_type', '?')}`, {fps} Hz.
- {n_total if n_total else '?'} episodes / {info.get('total_frames', '?')} frames in total.
  **Trained on {len(train_eps) if train_eps else 'all'} episodes** ({fmt_eps(train_eps) if train_eps else 'all'});
  **held out: {', '.join(map(str, held)) if held else 'none'}** (used only for the evaluation below).
- Task: {a.task}
- Recorded with GELLO teleoperation of a Franka Panda (polymetis, 100 Hz loop), two RealSense cameras
  (wrist + base) at 640x480, converted with `scripts/convert_to_lerobot.py` of JELLO_ARM_FRANKA
  (branch `il-recording`), default joint/absolute actions.
{a.data_note}"""

    return f"""---
library_name: lerobot
pipeline_tag: robotics
tags:
- lerobot
- act
- robotics
- imitation-learning
- franka
---

# {name}

LeRobot **ACT** policy for a Franka Panda pick-and-place task ({a.task}), trained with
`lerobot-train` (lerobot 0.4.1) from GELLO teleoperation demonstrations. Code, environment recipe and training
scripts: [{a.code_url}]({a.code_url}) (folder `franka_lerobot/`).

## Inputs and outputs

| feature | type | shape |
|---|---|---|
{feats_in}
{feats_out}

- `observation.state` = 7 measured joint angles [rad] + gripper **closed fraction** (1 = closed, = 1 - width/0.09).
- `action` = 7 commanded joint angles [rad] (absolute) + gripper closed fraction, at {fps} Hz.
- Images: RGB, 480x640 (H x W), the latest frame that had arrived at each control tick.
- `observation.ee_pose` (flange pose) is in the dataset and the normaliser but ACT in lerobot 0.4.1 does not read it.

## Training data

{data_md}

## Hyperparameters

| parameter | value |
|---|---|
{hp_rows}

Exact command line: `franka_lerobot/scripts/train_act.sh` in the code repo; the full config is `train_config.json`.
{a.train_note}

## Held-out evaluation

{eval_md}

## Usage

```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

repo, device = "{a.repo_id}", "cuda"                               # or "cpu"
cfg = PreTrainedConfig.from_pretrained(repo)
cfg.device = device                                                  # the stored config says "cuda"
policy = ACTPolicy.from_pretrained(repo, config=cfg).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=repo,
    preprocessor_overrides={{"device_processor": {{"device": device}}}},
    postprocessor_overrides={{"device_processor": {{"device": device}}}})
# obs: {{"observation.state": (8,), "observation.ee_pose": (7,), "observation.images.wrist"/"base": (3,480,640) float in [0,1]}}
# chunk = post(policy.predict_action_chunk(pre(obs)))  -> ({pol['chunk_size']}, 8) unnormalised actions
```

Private repo: log in first (`hf auth login`) or set `HF_TOKEN`.

## Deployment on the robot

Served with `scripts/act_policy_server.py --ckpt <downloaded dir>` (conda env `franka_il`) and driven by
`experiments/run_env.py --agent policy --policy-gui` in the JELLO_ARM_FRANKA repo (branch `il-recording`,
`docs/IL_DATA_PIPELINE.md` section 7). The server applies the deployment conventions (gripper flip, FK for
`observation.ee_pose`, image resize); `run_env.py` plays chunks at {fps} Hz with ACT temporal ensembling.
The server reads the policy rate from the training dataset's `meta/info.json`; on a machine without that
dataset pass `--fps {fps}`.

## Files

`config.json`, `model.safetensors` (policy), `policy_preprocessor*.{{json,safetensors}}` /
`policy_postprocessor*.{{json,safetensors}}` (normaliser stats; required), `train_config.json`
(training config; contains absolute paths of the training machine){', `eval_heldout.json`' if ev else ''}.

## Provenance

{prov_md}

Card generated {date.today().isoformat()} by `franka_lerobot/scripts/push_model_to_hub.py`.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--repo-id", required=True, help="e.g. Lusmse/act_franka_cubes_c100")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    ap.add_argument("--eval-json", default=None, help="output of eval_heldout.py --out")
    ap.add_argument("--jello-repo", default=None, help="JELLO_ARM_FRANKA checkout, for the provenance section")
    ap.add_argument("--heldout", type=int, nargs="*", default=[10, 25, 40],
                    help="held-out episodes, only used when the dataset meta is not available locally")
    ap.add_argument("--task", default="pick the coloured cubes and place them in the box")
    ap.add_argument("--data-note", default="", help="extra markdown appended to the training-data section")
    ap.add_argument("--train-note", default="", help="extra markdown appended to the hyperparameter section")
    ap.add_argument("--provenance", action="append", default=[], help="extra 'key=value' provenance line")
    ap.add_argument("--code-url", default="https://github.com/Muslinmin/act/tree/franka-lerobot-act")
    ap.add_argument("--out-card", default=None, help="also write the card here")
    ap.add_argument("--dry-run", action="store_true", help="write the card, upload nothing")
    ap.add_argument("--commit-message", default=None)
    a = ap.parse_args()

    ckpt = Path(a.ckpt).expanduser().resolve()
    files = [ckpt / f for f in UPLOAD_FILES] + sorted(ckpt.glob("policy_*processor*.safetensors"))
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"{ckpt} is not a complete pretrained_model dir, missing {missing}")
    card = build_card(a, ckpt)
    if a.out_card:
        Path(a.out_card).write_text(card)
        print(f"wrote {a.out_card}")
    if a.dry_run:
        print(card)
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    url = api.create_repo(a.repo_id, repo_type="model", private=not a.public, exist_ok=True)
    print(f"repo: {url}")
    ops = [CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f)) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    if a.eval_json:
        ops.append(CommitOperationAdd(path_in_repo="eval_heldout.json", path_or_fileobj=str(Path(a.eval_json))))
    msg = a.commit_message or f"Upload {ckpt.parent.name}/pretrained_model + model card"
    info = api.create_commit(a.repo_id, operations=ops, commit_message=msg, repo_type="model")
    print(f"commit: {info.commit_url}")
    for f in api.list_repo_files(a.repo_id, repo_type="model"):
        print("  ", f)


if __name__ == "__main__":
    main()
