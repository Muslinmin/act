# Provenance: copied from Muslinmin/JELLO_ARM_FRANKA, branch `sim-stack`, sim/scripts/wps/eval_heldout.py
# (last changed in commit 1182f05, "WPS prep: ... eval_heldout.py"), on 2026-10-16.
# Changes from the original, none of which touch the metric:
#   * the JELLO_ARM_FRANKA checkout is taken from --jello-repo / $JELLO_REPO instead of this file's location
#     (the inference path is still that repo's scripts/act_policy_server.py::LeRobotACT, i.e. the deployed one);
#   * --ckpt may also be a Hugging Face model repo id (downloaded with snapshot_download);
#   * the policy rate is read from the eval dataset's meta/info.json when the checkpoint's training dataset root
#     (stored as an absolute path in train_config.json) does not exist on this machine, or given with --fps.
"""Offline open-loop eval of an ACT checkpoint on the REAL held-out episodes (COTRAIN.md "Evaluation").

    conda franka_il (LeRobot 0.4.1), GPU, inference only:
    HF_HUB_OFFLINE=1 python franka_lerobot/scripts/eval_heldout.py \
        --jello-repo /home/cir/jello/JELLO_ARM_FRANKA \
        --ckpt /home/cir/jello/JELLO_ARM_FRANKA/outputs/train/act_cubes_c100/checkpoints/last/pretrained_model \
        --out results/eval_act_cubes_c100.json
    # or a model on the Hub (drop HF_HUB_OFFLINE): --ckpt Lusmse/act_franka_cubes_c100

Metric (fixed in COTRAIN.md at the 01:00 check-in, before any number was seen; details in the WPS-prep entry):
  * dataset: the REAL LeRobot dataset `franka/cubes_20261014_30hz` (~/bc_data/lerobot/franka/cubes_20261014_30hz),
    held-out episodes 10, 25, 40 (dataset indices, as in run_act_cubes_c100.sh), read only;
  * query points: every 3rd frame of the 30 Hz episode (frame 0, 3, 6, ...);
  * at each query point the policy sees that frame's observation (state + wrist + base images, exactly as stored) and
    predicts its 100-step action chunk; the chunk is compared with the RECORDED actions of frames t .. t+99. Near the
    episode end the chunk runs past the last frame: only the steps that exist are compared (= the action_is_pad mask
    LeRobot's ACT loss uses in training);
  * joint L1 [mrad] = mean |predicted - recorded| over the 7 joints; gripper L1 = mean |predicted - recorded| of the
    closed fraction (dimensionless, 0..1);
  * averaged over every compared chunk step of an episode, then over the 3 episodes (equal weight). The pooled mean
    (every compared step weighted alike) and the error per chunk horizon are reported too.
Inference path = the deployed one: `scripts/act_policy_server.py::LeRobotACT` (ACTPolicy.from_pretrained + the
checkpoint's own pre/post processors, `predict_action_chunk`, per-step unnormalisation), fed with the dataset frame
converted back to what the server receives (raw joints with width/0.09, uint8 HWC images). Same code for any
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_JELLO_REPO = os.environ.get("JELLO_REPO", "/home/cir/jello/JELLO_ARM_FRANKA")

HELDOUT = (10, 25, 40)  # run_act_cubes_c100.sh: dataset indices left out of training
STRIDE = 3  # query every 3rd frame at 30 Hz
REAL_REPO_ID = "franka/cubes_20261014_30hz"
REAL_ROOT = Path("~/bc_data/lerobot/franka/cubes_20261014_30hz").expanduser()


def to_request(item) -> dict:
    """A LeRobot dataset item -> the policy server's request (inverse of convert_to_lerobot / the server's infer)."""
    state = item["observation.state"].numpy().astype(np.float32)
    q = state.copy()
    q[7] = 1.0 - state[7]  # dataset: closed fraction; server input: width / 0.09 (1 = open)
    images = {}
    for k, v in item.items():
        if k.startswith("observation.images."):
            # decoded video frame, float CHW in [0, 1] -> uint8 HWC (the decoder produced uint8, so this is exact)
            images[k[len("observation.images."):]] = (v.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return {"joint_positions": q, "images": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="…/checkpoints/<step>/pretrained_model, or a Hub model repo id")
    ap.add_argument("--jello-repo", default=DEFAULT_JELLO_REPO,
                    help="JELLO_ARM_FRANKA checkout (branch il-recording) providing scripts/act_policy_server.py")
    ap.add_argument("--fps", type=float, default=None,
                    help="policy rate; default: from the training dataset's meta/info.json, else the eval dataset's")
    ap.add_argument("--repo-id", default=REAL_REPO_ID)
    ap.add_argument("--root", default=str(REAL_ROOT))
    ap.add_argument("--episodes", type=int, nargs="+", default=list(HELDOUT))
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-queries", type=int, default=0, help="debug: stop each episode after N query points")
    ap.add_argument("--out", default=None, help="JSON result file")
    a = ap.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    jello = Path(a.jello_repo).expanduser().resolve()
    sys.path.insert(0, str(jello))
    sys.path.insert(0, str(jello / "scripts"))
    from act_policy_server import LeRobotACT

    ckpt = Path(a.ckpt).expanduser()
    if not ckpt.exists():  # not a local directory: treat it as a Hub model repo id
        from huggingface_hub import snapshot_download

        ckpt = Path(snapshot_download(a.ckpt, repo_type="model"))
    ckpt = ckpt.resolve()
    fps = a.fps
    if fps is None:
        try:
            train_root = Path(json.loads((ckpt / "train_config.json").read_text())["dataset"]["root"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            train_root = None
        if train_root is None or not (train_root / "meta" / "info.json").exists():
            fps = json.loads((Path(a.root).expanduser() / "meta" / "info.json").read_text())["fps"]
    policy = LeRobotACT(ckpt, fps=fps, device=a.device)
    policy.warm_up()
    H = policy.chunk
    ds = LeRobotDataset(a.repo_id, root=a.root)
    ep_idx = np.asarray(ds.hf_dataset["episode_index"])
    frame_idx = np.asarray(ds.hf_dataset["frame_index"])
    actions_all = np.stack(ds.hf_dataset.with_format("numpy")["action"]).astype(np.float64)
    torch.backends.cudnn.benchmark = False

    per_ep, pooled_j, pooled_g = {}, [], []
    horizon_j_sum, horizon_g_sum, horizon_n = np.zeros(H), np.zeros(H), np.zeros(H)
    t0 = time.time()
    for e in a.episodes:
        rows = np.nonzero(ep_idx == e)[0]
        rows = rows[np.argsort(frame_idx[rows])]
        rec = actions_all[rows]  # (N, 8) recorded actions of the episode, frame order
        N = len(rows)
        q_points = list(range(0, N, a.stride))
        if a.max_queries:
            q_points = q_points[: a.max_queries]
        ej, eg = [], []
        for t in q_points:
            pred = policy.infer(to_request(ds[int(rows[t])]))["actions"].astype(np.float64)  # (H, 8)
            n = min(H, N - t)
            d = np.abs(pred[:n] - rec[t : t + n])
            j = d[:, :7].mean(axis=1) * 1e3  # mrad, mean over joints, per chunk step
            g = d[:, 7]
            ej.append(j)
            eg.append(g)
            horizon_j_sum[:n] += j
            horizon_g_sum[:n] += g
            horizon_n[:n] += 1
        ej, eg = np.concatenate(ej), np.concatenate(eg)
        pooled_j.append(ej)
        pooled_g.append(eg)
        per_ep[e] = {"frames": int(N), "query_points": len(q_points), "compared_steps": int(len(ej)),
                     "joint_l1_mrad": float(ej.mean()), "gripper_l1": float(eg.mean())}
        print(f"episode {e}: {N} frames, {len(q_points)} queries, joint L1 {ej.mean():.2f} mrad, gripper L1 "
              f"{eg.mean():.4f}  ({time.time() - t0:.0f} s)", flush=True)

    pj, pg = np.concatenate(pooled_j), np.concatenate(pooled_g)
    res = {
        "ckpt": str(ckpt), "dataset": {"repo_id": a.repo_id, "root": a.root}, "episodes": a.episodes,
        "stride": a.stride, "chunk": H,
        "joint_l1_mrad": float(np.mean([v["joint_l1_mrad"] for v in per_ep.values()])),
        "gripper_l1": float(np.mean([v["gripper_l1"] for v in per_ep.values()])),
        "pooled": {"joint_l1_mrad": float(pj.mean()), "gripper_l1": float(pg.mean())},
        "per_episode": {str(k): v for k, v in per_ep.items()},
        "per_horizon": {"step": [0, 9, 24, 49, 74, 99],
                        "joint_l1_mrad": [float(horizon_j_sum[k] / horizon_n[k]) for k in (0, 9, 24, 49, 74, 99)],
                        "gripper_l1": [float(horizon_g_sum[k] / horizon_n[k]) for k in (0, 9, 24, 49, 74, 99)]},
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps({k: res[k] for k in ("joint_l1_mrad", "gripper_l1", "pooled", "per_horizon")}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
