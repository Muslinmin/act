#!/usr/bin/env python
"""Stock `lerobot-train` (lerobot 0.4.1) with ONE change: a per-frame weighted co-training sampler.

    python franka_lerobot/scripts/train_weighted.py --alpha-sim 0.9 \
        --slot-mapping franka_lerobot/mappings/real50_mimic47_mapping.json \
        <every usual lerobot-train argument, unchanged>

What changes versus `lerobot-train`
-----------------------------------
Only the DataLoader's sampler. Stock 0.4.1 uses `shuffle=True, sampler=None` for ACT (lerobot_train.py, the
`else` branch of `hasattr(cfg.policy, "drop_n_last_frames")`), i.e. a uniform permutation of
`range(len(dataset))` per epoch. Here the DataLoader gets a `torch.utils.data.WeightedRandomSampler`
(replacement=True, seeded, `num_samples = len(dataset)` so one "epoch" has the stock length) with per-frame weights

    w = (1 - ALPHA_SIM) / N_real   for a frame of a real slot
    w =      ALPHA_SIM  / N_sim    for a frame of a sim / mimic slot
    w = 0                          for a frame of an --exclude-episodes slot (OFF by default)

so the expected sim share of the draws is exactly ALPHA_SIM, and every frame within one source is equally likely
(per-episode draws are proportional to the episode length).

The pools are the frames that STOCK training samples, i.e. rows 0..len(dataset)-1 of `dataset.hf_dataset`. In 0.4.1,
`--dataset.episodes=[...]` does not filter `hf_dataset` (lerobot_dataset.py: `num_frames` returns
`len(self.hf_dataset)`, which holds every episode), so stock training also draws the "held-out" episodes 10/25/40.
They stay in the real pool here, on purpose: the sampler is the only variable versus the earlier runs
(act_cubes_c100, act_cubes_real50_mimic176_b40, ...). `--exclude-episodes 10 25 40` removes them (weight 0).

Everything else is lerobot's own `train()`: the same config parsing, normalisation stats (`dataset.meta.stats`, the
stats of the whole merged dataset, with lerobot's ImageNet override for the cameras), policy, optimizer,
checkpointing and logging. The hook is two scoped monkeypatches on the `lerobot.scripts.lerobot_train` module:
  * `make_dataset`: calls the original, then builds the sampler from the returned dataset;
  * `torch.utils.data.DataLoader`: for the ONE construction whose dataset is that dataset (asserted to be the stock
    `shuffle=True, sampler=None` call) the sampler is put in and `shuffle` set to False; the original class is
    restored right after;
  * `cycle`: asserts that the dataloader after `accelerator.prepare` still draws from this sampler.
The lerobot install is not modified.

Logging: at start the frames/episodes per source, ALPHA_SIM, the stock (natural) sim share, the expected real and
sim samples over the run (batch x steps x (1-alpha) / alpha), the stats source, and a startup check of the realised
sim fraction on an independent generator. During training the cumulative realised sim fraction of the drawn
indices is logged every batch_size x log_freq draws (the sampler runs ahead of the optimizer by the DataLoader
prefetch, num_workers x prefetch_factor batches). `<output_dir>/weighted_sampler.json` records the settings.

Resume: pass the same --alpha-sim / --slot-mapping again. The sampler seed is then seed + resumed step, so the
resumed run does not replay the first draws (stock lerobot does not restore its sampler state either).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

MAPPING_DIR = Path(__file__).resolve().parent.parent / "mappings"
REAL_SOURCE = "real"  # mapping JSON `slots[i].source`; every other value ("sim", "mimic") is a sim source


# ----------------------------------------------------------------------------------------------- pure helpers
def episode_bounds(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(episode_index, from, to) per episode from dataset.meta.episodes, checked against hf_dataset.

    Asserts: the episodes tile rows 0..len(hf_dataset) contiguously in order; hf_dataset.episode_index equals the
    episode of each row; len(dataset) == len(hf_dataset) (the range the stock sampler draws from).
    """
    m = dataset.meta.episodes
    ep = np.asarray(m["episode_index"], dtype=np.int64)
    fr = np.asarray(m["dataset_from_index"], dtype=np.int64)
    to = np.asarray(m["dataset_to_index"], dtype=np.int64)
    dataset._ensure_hf_dataset_loaded()
    hf_ep = np.asarray(dataset.hf_dataset.with_format("numpy")["episode_index"], dtype=np.int64)
    n_rows = len(hf_ep)
    assert np.array_equal(ep, np.arange(len(ep))), "meta.episodes is not episodes 0..N-1 in order"
    assert fr[0] == 0 and to[-1] == n_rows and np.array_equal(fr[1:], to[:-1]), (
        "meta.episodes dataset_from/to_index do not tile hf_dataset contiguously"
    )
    assert (to > fr).all(), "empty episode in meta.episodes"
    expected = np.repeat(ep, to - fr)
    bad = np.flatnonzero(expected != hf_ep)
    assert bad.size == 0, f"hf_dataset.episode_index disagrees with meta.episodes at {bad.size} rows, first {bad[:5]}"
    assert len(dataset) == n_rows, (
        f"len(dataset)={len(dataset)} != len(hf_dataset)={n_rows}: this lerobot filters episodes; the 'stock pool' "
        "assumption of this script (lerobot 0.4.1) does not hold, revisit before training"
    )
    return ep, fr, to


def load_slot_sources(mapping_path: str | Path | None, ep_from: np.ndarray, ep_to: np.ndarray) -> np.ndarray:
    """Bool per episode, True = sim source. No mapping = every episode is real (a real-only dataset).

    The mapping JSON (built with the dataset, copied to franka_lerobot/mappings/) has `slots`: one entry per dataset
    episode index with `slot` and `source`. Checks: one entry per episode, slot == position, and the stored `frames`
    (when present) equal the episode length in meta.episodes.
    """
    n_ep = len(ep_from)
    if mapping_path is None:
        return np.zeros(n_ep, dtype=bool)
    mp = json.loads(Path(mapping_path).read_text())
    slots = mp["slots"]
    assert len(slots) == n_ep, f"{mapping_path}: {len(slots)} slots but the dataset has {n_ep} episodes"
    is_sim = np.zeros(n_ep, dtype=bool)
    for i, s in enumerate(slots):
        assert s["slot"] == i, f"{mapping_path}: slots[{i}].slot = {s['slot']}"
        if "frames" in s:
            n = int(ep_to[i] - ep_from[i])
            assert s["frames"] == n, f"{mapping_path}: slot {i} frames {s['frames']} != dataset episode length {n}"
        is_sim[i] = s["source"] != REAL_SOURCE
    return is_sim


def frame_weights(
    ep_from: np.ndarray, ep_to: np.ndarray, ep_is_sim: np.ndarray, alpha_sim: float, exclude_episodes=()
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-frame weights (float64, sum 1) and per-frame is_sim, plus a summary dict."""
    assert 0.0 <= alpha_sim <= 1.0, f"alpha_sim={alpha_sim} not in [0, 1]"
    n_ep = len(ep_from)
    excl = np.zeros(n_ep, dtype=bool)
    for e in exclude_episodes:
        assert 0 <= e < n_ep, f"exclude episode {e} out of range 0..{n_ep - 1}"
        excl[e] = True
    lens = ep_to - ep_from
    frame_ep = np.repeat(np.arange(n_ep), lens)
    is_sim = ep_is_sim[frame_ep]
    use = ~excl[frame_ep]
    n_real = int((use & ~is_sim).sum())
    n_sim = int((use & is_sim).sum())
    if alpha_sim < 1.0:
        assert n_real > 0, "alpha_sim < 1 but the real pool is empty"
    if alpha_sim > 0.0:
        assert n_sim > 0, "alpha_sim > 0 but the sim pool is empty (real-only dataset or no --slot-mapping?)"
    w = np.zeros(len(frame_ep), dtype=np.float64)
    if n_real:
        w[use & ~is_sim] = (1.0 - alpha_sim) / n_real
    if n_sim:
        w[use & is_sim] = alpha_sim / n_sim
    info = {
        "alpha_sim": alpha_sim,
        "n_frames_total": int(len(frame_ep)),
        "n_real_frames": n_real,
        "n_sim_frames": n_sim,
        "n_real_episodes": int((~ep_is_sim & ~excl).sum()),
        "n_sim_episodes": int((ep_is_sim & ~excl).sum()),
        "excluded_episodes": sorted(int(e) for e in exclude_episodes),
        "n_excluded_frames": int((~use).sum()),
        "stock_sim_share": n_sim / max(1, n_real + n_sim),
    }
    return w, is_sim, info


class CountingWeightedSampler(WeightedRandomSampler):
    """WeightedRandomSampler that draws exactly as torch's (same multinomial call) and logs the realised sim share."""

    def __init__(self, weights, num_samples, is_sim, generator, log_every: int = 0, alpha_sim: float | None = None):
        super().__init__(weights, num_samples, replacement=True, generator=generator)
        self.is_sim = np.asarray(is_sim, dtype=bool)
        self.log_every = int(log_every)
        self.alpha_sim = alpha_sim
        self.n_drawn = 0
        self.n_sim = 0

    def __iter__(self):
        idx = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator).tolist()
        for i in idx:
            self.n_drawn += 1
            self.n_sim += int(self.is_sim[i])
            if self.log_every and self.n_drawn % self.log_every == 0:
                logging.info(
                    f"weighted sampler: {self.n_drawn} draws, realised sim fraction {self.n_sim / self.n_drawn:.4f}"
                    f" (alpha_sim {self.alpha_sim})"
                )
            yield i


def build_sampler(dataset, alpha_sim, slot_mapping, seed, exclude_episodes=(), log_every=0):
    _, fr, to = episode_bounds(dataset)
    ep_is_sim = load_slot_sources(slot_mapping, fr, to)
    w, is_sim, info = frame_weights(fr, to, ep_is_sim, alpha_sim, exclude_episodes)
    g = torch.Generator().manual_seed(int(seed))
    sampler = CountingWeightedSampler(
        torch.as_tensor(w, dtype=torch.double), len(dataset), is_sim, g, log_every=log_every, alpha_sim=alpha_sim
    )
    return sampler, info


def realised_sim_fraction(weights: torch.Tensor, is_sim: np.ndarray, n: int, seed: int) -> float:
    """Sim share of n draws from `weights` on an independent generator (does not touch the training sampler)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.multinomial(weights, n, replacement=True, generator=g).numpy()
    return float(is_sim[idx].mean())


# ------------------------------------------------------------------------------------------------- the hook
def parse_own_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--alpha-sim", type=float, required=True, help="expected share of sim frames per batch, 0..1")
    p.add_argument("--slot-mapping", default=None, help="mapping JSON (slots[i].source); omit for real-only data")
    p.add_argument("--exclude-episodes", type=int, nargs="*", default=[],
                   help="dataset episode indices given weight 0 (OFF by default; stock 0.4.1 trains on all)")
    p.add_argument("--sampler-seed", type=int, default=None, help="default: the lerobot --seed (1000)")
    p.add_argument("--startup-check-batches", type=int, default=10000,
                   help="batches drawn on an independent generator at start to check the realised sim fraction")
    own, rest = p.parse_known_args(argv)
    return own, rest


def install_hooks(own):
    import lerobot.scripts.lerobot_train as lt
    from lerobot.utils.constants import TRAINING_STATE_DIR
    from lerobot.utils.train_utils import load_training_step

    orig_make_dataset, orig_cycle, orig_dl = lt.make_dataset, lt.cycle, torch.utils.data.DataLoader
    state = {}

    def make_dataset(cfg):
        dataset = orig_make_dataset(cfg)
        seed = own.sampler_seed if own.sampler_seed is not None else (cfg.seed if cfg.seed is not None else 0)
        resumed_step = 0
        if cfg.resume:
            resumed_step = load_training_step(Path(cfg.checkpoint_path) / TRAINING_STATE_DIR)
            seed += resumed_step
        sampler, info = build_sampler(
            dataset, own.alpha_sim, own.slot_mapping, seed, own.exclude_episodes,
            log_every=cfg.batch_size * cfg.log_freq if cfg.log_freq > 0 else 0,
        )
        n_check = own.startup_check_batches * cfg.batch_size
        check = realised_sim_fraction(sampler.weights, sampler.is_sim, n_check, seed + 1) if n_check else None
        remaining = cfg.steps - resumed_step
        info.update({
            "slot_mapping": str(own.slot_mapping) if own.slot_mapping else None,
            "dataset_root": str(dataset.root),
            "sampler_seed": seed,
            "resumed_step": resumed_step,
            "batch_size": cfg.batch_size,
            "steps": cfg.steps,
            "expected_real_samples": cfg.batch_size * remaining * (1.0 - own.alpha_sim),
            "expected_sim_samples": cfg.batch_size * remaining * own.alpha_sim,
            "stats_source": f"stock: dataset.meta.stats of {dataset.root} (whole merged dataset"
                            + (", cameras = ImageNet stats)" if cfg.dataset.use_imagenet_stats else ")"),
            "startup_check": {"draws": n_check, "realised_sim_fraction": check},
        })
        info["expected_real_epochs"] = info["expected_real_samples"] / max(1, info["n_real_frames"])
        info["expected_sim_epochs"] = info["expected_sim_samples"] / max(1, info["n_sim_frames"])
        for k, v in info.items():
            logging.info(f"weighted sampler: {k} = {v}")
        if check is not None and abs(check - own.alpha_sim) > 0.01:
            raise RuntimeError(f"startup check: realised sim fraction {check:.4f} vs alpha_sim {own.alpha_sim}")
        if cfg.output_dir is not None and (not cfg.resume):
            out = Path(cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "weighted_sampler.json").write_text(json.dumps(info, indent=1))
        state.update(dataset=dataset, sampler=sampler, info=info)
        return dataset

    class WeightedDataLoader(orig_dl):
        def __init__(self, dataset, *args, **kwargs):
            if "dataset" in state and dataset is state["dataset"]:
                assert kwargs.get("sampler") is None and kwargs.get("shuffle") is True, (
                    f"not the stock DataLoader call (sampler={kwargs.get('sampler')}, shuffle={kwargs.get('shuffle')})"
                )
                kwargs["sampler"], kwargs["shuffle"] = state["sampler"], False
                torch.utils.data.DataLoader = orig_dl  # one construction only
                state["injected"] = True
            super().__init__(dataset, *args, **kwargs)

    def cycle(dataloader):
        assert state.get("injected"), "the weighted sampler was not put into the training DataLoader"
        bs = getattr(dataloader, "batch_sampler", None)
        assert bs is not None and getattr(bs, "sampler", None) is state["sampler"], (
            "after accelerator.prepare the training DataLoader no longer draws from the weighted sampler"
        )
        logging.info("weighted sampler: in use by the prepared training DataLoader")
        return orig_cycle(dataloader)

    lt.make_dataset = make_dataset
    lt.cycle = cycle
    torch.utils.data.DataLoader = WeightedDataLoader

    def uninstall():
        lt.make_dataset, lt.cycle = orig_make_dataset, orig_cycle
        torch.utils.data.DataLoader = orig_dl

    return lt, state, uninstall


def main(argv=None):
    own, rest = parse_own_args(sys.argv[1:] if argv is None else argv)
    if own.slot_mapping:
        p = Path(own.slot_mapping)
        if not p.exists() and (MAPPING_DIR / p.name).exists():
            p = MAPPING_DIR / p.name
        own.slot_mapping = str(p.resolve())
    sys.argv = [sys.argv[0], *rest]  # lerobot's parser (and cfg.validate) read sys.argv
    lt, state, uninstall = install_hooks(own)
    try:
        lt.train()
    finally:
        uninstall()
    return state


if __name__ == "__main__":
    main()
