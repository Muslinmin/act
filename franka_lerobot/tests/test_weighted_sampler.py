"""CPU tests of the weighted co-training sampler in franka_lerobot/scripts/train_weighted.py (no training, no GPU).

    conda activate franka_il
    HF_HUB_OFFLINE=1 python -m pytest -q -s franka_lerobot/tests/test_weighted_sampler.py
    # datasets are read from $FRANKA_DATA_ROOT (default ~/bc_data/lerobot/franka); missing ones are skipped

The datasets are opened exactly as the launchers open them (`--dataset.episodes` = all but 10/25/40), so the sampled
index range is the one stock lerobot 0.4.1 training uses.
"""

from __future__ import annotations

import itertools
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import chisquare
from torch.utils.data import BatchSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import train_weighted as tw  # noqa: E402

DATA_ROOT = Path(os.environ.get("FRANKA_DATA_ROOT", "~/bc_data/lerobot/franka")).expanduser()
MAPPINGS = HERE.parent / "mappings"
HELDOUT = (10, 25, 40)  # launcher held-out list (dataset indices); stock 0.4.1 still loads and samples them
DATASETS = {  # name -> (n episodes, mapping file or None for real-only)
    "cubes_20261014_30hz": (50, None),
    "cubes_20261014_30hz_simtele24": (50, "simtele24_mapping.json"),
    "cubes_20261014_30hz_real50_mimic47": (97, "real50_mimic47_mapping.json"),
    "cubes_20261014_30hz_real50_mimic176": (226, "real50_mimic176_mapping.json"),
}
COTRAIN = [k for k, v in DATASETS.items() if v[1]]
BATCH, N_BATCHES = 40, 10_000  # the training batch size; the task's 10k batches
TOL_FRACTION = 0.01  # task: realised sim fraction within +-1 % (absolute) of alpha
TOL_REL = 0.05  # task: per-episode draws within 5 % of the length-proportional expectation
SIGMAS = 5.0  # draws for (c) are sized so that TOL_REL is >= SIGMAS binomial sigmas for the shortest episode
SEED = 1000  # lerobot's default --seed, the one the runs use


@lru_cache(maxsize=None)
def load(name):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = DATA_ROOT / name
    if not (root / "meta/info.json").exists():
        pytest.skip(f"no dataset at {root}")
    n_ep, mapping = DATASETS[name]
    eps = [i for i in range(n_ep) if i not in HELDOUT]
    ds = LeRobotDataset(f"franka/{name}", root=root, episodes=eps)
    return ds, (MAPPINGS / mapping if mapping else None)


def sampler_for(name, alpha, exclude=()):
    ds, mapping = load(name)
    return tw.build_sampler(ds, alpha, mapping, SEED, exclude)


@pytest.mark.parametrize("name", list(DATASETS))
def test_bounds_mapping_and_stock_pool(name):
    """meta.episodes bounds agree with hf_dataset; the pools = every frame stock training draws (incl. 10/25/40)."""
    ds, mapping = load(name)
    _, fr, to = tw.episode_bounds(ds)  # asserts inside
    ep_is_sim = tw.load_slot_sources(mapping, fr, to)
    assert not ep_is_sim[list(HELDOUT)].any(), "a held-out slot is not real"
    alpha = 0.5 if ep_is_sim.any() else 0.0  # a real-only set has no sim pool
    w, is_sim, info = tw.frame_weights(fr, to, ep_is_sim, alpha)
    assert info["n_real_frames"] + info["n_sim_frames"] == len(ds) == ds.meta.total_frames
    held = sum(int(to[e] - fr[e]) for e in HELDOUT)
    assert (w[np.concatenate([np.arange(fr[e], to[e]) for e in HELDOUT])] > 0).all()
    assert abs(w.sum() - 1.0) < 1e-9
    print(f"\n{name}: {len(ds)} frames = real {info['n_real_frames']} ({info['n_real_episodes']} eps, incl. held-out "
          f"{held} frames) + sim {info['n_sim_frames']} ({info['n_sim_episodes']} eps); stock sim share "
          f"{info['stock_sim_share']:.4f}")


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.9])
@pytest.mark.parametrize("name", COTRAIN)
def test_a_realised_sim_fraction(name, alpha):
    """(a) 10k batches of 40 from the sampler's own iterator: sim fraction within +-1 % of alpha."""
    sampler, _ = sampler_for(name, alpha)
    batches = itertools.islice(itertools.chain.from_iterable(itertools.repeat(BatchSampler(sampler, BATCH, False))),
                               N_BATCHES)
    idx = np.fromiter(itertools.chain.from_iterable(batches), dtype=np.int64)
    frac = float(sampler.is_sim[idx].mean())
    sigma = (alpha * (1 - alpha) / len(idx)) ** 0.5
    print(f"\n{name} alpha {alpha}: {len(idx)} draws, realised sim fraction {frac:.4f} "
          f"(|diff| {abs(frac - alpha):.4f}, binomial sigma {sigma:.4f})")
    assert abs(frac - alpha) <= TOL_FRACTION
    if alpha == 0.0:
        assert frac == 0.0


@pytest.mark.parametrize("alpha", [0.25, 0.9])
@pytest.mark.parametrize("name", COTRAIN)
def test_c_per_episode_draws_proportional_to_length(name, alpha):
    """(c) Within each source, draws per episode are proportional to the episode length.

    Draws use the sampler's own weights and the same torch.multinomial call as its __iter__. The number of draws is
    chosen so that the 5 % tolerance is SIGMAS binomial sigmas for the shortest episode of the smaller expected pool.
    """
    sampler, info = sampler_for(name, alpha)
    ds, mapping = load(name)
    _, fr, to = tw.episode_bounds(ds)
    ep_is_sim = tw.load_slot_sources(mapping, fr, to)
    lens = (to - fr).astype(np.float64)
    need = 0
    for src, share in ((False, 1 - alpha), (True, alpha)):
        sel = ep_is_sim == src
        p_min = lens[sel].min() / lens[sel].sum()  # shortest episode's share within its source
        need = max(need, (SIGMAS / TOL_REL) ** 2 / (p_min * share))
    n = int(np.ceil(need))
    g = torch.Generator().manual_seed(SEED)
    frame_ep = np.repeat(np.arange(len(lens)), (to - fr))
    counts = np.zeros(len(lens), dtype=np.int64)
    chunk = 2_000_000
    for k in range(0, n, chunk):
        idx = torch.multinomial(sampler.weights, min(chunk, n - k), True, generator=g).numpy()
        counts += np.bincount(frame_ep[idx], minlength=len(lens))
    for src, label in ((False, "real"), (True, "sim")):
        sel = ep_is_sim == src
        c = counts[sel]
        exp = c.sum() * lens[sel] / lens[sel].sum()
        rel = np.abs(c - exp) / exp
        p = chisquare(c, exp).pvalue
        print(f"\n{name} alpha {alpha} {label}: {sel.sum()} eps, {c.sum()} draws, max rel err {rel.max():.4f} "
              f"(ep {np.flatnonzero(sel)[rel.argmax()]}), chi2 p {p:.3f}")
        assert rel.max() < TOL_REL
    assert counts.sum() == n


def test_c_within_source_uniform_per_frame():
    """Every frame of one source has the same weight (uniform within source), checked on all co-training sets."""
    for name in COTRAIN:
        sampler, info = sampler_for(name, 0.9)
        w = sampler.weights.numpy()
        for src, n_src, share in ((True, info["n_sim_frames"], 0.9), (False, info["n_real_frames"], 0.1)):
            ws = w[sampler.is_sim == src]
            assert np.allclose(ws, share / n_src, rtol=1e-12, atol=0)


def test_real_only_alpha0_is_uniform_over_stock_pool():
    """Real-only dataset, no mapping, alpha 0: uniform over the 66930 frames stock training draws."""
    sampler, info = sampler_for("cubes_20261014_30hz", 0.0)
    w = sampler.weights.numpy()
    assert info["n_sim_frames"] == 0 and len(w) == 66930 and np.allclose(w, 1 / 66930, rtol=1e-12, atol=0)
    with pytest.raises(AssertionError):
        sampler_for("cubes_20261014_30hz", 0.5)  # alpha > 0 needs a sim pool


def test_option_exclude_episodes():
    """OFF-by-default option: excluded episodes get weight 0 and are never drawn."""
    name = "cubes_20261014_30hz_real50_mimic47"
    sampler, info = sampler_for(name, 0.9, exclude=HELDOUT)
    ds, _ = load(name)
    _, fr, to = tw.episode_bounds(ds)
    held = np.concatenate([np.arange(fr[e], to[e]) for e in HELDOUT])
    idx = torch.multinomial(sampler.weights, N_BATCHES * BATCH, True, generator=torch.Generator().manual_seed(SEED))
    assert info["n_excluded_frames"] == len(held)
    assert not np.isin(idx.numpy(), held).any()
    assert abs(sampler.weights.sum().item() - 1.0) < 1e-9


def test_sampler_matches_torch_weighted_random_sampler():
    """CountingWeightedSampler draws the same sequence as torch's WeightedRandomSampler with the same seed."""
    sampler, _ = sampler_for("cubes_20261014_30hz_simtele24", 0.9)
    ref = torch.utils.data.WeightedRandomSampler(sampler.weights, sampler.num_samples, True,
                                                 generator=torch.Generator().manual_seed(SEED))
    sampler.generator.manual_seed(SEED)
    assert list(itertools.islice(iter(sampler), 5000)) == list(itertools.islice(iter(ref), 5000))


def test_accelerate_prepare_keeps_the_sampler():
    """The DataLoader after accelerate.prepare (single process, CPU) still draws from the weighted sampler."""
    from accelerate import Accelerator

    ds, _ = load("cubes_20261014_30hz_simtele24")
    sampler, _ = sampler_for("cubes_20261014_30hz_simtele24", 0.9)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, sampler=sampler, shuffle=False, num_workers=0)
    dl = Accelerator(cpu=True).prepare(dl)
    assert dl.batch_sampler.sampler is sampler
    sampler.generator.manual_seed(SEED)
    got = list(itertools.islice(iter(dl.batch_sampler), 3))
    sampler.generator.manual_seed(SEED)
    exp = list(itertools.islice(iter(BatchSampler(sampler, BATCH, False)), 3))
    assert got == exp
