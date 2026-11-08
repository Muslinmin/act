#!/usr/bin/env python3
"""Write the co-training paper cells (alpha sweep @350, quantity sweep, real-only, sim-only) as exclude lists over the
426-slot real50_mimic376 dataset, plus the queue file for queue_cells.sh.

Sim subsets = the first N Mimic demos in coverage-checklist order (cover_entry, farthest-point order, so every prefix is
spread over the quadrants; nested). First 47 == the old mimic47 set. R25 = every 2nd real episode.
Usage: make_cells.py <out_dir>
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
slots = json.loads((HERE.parent / "mappings/real50_mimic376_mapping.json").read_text())["slots"]
real = [s["slot"] for s in slots if s["source"] == "real"]
sim = [s["slot"] for s in sorted((s for s in slots if s["source"] == "mimic"), key=lambda s: s["cover_entry"])]
assert len(real) == 50 and len(sim) == 376

# (job name, alpha, n_real kept, n_sim kept), in training order
CELLS = [
    ("S350_a090", 0.9, 50, 350), ("R50", 0.0, 50, 0), ("S350_a050", 0.5, 50, 350), ("S350_a100", 1.0, 0, 350),
    ("S350_a070", 0.7, 50, 350), ("S350_a095", 0.95, 50, 350), ("S350_a080", 0.8, 50, 350),
    ("S350_a030", 0.3, 50, 350), ("S200_a090", 0.9, 50, 200), ("S100_a090", 0.9, 50, 100),
    ("S050_a090", 0.9, 50, 50), ("R25", 0.0, 25, 0), ("S050_a100", 1.0, 0, 50), ("S200_a100", 1.0, 0, 200),
    ("S050_a050", 0.5, 50, 50),
]
lines = []
summary = {}
for name, a, nr, ns in CELLS:
    keep_real = real if nr == 50 else (real[::2] if nr == 25 else [])
    assert len(keep_real) == nr
    keep = set(keep_real) | set(sim[:ns])
    excl = sorted(set(range(len(slots))) - keep)
    (OUT / f"excl_{name}.txt").write_text(" ".join(map(str, excl)) + "\n")
    job = f"act_cot_{name}_b40"
    lines.append(f"{job} {a} excl_{name}.txt")
    summary[job] = {"alpha_sim": a, "n_real": nr, "n_sim": ns, "n_excluded": len(excl),
                    "kept_real_slots": keep_real, "kept_sim_slots": sim[:ns]}
(OUT / "cells.txt").write_text("# <job> <alpha_sim> <exclude file>; delete or '#' a line that has not started to drop it\n"
                               + "\n".join(lines) + "\n")
(OUT / "cells.json").write_text(json.dumps(summary, indent=1))
print("\n".join(lines))
