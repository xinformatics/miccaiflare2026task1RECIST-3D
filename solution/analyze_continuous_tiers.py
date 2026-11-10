#!/usr/bin/env python3
"""Evaluate Small/Tiny continuous-MoE case-routing policies on public data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docker_moe.route_cpu_cases import slab_slices
from solution.evaluate import dice, nsd


GT = ROOT / "dataset/validation_npz/validation_public_npz"
MOE = ROOT / "artifacts/public-roi3d-organ-moe-e65-router-standalone"
SMALL = ROOT / "artifacts/public-cpu-moe-smallbox-sweep/w0.5700"
TINY = ROOT / "artifacts/public-cpu-moe-tinybox-sweep/w0.5700"


def seg(root: Path, name: str) -> np.ndarray:
    with np.load(root / name, allow_pickle=False) as data:
        return data["segs"]


def main() -> int:
    thresholds = (50, 60, 80, 100, 140, 200, 10_000)
    names = ["current_small40_moe"]
    names += [f"small40_tiny{threshold}_moe" for threshold in thresholds]
    names += [f"tiny{threshold}_moe" for threshold in thresholds]
    values = {name: [] for name in names}

    for path in sorted(GT.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            gt = data["gts"]
            spacing = np.asarray(data["spacing"], dtype=np.float64)
        cost = slab_slices(path, 1.6)
        predictions = {
            "moe": seg(MOE, path.name),
            "small": seg(SMALL, path.name),
            "tiny": seg(TINY, path.name),
        }
        selected = {
            "current_small40_moe": predictions["small"] if cost <= 40 else predictions["moe"]
        }
        for threshold in thresholds:
            selected[f"small40_tiny{threshold}_moe"] = (
                predictions["small"] if cost <= 40 else
                predictions["tiny"] if cost <= threshold else predictions["moe"]
            )
            selected[f"tiny{threshold}_moe"] = (
                predictions["tiny"] if cost <= threshold else predictions["moe"]
            )
        for lesion_id in np.unique(gt):
            if lesion_id == 0:
                continue
            target = gt == lesion_id
            for name, prediction in selected.items():
                mask = prediction == lesion_id
                values[name].append((dice(target, mask), nsd(target, mask, spacing)))

    summary = {
        name: {
            "DSC": float(np.mean([score[0] for score in scores])),
            "NSD": float(np.mean([score[1] for score in scores])),
        }
        for name, scores in values.items()
    }
    print(json.dumps(dict(sorted(summary.items(), key=lambda item: item[1]["DSC"], reverse=True)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
