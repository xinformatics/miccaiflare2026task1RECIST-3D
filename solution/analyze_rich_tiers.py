#!/usr/bin/env python3
"""Evaluate runtime-tiered four-way/three-way/Tiny/MoE policies."""

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
PREDICTIONS = {
    "four": ROOT / "artifacts/public-roi3d-organ-moe-e65-router-4way",
    "three": ROOT / "artifacts/calibration_moe_small_tiny_grid/w052_024_024",
    "tiny": ROOT / "artifacts/public-cpu-moe-tinybox-sweep/w0.5700",
    "moe": ROOT / "artifacts/public-roi3d-organ-moe-e65-router-standalone",
}


def load_seg(root: Path, name: str) -> np.ndarray:
    with np.load(root / name, allow_pickle=False) as data:
        return data["segs"]


def main() -> int:
    four_caps = (20, 30, 40, 50, 60, 80, 100, 10_000)
    medium_caps = (60, 80, 100, 140, 200, 10_000)
    policies: dict[str, tuple[int, int, str]] = {}
    for four_cap in four_caps:
        policies[f"four{four_cap}_three_all"] = (four_cap, 10_000, "three")
        policies[f"four{four_cap}_tiny_all"] = (four_cap, 10_000, "tiny")
        for medium_cap in medium_caps:
            if medium_cap > four_cap:
                policies[f"four{four_cap}_three{medium_cap}_moe"] = (
                    four_cap, medium_cap, "three"
                )
                policies[f"four{four_cap}_tiny{medium_cap}_moe"] = (
                    four_cap, medium_cap, "tiny"
                )

    values = {name: [] for name in policies}
    for path in sorted(GT.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            gt = data["gts"]
            spacing = np.asarray(data["spacing"], dtype=np.float64)
        cost = slab_slices(path, 1.6)
        predictions = {key: load_seg(root, path.name)
                       for key, root in PREDICTIONS.items()}
        for lesion_id in np.unique(gt):
            if lesion_id == 0:
                continue
            target = gt == lesion_id
            scores = {}
            for source, prediction in predictions.items():
                mask = prediction == lesion_id
                scores[source] = (dice(target, mask), nsd(target, mask, spacing))
            for name, (four_cap, medium_cap, medium) in policies.items():
                source = ("four" if cost <= four_cap else
                          medium if cost <= medium_cap else "moe")
                values[name].append(scores[source])

    summary = {
        name: {
            "DSC": float(np.mean([score[0] for score in scores])),
            "NSD": float(np.mean([score[1] for score in scores])),
        }
        for name, scores in values.items()
    }
    ranked = dict(sorted(summary.items(), key=lambda item: item[1]["DSC"], reverse=True))
    print(json.dumps(ranked, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
