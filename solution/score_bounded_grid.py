#!/usr/bin/env python3
"""Score bounded-3-point weight candidates with fixed medium/fallback tiers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docker_moe.route_cpu_cases import slab_slices
from solution.evaluate import dice, nsd


def load(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data["segs"]


def main() -> int:
    gt_root = ROOT / "dataset/validation_npz/validation_public_npz"
    grid_root = ROOT / "artifacts/public-bounded3point-calibration/grid"
    fixed_root = ROOT / "artifacts/public-cpu-rich-router-integration"
    fallback_root = ROOT / "artifacts/public-roi3d-organ-moe-e65-router-standalone"
    candidates = sorted(path for path in grid_root.glob("w*") if path.is_dir())
    values = {path.name: [] for path in candidates}

    for gt_path in sorted(gt_root.glob("*.npz")):
        with np.load(gt_path, allow_pickle=False) as data:
            gt = data["gts"]
            spacing = np.asarray(data["spacing"], dtype=np.float64)
        cost = slab_slices(gt_path, 1.6)
        cheap = cost <= 40
        predictions = ({path.name: load(path / gt_path.name) for path in candidates}
                       if cheap else {})
        fixed = (None if cheap else
                 load((fixed_root if cost <= 120 else fallback_root) / gt_path.name))
        for lesion_id in np.unique(gt):
            if lesion_id == 0:
                continue
            target = gt == lesion_id
            if cheap:
                for name, prediction in predictions.items():
                    mask = prediction == lesion_id
                    values[name].append((dice(target, mask), nsd(target, mask, spacing)))
            else:
                mask = fixed == lesion_id
                score = (dice(target, mask), nsd(target, mask, spacing))
                for name in values:
                    values[name].append(score)

    summary = {
        name: {
            "DSC": float(np.mean([score[0] for score in scores])),
            "NSD": float(np.mean([score[1] for score in scores])),
        }
        for name, scores in values.items()
    }
    print(json.dumps(dict(sorted(summary.items(), key=lambda x: x[1]["DSC"], reverse=True)),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
