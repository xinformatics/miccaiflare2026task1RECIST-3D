#!/usr/bin/env python3
"""Screen case-level CPU routing policies from cached public predictions."""

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
SMALL_FUSION = ROOT / "artifacts/public-cpu-moe-smallbox-sweep/w0.5700"
TINY = ROOT / "artifacts/public-tiny-box09-z16"
FULL4 = ROOT / "artifacts/public-roi3d-organ-moe-e65-router-4way"


def load_seg(root: Path, name: str) -> np.ndarray:
    with np.load(root / name, allow_pickle=False) as data:
        return data["segs"]


def combine(moe: np.ndarray, tiny: np.ndarray, mode: str) -> np.ndarray:
    if mode == "moe":
        return moe
    if mode == "tiny":
        return tiny
    if mode == "union_moe_priority":
        result = moe.copy()
        add = (result == 0) & (tiny != 0)
        result[add] = tiny[add]
        return result
    if mode == "union_tiny_priority":
        result = tiny.copy()
        add = (result == 0) & (moe != 0)
        result[add] = moe[add]
        return result
    if mode == "intersection":
        return np.where(moe == tiny, moe, 0).astype(np.uint8)
    raise ValueError(mode)


def main() -> int:
    modes = ("moe", "tiny", "union_moe_priority", "union_tiny_priority", "intersection")
    thresholds = (40, 60, 80, 100, 140, 200)
    scores: dict[str, list[tuple[float, float]]] = {
        "current_t40": [], "full4_t40_upper_bound": []
    }
    for threshold in thresholds:
        for mode in modes:
            scores[f"small40_{mode}_to{threshold}"] = []

    for path in sorted(GT.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            gt = data["gts"]
            spacing = np.asarray(data["spacing"], dtype=np.float64)
        cost = slab_slices(path, 1.6)
        moe = load_seg(MOE, path.name)
        small_fusion = load_seg(SMALL_FUSION, path.name)
        tiny = load_seg(TINY, path.name)
        full4 = load_seg(FULL4, path.name)
        predictions = {
            mode: combine(moe, tiny, mode) for mode in modes
        }
        case_candidates = {
            "current_t40": small_fusion if cost <= 40 else moe,
            "full4_t40_upper_bound": full4 if cost <= 40 else moe,
        }
        for threshold in thresholds:
            for mode in modes:
                name = f"small40_{mode}_to{threshold}"
                if cost <= 40:
                    case_candidates[name] = small_fusion
                elif cost <= threshold:
                    case_candidates[name] = predictions[mode]
                else:
                    case_candidates[name] = moe
        for lesion_id in np.unique(gt):
            if lesion_id == 0:
                continue
            target = gt == lesion_id
            for name, pred in case_candidates.items():
                candidate = pred == lesion_id
                scores[name].append((dice(target, candidate), nsd(target, candidate, spacing)))

    summary = {
        name: {
            "DSC": float(np.mean([item[0] for item in values])),
            "NSD": float(np.mean([item[1] for item in values])),
        }
        for name, values in scores.items()
    }
    ranked = sorted(summary.items(), key=lambda item: item[1]["DSC"], reverse=True)
    print(json.dumps(dict(ranked), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
