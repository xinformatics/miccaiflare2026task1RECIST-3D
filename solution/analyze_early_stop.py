#!/usr/bin/env python3
"""Audit whether empty-mask early stopping would truncate later predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def would_reappear(active: np.ndarray, start: int, step: int, patience: int) -> bool:
    empty = 0
    z = start
    while 0 <= z < len(active):
        if active[z]:
            empty = 0
        else:
            empty += 1
            if empty >= patience:
                tail = active[z + step :: step] if step > 0 else active[:z][::-1]
                return bool(np.any(tail))
        z += step
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--patience", nargs="+", type=int, default=[1, 2, 3])
    args = p.parse_args()
    if any(value < 1 for value in args.patience):
        raise SystemExit("--patience must be positive")

    cases = lesions = empty_lesions = 0
    reappearances = {value: 0 for value in args.patience}
    examples = {value: [] for value in args.patience}
    for pred_path in sorted(args.predictions.glob("*.npz")):
        input_path = args.inputs / pred_path.name
        if not input_path.exists():
            continue
        with np.load(input_path, allow_pickle=False) as source, np.load(
            pred_path, allow_pickle=False
        ) as pred:
            recist, segs = source["recist"], pred["segs"]
            cases += 1
            for label in np.unique(recist):
                if label == 0:
                    continue
                lesions += 1
                z_prompt = int(np.where(recist == label)[0][0])
                active = np.any(segs == label, axis=(1, 2))
                if not np.any(active):
                    empty_lesions += 1
                for direction in (-1, 1):
                    for patience in args.patience:
                        if would_reappear(active, z_prompt, direction, patience):
                            reappearances[patience] += 1
                            if len(examples[patience]) < 20:
                                examples[patience].append(
                                    (pred_path.name, int(label), direction)
                                )

    for patience in args.patience:
        print({
            "cases": cases,
            "lesions": lesions,
            "directions": lesions * 2,
            "patience": patience,
            "empty_lesions": empty_lesions,
            "reappearing_directions": reappearances[patience],
            "examples": examples[patience],
        })
    return 1 if any(reappearances.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
