#!/usr/bin/env python3
"""Crop existing predictions to a physical slab for fast what-if analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analyze_slab import recist_diameter


def half_slices(recist_2d, spacing, scale, subtract_thickness):
    diameter = recist_diameter(recist_2d, subtract_thickness)
    z_span = diameter * ((float(spacing[0]) + float(spacing[1])) / 2) / float(spacing[2])
    return int(z_span * scale / 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--scale", type=float, default=1.2)
    p.add_argument("--max-planned-steps", type=int, default=120)
    p.add_argument("--fallback-scale", type=float, default=1.0)
    p.add_argument("--subtract-thickness", type=float, default=2.0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    changed_cases = 0
    for pred_path in sorted(args.predictions.glob("*.npz")):
        input_path = args.inputs / pred_path.name
        with np.load(input_path, allow_pickle=False) as source, np.load(
            pred_path, allow_pickle=False
        ) as prediction:
            recist, spacing = source["recist"], source["spacing"]
            labels = [int(x) for x in np.unique(recist) if x != 0]
            halves = {}
            for label in labels:
                z_mid = int(np.where(recist == label)[0][0])
                halves[label] = half_slices(
                    recist[z_mid] == label,
                    spacing,
                    args.scale,
                    args.subtract_thickness,
                )
            planned = sum(2 * (half + 1) for half in halves.values())
            scale = (
                args.fallback_scale
                if args.max_planned_steps and planned > args.max_planned_steps
                else args.scale
            )

            segs = prediction["segs"].copy()
            before = int(np.count_nonzero(segs))
            for label in labels:
                z_mid = int(np.where(recist == label)[0][0])
                half = half_slices(
                    recist[z_mid] == label,
                    spacing,
                    scale,
                    args.subtract_thickness,
                )
                z_lo, z_hi = max(0, z_mid - half), min(len(segs) - 1, z_mid + half)
                outside = np.ones(len(segs), dtype=bool)
                outside[z_lo : z_hi + 1] = False
                view = segs[outside]
                view[view == label] = 0
                segs[outside] = view
            changed_cases += int(np.count_nonzero(segs) != before)
            np.savez_compressed(args.output / pred_path.name, segs=segs.astype(np.uint8))

    print(f"wrote {len(list(args.output.glob('*.npz')))} cases; changed={changed_cases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
