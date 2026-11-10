#!/usr/bin/env python3
"""Measure GT coverage and ideal Dice ceilings for physical z-slab choices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def recist_diameter(mask: np.ndarray, subtract_thickness: float) -> float:
    ys, xs = np.where(mask)
    xy = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[0]
    return max(float(projection.max() - projection.min()) - subtract_thickness, 1.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--scales", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.4, 1.6])
    p.add_argument("--subtract-thickness", type=float, default=2.0)
    args = p.parse_args()

    coverage = {scale: [] for scale in args.scales}
    full = {scale: 0 for scale in args.scales}
    ideal_dice = {scale: [] for scale in args.scales}
    extent_ratios = []
    lesions = 0

    for path in sorted(args.root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            gt = data["gts"]
            recist = data["recist"]
            spacing = data["spacing"].astype(float)
            inplane_spacing = float((spacing[0] + spacing[1]) / 2)
            for label in np.unique(recist):
                if label == 0:
                    continue
                z_prompt = np.flatnonzero(np.any(recist == label, axis=(1, 2)))
                if len(z_prompt) != 1:
                    continue
                z_mid = int(z_prompt[0])
                diameter_px = recist_diameter(
                    recist[z_mid] == label, args.subtract_thickness
                )
                lesion = gt == label
                z_gt = np.flatnonzero(np.any(lesion, axis=(1, 2)))
                if len(z_gt) == 0:
                    continue
                physical_diameter = diameter_px * inplane_spacing
                physical_extent = (int(z_gt[-1]) - int(z_gt[0]) + 1) * float(spacing[2])
                extent_ratios.append(physical_extent / physical_diameter)
                total = int(lesion.sum())
                for scale in args.scales:
                    z_span = diameter_px * inplane_spacing / float(spacing[2])
                    half_z = int(z_span * scale / 2)
                    z_lo = max(0, z_mid - half_z)
                    z_hi = min(gt.shape[0] - 1, z_mid + half_z)
                    kept = int(lesion[z_lo : z_hi + 1].sum())
                    fraction = kept / total
                    coverage[scale].append(fraction)
                    full[scale] += kept == total
                    ideal_dice[scale].append(2 * fraction / (1 + fraction))
                lesions += 1

    ratios = np.asarray(extent_ratios)
    print(f"lesions={lesions}")
    print(
        "physical_extent/RECIST_diameter percentiles:",
        {q: round(float(np.percentile(ratios, q)), 3) for q in (50, 75, 90, 95, 99, 100)},
    )
    print("scale  full_containment  mean_voxel_coverage  ideal_mean_DSC_ceiling")
    for scale in args.scales:
        print(
            f"{scale:>4.1f}  {full[scale] / lesions:>16.3%}  "
            f"{np.mean(coverage[scale]):>19.5f}  {np.mean(ideal_dice[scale]):>22.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
