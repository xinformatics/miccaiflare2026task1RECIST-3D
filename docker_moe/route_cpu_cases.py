#!/usr/bin/env python3
"""Select cases whose bounded SAM propagation fits the CPU time budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def recist_diameter(mask: np.ndarray, subtract_thickness: float = 2.0) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 0.0
    coords = np.stack((xs, ys), axis=1).astype(np.float32)
    centroid = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - centroid, full_matrices=False)
    projections = (coords - centroid) @ vt[0]
    diameter = float(projections.max() - projections.min())
    return max(0.0, diameter - subtract_thickness)


def slab_slices(path: Path, slab_scale: float) -> int:
    with np.load(path, allow_pickle=False) as case:
        recist = case["recist"]
        spacing = np.asarray(case["spacing"], dtype=np.float64)
    depth = recist.shape[0]
    total = 0
    for lesion_id in np.unique(recist):
        if lesion_id == 0:
            continue
        z_indices = np.unique(np.where(recist == lesion_id)[0])
        if len(z_indices) != 1:
            continue
        z_mid = int(z_indices[0])
        diameter = recist_diameter(recist[z_mid] == lesion_id)
        z_span = diameter * (((spacing[0] + spacing[1]) / 2.0) / spacing[2])
        half_z = int(z_span * slab_scale / 2.0)
        z_lo = max(0, z_mid - half_z)
        z_hi = min(depth - 1, z_mid + half_z)
        total += z_hi - z_lo + 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slab-scale", type=float, default=1.6)
    parser.add_argument("--min-total-slices", type=int, default=-1,
                        help="exclusive lower bound for selected cases")
    parser.add_argument("--max-total-slices", type=int, default=40)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    selected = 0
    for path in sorted(args.input.rglob("*.npz")):
        cost = slab_slices(path, args.slab_scale)
        if args.min_total_slices < cost <= args.max_total_slices:
            (args.output / path.name).symlink_to(path.resolve())
            selected += 1
        else:
            print(
                f"[cpu-route] {path.name}: excluded ({cost} SAM slab slices; "
                f"need {args.min_total_slices} < slices <= {args.max_total_slices})"
            )
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
