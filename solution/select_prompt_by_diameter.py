#!/usr/bin/env python3
"""Select a challenge-format prediction source by RECIST physical diameter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roi3d.geometry import recist_geometries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--middle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-mm", type=float, default=25.0)
    parser.add_argument("--upper-mm", type=float, default=50.0)
    args = parser.parse_args()
    if not 0 < args.lower_mm < args.upper_mm:
        raise SystemExit("require 0 < lower-mm < upper-mm")
    args.output.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.inputs.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZ inputs below {args.inputs}")
    middle_count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as case:
            recist = case["recist"]
            spacing = np.asarray(case["spacing"], dtype=np.float64)
        with np.load(args.default / path.name, allow_pickle=False) as pred:
            default = pred["segs"]
        with np.load(args.middle / path.name, allow_pickle=False) as pred:
            middle = pred["segs"]
        output = default.copy()
        for geometry in recist_geometries(recist, spacing):
            if args.lower_mm <= geometry.diameter_true_mm < args.upper_mm:
                lesion_id = geometry.lesion_id
                output[output == lesion_id] = 0
                selected = middle == lesion_id
                output[selected & (output == 0)] = lesion_id
                middle_count += 1
        np.savez_compressed(args.output / path.name, segs=output.astype(np.uint8))
    print(f"wrote {len(paths)} cases; selected middle prompt for "
          f"{middle_count} lesions in [{args.lower_mm}, {args.upper_mm}) mm")


if __name__ == "__main__":
    main()
