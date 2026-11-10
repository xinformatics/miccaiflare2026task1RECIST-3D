#!/usr/bin/env python3
"""Validate challenge output structure and report accuracy-risk warnings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("predictions", type=Path)
    args = p.parse_args()
    inputs = sorted(args.inputs.glob("*.npz"))
    errors = []
    warnings = []
    extras = {x.name for x in args.predictions.glob("*.npz")} - {x.name for x in inputs}
    for src in inputs:
        dst = args.predictions / src.name
        if not dst.exists():
            errors.append(f"missing: {src.name}")
            continue
        with np.load(src, allow_pickle=False) as a, np.load(dst, allow_pickle=False) as b:
            if "segs" not in b.files:
                errors.append(f"{src.name}: missing 'segs' key")
                continue
            segs = b["segs"]
            if segs.shape != a["imgs"].shape:
                errors.append(f"{src.name}: shape {segs.shape} != {a['imgs'].shape}")
            if segs.dtype != np.uint8:
                errors.append(f"{src.name}: dtype {segs.dtype} != uint8")
            expected = set(int(x) for x in np.unique(a["recist"]) if x != 0)
            present = set(int(x) for x in np.unique(segs) if x != 0)
            missing_ids = expected - present
            if missing_ids:
                # A missed lesion is scored as zero but does not invalidate the
                # submission file. Keep format failures and accuracy risks distinct.
                warnings.append(
                    f"{src.name}: empty prediction for prompt IDs {sorted(missing_ids)}"
                )
    errors.extend(f"extra prediction: {x}" for x in sorted(extras))
    if errors:
        print("INVALID")
        print("\n".join(errors[:100]))
        return 1
    print(f"VALID: {len(inputs)} flat NPZ predictions")
    if warnings:
        print(f"WARNINGS: {len(warnings)}")
        print("\n".join(warnings[:100]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
