#!/usr/bin/env python3
"""Create deterministic source-stratified train/holdout lists without case leakage."""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path


def patient_key(path: Path, root: Path) -> str:
    """Best-effort grouping of longitudinal/series variants from the filename."""
    rel = path.relative_to(root)
    stem = path.stem
    # WholeBodyLesion contains BL/FU and window variants from the same patient.
    stem = re.sub(r"_(?:BL|FU)_.*$", "", stem, flags=re.IGNORECASE)
    # Common trailing scan/series/crop identifiers.
    stem = re.sub(r"(?:__[^_]+|_\d{2}_\d{2}_\d{3}-\d{3})$", "", stem)
    source = rel.parts[0] if len(rel.parts) > 1 else "."
    return f"{source}/{stem}"


def stable_uniform(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("splits"))
    p.add_argument("--holdout-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if not 0 < args.holdout_fraction < 0.5:
        raise SystemExit("--holdout-fraction must be between 0 and 0.5")
    groups = defaultdict(list)
    for path in sorted(args.root.rglob("*.npz")):
        groups[patient_key(path, args.root)].append(path.relative_to(args.root).with_suffix(""))
    train, holdout = [], []
    for key, members in sorted(groups.items()):
        target = holdout if stable_uniform(key, args.seed) < args.holdout_fraction else train
        target.extend(members)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text("\n".join(map(str, train)) + "\n")
    (args.output_dir / "holdout.txt").write_text("\n".join(map(str, holdout)) + "\n")
    print(f"groups={len(groups)} train={len(train)} holdout={len(holdout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

