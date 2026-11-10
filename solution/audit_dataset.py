#!/usr/bin/env python3
"""Stream an NPZ dataset and report schema, source, geometry, and lesion statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    files = sorted(args.root.rglob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    schemas, dtypes, sources = Counter(), Counter(), Counter()
    shapes, spacings, lesions = [], [], []
    bad = []
    per_source = defaultdict(lambda: {"cases": 0, "lesions": 0})
    for i, path in enumerate(files, 1):
        try:
            with np.load(path, allow_pickle=False) as data:
                schemas[tuple(sorted(data.files))] += 1
                img, gt, recist = data["imgs"], data["gts"], data["recist"]
                if img.shape != gt.shape or img.shape != recist.shape:
                    raise ValueError(f"shape mismatch: {img.shape}, {gt.shape}, {recist.shape}")
                ids = [int(x) for x in np.unique(recist) if x != 0]
                missing = set(ids) - set(int(x) for x in np.unique(gt))
                if missing:
                    raise ValueError(f"RECIST IDs absent from GT: {sorted(missing)}")
                source = path.relative_to(args.root).parts[0] if path != args.root else "."
                sources[source] += 1
                per_source[source]["cases"] += 1
                per_source[source]["lesions"] += len(ids)
                dtypes[str(img.dtype)] += 1
                shapes.append(tuple(int(x) for x in img.shape))
                spacings.append(tuple(float(x) for x in data["spacing"]))
                lesions.append(len(ids))
        except Exception as exc:  # an audit should finish and list all corrupt cases
            bad.append({"path": str(path), "error": str(exc)})
        if i % 500 == 0:
            print(f"audited {i}/{len(files)}")
    arr_shapes = np.asarray(shapes) if shapes else np.empty((0, 3))
    arr_spacing = np.asarray(spacings) if spacings else np.empty((0, 3))
    report = {
        "root": str(args.root.resolve()),
        "files": len(files),
        "valid": len(shapes),
        "invalid": bad,
        "sources": dict(sorted(sources.items())),
        "per_source": dict(sorted(per_source.items())),
        "schemas": {"|".join(k): v for k, v in schemas.items()},
        "image_dtypes": dict(dtypes),
        "shape_zyx": {
            "min": arr_shapes.min(0).tolist() if len(arr_shapes) else [],
            "median": np.median(arr_shapes, 0).tolist() if len(arr_shapes) else [],
            "max": arr_shapes.max(0).tolist() if len(arr_shapes) else [],
        },
        "spacing_xyz": {
            "min": arr_spacing.min(0).tolist() if len(arr_spacing) else [],
            "median": np.median(arr_spacing, 0).tolist() if len(arr_spacing) else [],
            "max": arr_spacing.max(0).tolist() if len(arr_spacing) else [],
        },
        "lesions_per_case": {
            "total": int(sum(lesions)),
            "mean": float(np.mean(lesions)) if lesions else 0.0,
            "max": max(lesions, default=0),
        },
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

