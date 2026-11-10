#!/usr/bin/env python3
"""Local per-lesion DSC/NSD evaluator matching the published challenge contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else 2.0 * float(np.logical_and(a, b).sum()) / denom


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, border_value=0))


def nsd(a: np.ndarray, b: np.ndarray, spacing_xyz: np.ndarray, tolerance=1.0) -> float:
    """Symmetric normalized surface Dice at ``tolerance`` millimetres."""
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    # Arrays are z,y,x while SimpleITK spacing is x,y,z.
    sampling = tuple(float(x) for x in spacing_xyz[::-1])
    # Distance transforms over an entire 512x512 CT volume are unnecessarily
    # expensive. Crop to the union plus >tolerance padding; distances outside it
    # cannot change whether any surface point is within tolerance.
    coords = np.where(np.logical_or(a, b))
    slices = []
    for axis, values in enumerate(coords):
        pad = int(np.ceil(tolerance / sampling[axis])) + 1
        lo = max(0, int(values.min()) - pad)
        hi = min(a.shape[axis], int(values.max()) + pad + 1)
        slices.append(slice(lo, hi))
    crop = tuple(slices)
    sa, sb = surface(a[crop]), surface(b[crop])
    d_to_b = distance_transform_edt(~sb, sampling=sampling)
    d_to_a = distance_transform_edt(~sa, sampling=sampling)
    close = int((d_to_b[sa] <= tolerance).sum()) + int((d_to_a[sb] <= tolerance).sum())
    return close / float(sa.sum() + sb.sum())


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        key = "gts" if "gts" in data.files else "gts_hidden"
        return data[key], np.asarray(data["spacing"], dtype=np.float64)


def score_case(gt_path: Path, pred_path: Path) -> list[dict]:
    gt, spacing = load_gt(gt_path)
    if not pred_path.exists():
        return [
            {"case": gt_path.stem, "lesion": int(k), "DSC": 0.0, "NSD": 0.0}
            for k in np.unique(gt) if k != 0
        ]
    with np.load(pred_path, allow_pickle=False) as data:
        pred = data["segs"]
    if pred.shape != gt.shape:
        return [
            {"case": gt_path.stem, "lesion": int(k), "DSC": 0.0, "NSD": 0.0}
            for k in np.unique(gt) if k != 0
        ]
    rows = []
    for lesion_id in np.unique(gt):
        if lesion_id == 0:
            continue
        a, b = gt == lesion_id, pred == lesion_id
        rows.append({
            "case": gt_path.stem,
            "lesion": int(lesion_id),
            "DSC": dice(a, b),
            # ``nsd`` accepts NPZ/SimpleITK x,y,z spacing and converts it once
            # to the z,y,x NumPy array order internally.
            "NSD": nsd(a, b, spacing),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--file-list", type=Path,
                        help="optional relative case list, with or without .npz")
    args = parser.parse_args()
    if args.file_list:
        rels = [x.strip() for x in args.file_list.read_text().splitlines() if x.strip()]
        gt_files = [args.ground_truth / (x if x.endswith(".npz") else x + ".npz")
                    for x in rels]
    else:
        gt_files = sorted(args.ground_truth.rglob("*.npz"))
    if not gt_files:
        raise SystemExit(f"no NPZ files under {args.ground_truth}")
    rows = [r for f in gt_files for r in score_case(f, args.predictions / f.name)]
    if not rows:
        raise SystemExit("ground truth contains no lesion IDs")
    cases = sorted({r["case"] for r in rows})
    case_dsc = [np.mean([r["DSC"] for r in rows if r["case"] == case])
                for case in cases]
    case_nsd = [np.mean([r["NSD"] for r in rows if r["case"] == case])
                for case in cases]
    summary = {
        "cases": len(cases),
        "lesions": len(rows),
        "Lesion_DSC": float(np.mean([r["DSC"] for r in rows])),
        "Lesion_NSD": float(np.mean([r["NSD"] for r in rows])),
        "Case_DSC": float(np.mean(case_dsc)),
        "Case_NSD": float(np.mean(case_nsd)),
    }
    print(json.dumps(summary, indent=2))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("case", "lesion", "DSC", "NSD"))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
