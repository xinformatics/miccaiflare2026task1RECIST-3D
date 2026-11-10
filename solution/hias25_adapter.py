#!/usr/bin/env python3
"""Adapt RECIST challenge NPZ volumes to/from the hias25 nnU-Net image.

The challenge stores a lossy uint8 windowed CT. hias25 was trained on HU, so
``prepare`` approximately reverses the original window before nnU-Net input.
``convert`` assigns unprompted binary tumor components to RECIST lesion IDs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


WINDOWS = {
    "soft_tissue": (-160.0, 240.0),
    "lung": (-1350.0, 150.0),
    "bone": (-500.0, 1300.0),
    "brain": (0.0, 80.0),
}


def npz_files(path: Path, limit: int | None) -> list[Path]:
    files = sorted(path.glob("*.npz"))
    return files if limit is None else files[:limit]


def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    lo, hi = WINDOWS[args.window]
    files = npz_files(args.input, args.limit)
    for index, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as data:
            pixels = np.asarray(data["imgs"], dtype=np.float32)
            hu = lo + pixels * ((hi - lo) / 255.0)
            image = sitk.GetImageFromArray(hu)
            image.SetSpacing(tuple(float(x) for x in data["spacing"]))
            image.SetDirection(tuple(float(x) for x in data["direction"]))
            image.SetOrigin(tuple(float(x) for x in data["origin"]))
        sitk.WriteImage(image, str(args.output / f"{path.stem}_0000.nii.gz"), True)
        print(f"prepare {index}/{len(files)} {path.name}", flush=True)


def component_for_prompt(binary: np.ndarray, prompt: np.ndarray,
                         spacing_xyz: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(binary, structure=ndi.generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros_like(binary, dtype=bool)

    touched = np.unique(labels[prompt])
    touched = touched[touched != 0]
    if len(touched):
        return np.isin(labels, touched)

    # Sparse physical distances are much cheaper than a full-volume distance
    # transform because tumor predictions and RECIST lines occupy few voxels.
    scale = spacing_xyz[::-1]
    foreground = np.argwhere(binary)
    tree = cKDTree(np.argwhere(prompt) * scale)
    distances = tree.query(foreground * scale, workers=-1)[0]
    foreground_labels = labels[tuple(foreground.T)]
    minima = np.full(count + 1, np.inf)
    np.minimum.at(minima, foreground_labels, distances)
    chosen = int(np.argmin(minima[1:])) + 1
    return labels == chosen


def convert(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    files = npz_files(args.input, args.limit)
    for index, path in enumerate(files, 1):
        prediction_path = args.predictions / f"{path.stem}.nii.gz"
        with np.load(path, allow_pickle=False) as data:
            shape = data["imgs"].shape
            recist = np.asarray(data["recist"])
            spacing = np.asarray(data["spacing"], dtype=np.float64)
        if prediction_path.exists():
            binary = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_path))) > 0
            if binary.shape != shape:
                raise ValueError(f"shape mismatch {path.name}: {binary.shape} != {shape}")
        else:
            binary = np.zeros(shape, dtype=bool)

        segs = np.zeros(shape, dtype=np.uint8)
        # Resolve overlaps by assigning each voxel to the closest RECIST line.
        candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for lesion_id in np.unique(recist):
            if lesion_id == 0:
                continue
            prompt = recist == lesion_id
            mask = component_for_prompt(binary, prompt, spacing)
            candidates.append((int(lesion_id), mask, prompt))
        if candidates:
            best_distance = np.full(shape, np.inf, dtype=np.float32)
            scale = spacing[::-1]
            for lesion_id, mask, prompt in candidates:
                coordinates = np.argwhere(mask)
                if not len(coordinates):
                    continue
                tree = cKDTree(np.argwhere(prompt) * scale)
                distances = tree.query(coordinates * scale, workers=-1)[0]
                old = best_distance[tuple(coordinates.T)]
                selected = distances < old
                chosen = coordinates[selected]
                segs[tuple(chosen.T)] = lesion_id
                best_distance[tuple(chosen.T)] = distances[selected]
        np.savez_compressed(args.output / path.name, segs=segs)
        print(f"convert {index}/{len(files)} {path.name} voxels={int(binary.sum())}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--window", choices=WINDOWS, default="soft_tissue")
    prep.add_argument("--limit", type=int)
    prep.set_defaults(func=prepare)
    conv = sub.add_parser("convert")
    conv.add_argument("--input", type=Path, required=True)
    conv.add_argument("--predictions", type=Path, required=True)
    conv.add_argument("--output", type=Path, required=True)
    conv.add_argument("--limit", type=int)
    conv.set_defaults(func=convert)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
