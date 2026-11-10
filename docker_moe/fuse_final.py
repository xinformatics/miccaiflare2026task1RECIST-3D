#!/usr/bin/env python3
"""Fuse the released 3D probability with five binary members."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

RELEASE_SRC = Path(__file__).resolve().parents[1] / "release_src"
if not RELEASE_SRC.exists():
    RELEASE_SRC = Path(__file__).resolve().parents[1] / "othermachine" / "release" / "src"
sys.path.insert(0, str(RELEASE_SRC))
from hybrid_anchor import load_net, ours_prob  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--member-dirs", type=Path, nargs=5, required=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def anchored(mask: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    components, count = ndi.label(mask)
    if count <= 1:
        return mask
    seeds = np.unique(components[prompt])
    seeds = seeds[seeds != 0]
    if not len(seeds):
        sizes = ndi.sum(mask, components, range(1, count + 1))
        seeds = np.asarray([int(np.argmax(sizes)) + 1])
    return np.isin(components, seeds)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    net = load_net(str(args.checkpoint), args.device)
    weights = (0.40, 0.12, 0.12, 0.12, 0.12, 0.12)
    files = sorted(args.input.glob("*.npz"))
    for index, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as case:
            imgs = case["imgs"]
            recist = case["recist"]
            spacing = case["spacing"]
        members = []
        for directory in args.member_dirs:
            with np.load(directory / path.name, allow_pickle=False) as prediction:
                members.append(prediction["segs"])
        segs = np.zeros(imgs.shape, dtype=np.uint8)
        best = np.zeros(imgs.shape, dtype=np.float32)
        for lesion_id in np.unique(recist):
            if lesion_id == 0:
                continue
            probability = ours_prob(
                net, imgs, recist, spacing, lesion_id, "v2", args.device
            )
            score = weights[0] * probability
            for weight, member in zip(weights[1:], members):
                score += weight * (member == lesion_id)
            selected = anchored(score > 0.5, recist == lesion_id)
            replace = selected & (score > best)
            segs[replace] = lesion_id
            best[replace] = score[replace]
        np.savez_compressed(args.output / path.name, segs=segs)
        print(f"final {index}/{len(files)} {path.name}", flush=True)


if __name__ == "__main__":
    main()
