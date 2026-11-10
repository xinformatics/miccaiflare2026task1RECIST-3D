"""Keep one RECIST-anchored connected component per predicted lesion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-dilation", type=int, default=0)
    return parser.parse_args()


def select_component(mask: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    components, count = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    touching = np.unique(components[prompt])
    touching = touching[touching != 0]
    if len(touching):
        sizes = np.bincount(components.ravel())
        chosen = int(touching[np.argmax(sizes[touching])])
    else:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        chosen = int(np.argmax(sizes))
    return components == chosen


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for pred_path in sorted(args.pred.glob("*.npz")):
        with np.load(pred_path, allow_pickle=False) as archive:
            segs = archive["segs"]
        with np.load(args.cases / pred_path.name, allow_pickle=False) as case:
            recist = case["recist"]
        result = np.zeros_like(segs, dtype=np.uint8)
        for lesion_id in np.unique(recist):
            if lesion_id == 0:
                continue
            prompt = recist == lesion_id
            if args.prompt_dilation:
                prompt = binary_dilation(prompt, iterations=args.prompt_dilation)
            selected = select_component(segs == lesion_id, prompt)
            result[selected & (result == 0)] = lesion_id
        np.savez_compressed(args.out / pred_path.name, segs=result)


if __name__ == "__main__":
    main()
