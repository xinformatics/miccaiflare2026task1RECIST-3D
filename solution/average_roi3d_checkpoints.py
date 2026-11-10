"""Average compatible ROI3D model weights into an inference checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [torch.load(path, map_location="cpu", weights_only=False)
              for path in args.checkpoints]
    states = [checkpoint.get("model", checkpoint) for checkpoint in loaded]
    if any(state.keys() != states[0].keys() for state in states[1:]):
        raise ValueError("checkpoint parameter names differ")
    averaged = {}
    for name in states[0]:
        tensors = [state[name] for state in states]
        if tensors[0].is_floating_point():
            averaged[name] = torch.stack(tensors).mean(0)
        else:
            averaged[name] = tensors[-1].clone()
    output = {
        "model": averaged,
        "args": loaded[-1].get("args", {}),
        "averaged_from": [str(path) for path in args.checkpoints],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)


if __name__ == "__main__":
    main()
