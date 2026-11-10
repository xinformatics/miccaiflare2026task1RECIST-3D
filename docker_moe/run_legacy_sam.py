#!/usr/bin/env python3
"""Run the historically scored SAM2 inference constants without source edits."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--size", choices=("small", "tiny"), required=True)
    p.add_argument("--prompt", choices=("box", "3points"), required=True)
    p.add_argument("--box-scale", type=float, required=True)
    p.add_argument("--slab-scale", type=float, required=True)
    p.add_argument("--bounded-3points", action="store_true",
                   help="apply the configured z slab to 3-point propagation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.script.parent))
    spec = importlib.util.spec_from_file_location("legacy_effmedsam2", args.script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {args.script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BOX_SCALE = args.box_scale
    module.Z_SLAB_SCALE = args.slab_scale
    module.BOUNDED_3POINTS = args.bounded_3points
    sys.argv = [
        str(args.script), "--weights", str(args.weights),
        "--model-size", args.size, "--device", args.device,
        "--prompt", args.prompt, "-i", str(args.input), "-o", str(args.output),
    ]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
