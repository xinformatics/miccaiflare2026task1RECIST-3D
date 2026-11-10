#!/usr/bin/env python3
"""Create the flat ZIP required by Codabench."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    files = sorted(args.predictions.glob("*.npz"))
    if not files:
        raise SystemExit(f"no flat NPZ files in {args.predictions}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, arcname=path.name)
    print(f"wrote {args.output} ({len(files)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
