#!/usr/bin/env python3
"""Run a small, pre-declared prompt-geometry ablation on a training holdout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = (
    {"name": "box_z12", "prompt": "box", "z": 1.2},
    {"name": "box_z14", "prompt": "box", "z": 1.4},
    {"name": "box_z16", "prompt": "box", "z": 1.6},
    {"name": "points3_z14", "prompt": "3points", "z": 1.4},
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("file_list", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/ablations"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    inference = root / "baseline/Efficient_MedSAM2_infer_CT_lesion_npz_recist.py"
    evaluator = root / "solution/evaluate.py"
    results = []
    for config in CONFIGS:
        out = args.output / config["name"]
        command = [
            sys.executable, str(inference), "--weights", str(args.weights),
            "--model-size", "tiny", "--device", args.device, "--no-compile",
            "--prompt", config["prompt"], "--z-slab-scale", str(config["z"]),
            "--debug-output", "--imgs-path", str(args.data),
            "--file-list", str(args.file_list), "--pred-save-dir", str(out),
        ]
        subprocess.run(command, check=True)
        scored = subprocess.run(
            [sys.executable, str(evaluator), str(args.data), str(out),
             "--file-list", str(args.file_list), "--csv", str(out / "lesions.csv")],
            check=True, text=True, capture_output=True,
        )
        metrics = json.loads(scored.stdout)
        metrics["name"] = config["name"]
        results.append(metrics)
        print(json.dumps(metrics))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
