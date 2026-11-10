#!/usr/bin/env python3
"""Cross-validate diameter-conditioned selection among existing predictions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from roi3d.geometry import recist_geometries
from solution.evaluate import dice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2,
                        metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--folds", type=int, default=7)
    return parser.parse_args()


def load_records(root: Path, candidates: list[tuple[str, Path]]) -> list[dict]:
    records = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as case:
            gt = case["gts"]
            recist = case["recist"]
            spacing = np.asarray(case["spacing"], dtype=np.float64)
        predictions = {}
        for name, directory in candidates:
            with np.load(directory / path.name, allow_pickle=False) as pred:
                predictions[name] = pred["segs"]
        for geometry in recist_geometries(recist, spacing):
            lesion_id = geometry.lesion_id
            target = gt == lesion_id
            records.append({
                "case": path.stem,
                "diameter": geometry.diameter_true_mm,
                "scores": {
                    name: dice(target, prediction == lesion_id)
                    for name, prediction in predictions.items()
                },
            })
    return records


def case_mean(records: list[dict], choices: list[str]) -> float:
    by_case: dict[str, list[float]] = {}
    for record, choice in zip(records, choices):
        by_case.setdefault(record["case"], []).append(record["scores"][choice])
    return float(np.mean([np.mean(values) for values in by_case.values()]))


def apply_rule(records: list[dict], thresholds: tuple[float, ...],
               choices: tuple[str, ...]) -> list[str]:
    return [choices[int(np.searchsorted(thresholds, r["diameter"]))]
            for r in records]


def best_rule(records: list[dict], names: list[str]) -> tuple:
    diameters = np.asarray([r["diameter"] for r in records])
    # Fixed clinically interpretable grid limits flexibility and fold overfit.
    grid = [15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0]
    rules = [((), (name,)) for name in names]
    rules += [((threshold,), choices)
              for threshold in grid
              for choices in itertools.product(names, repeat=2)]
    rules += [(thresholds, choices)
              for thresholds in itertools.combinations(grid, 2)
              for choices in itertools.product(names, repeat=3)]
    best = None
    for thresholds, choices in rules:
        score = case_mean(records, apply_rule(records, thresholds, choices))
        # Prefer simpler and then lexicographically stable rules on ties.
        key = (score, -len(thresholds), tuple(choices), tuple(thresholds))
        if best is None or key > best[0]:
            best = (key, thresholds, choices)
    assert best is not None
    return best[1], best[2], best[0][0]


def main() -> None:
    args = parse_args()
    candidates = [(name, Path(directory)) for name, directory in args.candidate]
    names = [name for name, _ in candidates]
    if args.baseline not in names:
        raise SystemExit("--baseline must name one of --candidate")
    records = load_records(args.ground_truth, candidates)
    cases = sorted({record["case"] for record in records})
    folds = {case: index % args.folds for index, case in enumerate(cases)}
    selected_scores, baseline_scores, rules = [], [], []
    for fold in range(args.folds):
        train = [r for r in records if folds[r["case"]] != fold]
        test = [r for r in records if folds[r["case"]] == fold]
        thresholds, choices, train_score = best_rule(train, names)
        selected_scores.append(case_mean(test, apply_rule(test, thresholds, choices)))
        baseline_scores.append(case_mean(test, [args.baseline] * len(test)))
        rules.append({"fold": fold, "thresholds": thresholds, "choices": choices,
                      "train_score": train_score,
                      "test_score": selected_scores[-1],
                      "baseline_score": baseline_scores[-1]})
    thresholds, choices, full_score = best_rule(records, names)
    summary = {
        "cases": len(cases), "lesions": len(records),
        "candidates": {name: case_mean(records, [name] * len(records))
                       for name in names},
        "oof_selected": float(np.mean(selected_scores)),
        "oof_baseline": float(np.mean(baseline_scores)),
        "oof_gain": float(np.mean(np.asarray(selected_scores) - baseline_scores)),
        "fold_rules": rules,
        "full_rule": {"thresholds": thresholds, "choices": choices,
                      "score": full_score},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
