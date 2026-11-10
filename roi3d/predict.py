from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label

from roi3d.geometry import (
    V2_SHAPE_ZYX, rasterized_line, recist_geometries, sample_native, scatter_roi,
)
from roi3d.model import DOMAIN_GROUPS, GlanceFocusMoE, RECISTUNet3D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 RECIST ROI U-Net inference and fusion")
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--member-dirs", type=Path, nargs="*")
    p.add_argument("--ensemble-weights", type=str,
                   help="soft-model weight followed by one weight per member dir")
    p.add_argument(
        "--ensemble-grid", type=str, nargs="*",
        help=("named multi-member weight sets in NAME=W0,W1,... form; "
              "writes one NAME subdirectory per set while sharing inference"),
    )
    p.add_argument("--weight", type=float, default=0.58)
    p.add_argument("--weights", type=str,
                   help="comma-separated fusion weights; writes one wXXXX subdirectory each")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--file-list", type=Path)
    p.add_argument("--domain-routing", choices=("router", "filename"), default="router",
                   help="expert routing; filename uses an organ prefix when available")
    return p.parse_args()


def files(root: Path, listing: Path | None) -> list[Path]:
    if listing is None:
        return sorted(root.rglob("*.npz"))
    names = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    return [root / (name if name.endswith(".npz") else name + ".npz")
            for name in names]


def filename_domain(path: Path) -> int | None:
    lower = path.stem.lower()
    aliases = {"pancrea": "pancreas", "renal": "kidney"}
    for token, group in aliases.items():
        if token in lower:
            return DOMAIN_GROUPS.index(group)
    for group in DOMAIN_GROUPS[:-1]:
        if group in lower:
            return DOMAIN_GROUPS.index(group)
    return None


def model_input(image: np.ndarray, geometry,
                intensity_affinity: bool = False) -> np.ndarray:
    line = rasterized_line(V2_SHAPE_ZYX, geometry)
    distance = distance_transform_edt(~line.astype(bool),
                                      sampling=geometry.roi_spacing_mm)
    prior = np.exp(-distance / max(0.4 * geometry.diameter_true_mm, 1e-3))
    size = np.full(V2_SHAPE_ZYX,
                   np.log(geometry.diameter_true_mm / 30.0) / 1.5,
                   dtype=np.float32)
    normalized = image.astype(np.float32) / 255.0
    channels = [normalized, line.astype(np.float32), prior.astype(np.float32), size]
    if intensity_affinity:
        # Training rasterizes a one-voxel-radius RECIST tube.
        values = normalized[distance <= geometry.roi_spacing_mm]
        reference = float(values.mean()) if len(values) else float(normalized.mean())
        channels.append(np.exp(-np.abs(normalized - reference) / 0.12).astype(np.float32))
    return np.stack(channels)


def anchored_component(mask: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    """Select one 26-connected component, preferring one touched by RECIST."""
    if not np.any(mask):
        return mask
    coordinates = np.argwhere(mask | prompt)
    lo = np.maximum(coordinates.min(0) - 1, 0)
    hi = np.minimum(coordinates.max(0) + 2, mask.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    components, count = label(
        mask[crop], structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if count <= 1:
        return mask
    touching = np.unique(components[prompt[crop]])
    touching = touching[touching != 0]
    sizes = np.bincount(components.ravel())
    if len(touching):
        chosen = int(touching[np.argmax(sizes[touching])])
    else:
        sizes[0] = 0
        chosen = int(np.argmax(sizes))
    selected = np.zeros_like(mask)
    selected[crop] = components == chosen
    return selected


def main() -> None:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    weights = ([float(value) for value in args.weights.split(",")]
               if args.weights else [args.weight])
    if any(weight <= 0.5 or weight > 1 for weight in weights):
        raise ValueError("fusion weights must be in (0.5, 1]")
    member_dirs = args.member_dirs or []
    ensemble_weights = None
    ensemble_grid = []
    if args.ensemble_grid:
        if args.ensemble_weights:
            raise ValueError("use either --ensemble-weights or --ensemble-grid")
        for specification in args.ensemble_grid:
            try:
                name, values = specification.split("=", 1)
                parsed = [float(value) for value in values.split(",")]
            except ValueError as error:
                raise ValueError(
                    f"invalid ensemble grid entry {specification!r}"
                ) from error
            if not name or "/" in name or name in {".", ".."}:
                raise ValueError(f"invalid ensemble grid name {name!r}")
            if len(parsed) != len(member_dirs) + 1:
                raise ValueError(
                    "each ensemble grid entry requires one soft weight plus one "
                    "weight per member"
                )
            if not np.isclose(sum(parsed), 1.0):
                raise ValueError(f"ensemble weights for {name} must sum to one")
            ensemble_grid.append((name, parsed))
        if args.baseline_dir or args.weights:
            raise ValueError("multi-member mode is incompatible with baseline sweep mode")
    if args.ensemble_weights:
        ensemble_weights = [float(value) for value in args.ensemble_weights.split(",")]
        if len(ensemble_weights) != len(member_dirs) + 1:
            raise ValueError("ensemble weights require one soft weight plus one per member")
        if not np.isclose(sum(ensemble_weights), 1.0):
            raise ValueError("ensemble weights must sum to one")
        if args.baseline_dir or args.weights:
            raise ValueError("multi-member mode is incompatible with baseline sweep mode")
    output_dirs = []
    if ensemble_grid:
        for name, _ in ensemble_grid:
            output = args.output / name
            output.mkdir(parents=True, exist_ok=True)
            output_dirs.append(output)
    else:
        for weight in weights:
            output = (args.output / f"w{weight:.4f}" if args.weights else args.output)
            output.mkdir(parents=True, exist_ok=True)
            output_dirs.append(output)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    checkpoint_args = checkpoint.get("args", {})
    base = int(checkpoint_args.get("base", 32))
    intensity_affinity = bool(checkpoint_args.get("intensity_affinity", False))
    organ_experts = bool(checkpoint_args.get("organ_experts", False))
    glance_focus_moe = bool(checkpoint_args.get("glance_focus_moe", False))
    expert_rank = int(checkpoint_args.get("expert_rank", 8))
    if glance_focus_moe:
        model = GlanceFocusMoE(
            in_channels=5 if intensity_affinity else 4,
            base=base, deep_supervision=False,
            experts=int(checkpoint_args.get("experts", 4)),
            expert_rank=expert_rank,
            router_temperature=float(
                checkpoint_args.get("router_temperature", 1.0)
            ),
        )
    else:
        model = RECISTUNet3D(
            in_channels=5 if intensity_affinity else 4,
            base=base, deep_supervision=False,
            organ_experts=organ_experts, expert_rank=expert_rank,
        )
    model = model.to(device).eval()
    model.load_state_dict(state, strict=True)
    timings = []
    for path in files(args.input, args.file_list):
        started = time.time()
        with np.load(path, allow_pickle=False) as case:
            image = case["imgs"]
            recist = case["recist"]
            spacing = np.asarray(case["spacing"], dtype=np.float64)
        geometries = recist_geometries(recist, spacing)
        probabilities = []
        # Build only one inference batch at a time. Some pan-cancer volumes
        # contain dozens of RECIST labels; retaining every 4-channel 3D ROI can
        # otherwise exceed the CPU challenge's 8 GB memory limit.
        for start in range(0, len(geometries), args.batch_size):
            batch_geometries = geometries[start:start + args.batch_size]
            batch_inputs = []
            for geometry in batch_geometries:
                roi = sample_native(image, geometry, spacing, V2_SHAPE_ZYX, order=1)
                batch_inputs.append(model_input(roi, geometry, intensity_affinity))
            tensor = torch.from_numpy(np.stack(batch_inputs)).to(device)
            domain = None
            if organ_experts and args.domain_routing == "filename":
                index = filename_domain(path)
                if index is not None:
                    domain = torch.full(
                        (len(tensor),), index, dtype=torch.long, device=device
                    )
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                 dtype=torch.bfloat16):
                logits = (model(tensor, domain) if organ_experts
                          else model(tensor))
            probabilities.extend(logits.sigmoid()[:, 0].float().cpu().numpy())
        baseline = None
        if args.baseline_dir:
            baseline_path = args.baseline_dir / path.name
            if baseline_path.exists():
                with np.load(baseline_path, allow_pickle=False) as pred:
                    baseline = pred["segs"]
        members = []
        for member_dir in member_dirs:
            with np.load(member_dir / path.name, allow_pickle=False) as pred:
                members.append(pred["segs"])
        active_ensembles = ([weights for _, weights in ensemble_grid]
                            if ensemble_grid else
                            ([ensemble_weights] if ensemble_weights else []))
        output_count = len(active_ensembles) if active_ensembles else len(weights)
        segmentations = [np.zeros_like(image, dtype=np.uint8)
                         for _ in range(output_count)]
        best_scores = ([np.full(image.shape, -np.inf, dtype=np.float16)
                        for _ in active_ensembles] if active_ensembles else [])
        for geometry, roi_probability in zip(geometries, probabilities):
            probability = scatter_roi(
                roi_probability, geometry, spacing, image.shape, order=1
            )
            if active_ensembles:
                for segs, best_score, active_weights in zip(
                        segmentations, best_scores, active_ensembles):
                    score = active_weights[0] * probability
                    for member, weight in zip(members, active_weights[1:]):
                        score += weight * (member == geometry.lesion_id)
                    selected = anchored_component(
                        score > args.threshold, recist == geometry.lesion_id
                    )
                    replace = selected & (score > best_score)
                    segs[replace] = geometry.lesion_id
                    best_score[replace] = score[replace]
                continue
            if baseline is None:
                selections = [probability > args.threshold for _ in weights]
            else:
                baseline_mask = baseline == geometry.lesion_id
                selections = [
                    (weight * probability + (1 - weight) * baseline_mask) > args.threshold
                    for weight in weights
                ]
            for segs, selected in zip(segmentations, selections):
                segs[selected & (segs == 0)] = geometry.lesion_id
        for output, segs in zip(output_dirs, segmentations):
            np.savez_compressed(output / path.name, segs=segs)
        timings.append({"case": path.stem, "seconds": time.time() - started})
        print(f"{path.name}: {timings[-1]['seconds']:.2f}s", flush=True)
    (args.output / "inference_times.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
