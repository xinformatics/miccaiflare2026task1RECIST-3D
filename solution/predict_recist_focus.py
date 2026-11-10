#!/usr/bin/env python3
"""Run GF-Screen Focus inference around each RECIST marker and optionally fuse SAM.

The output contract is the challenge's flat per-case NPZ format. When ground
truth exists, ``--calibration-json`` evaluates a declared grid of Focus/SAM
threshold pairs during the same forward pass, avoiding repeated validation
inference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import generate_binary_structure, label as cc_label

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT / "gf-screen"
sys.path.insert(0, str(GF_ROOT))
sys.path.insert(0, str(ROOT))

from models.GF_Screen_RL import GFScreen_RL  # noqa: E402
from roi3d.geometry import (  # noqa: E402
    V2_CACHE_SHAPE_ZYX, recist_geometries, sample_native, scatter_roi,
)

FOCUS_SHAPE = (96, 96, 64)  # model order: in-plane y, in-plane x, axial z
FOCUS_SPACING = (1.0, 1.0, 3.0)
DOMAIN_GROUPS = ("liver", "pancreas", "kidney", "lung", "colon", "other")
INSIDE_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
OUTSIDE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=GF_ROOT / "model_official.pt")
    p.add_argument("--sam-dir", type=Path,
                   help="optional challenge-format MedSAM2 predictions to fuse")
    p.add_argument("--inside-threshold", type=float, default=0.20,
                   help="Focus probability required where SAM predicts the lesion")
    p.add_argument("--outside-threshold", type=float, default=0.80,
                   help="Focus probability required to add voxels outside SAM")
    p.add_argument("--calibration-json", type=Path)
    p.add_argument("--file-list", type=Path)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _widen_for_domains(model: torch.nn.Module) -> None:
    def widen(old: torch.nn.Conv3d) -> torch.nn.Conv3d:
        return torch.nn.Conv3d(
            1 + len(DOMAIN_GROUPS), old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, dilation=old.dilation,
            groups=old.groups, bias=old.bias is not None,
        )
    model.swinViT.patch_embed.proj = widen(model.swinViT.patch_embed.proj)
    model.encoder1.layer.conv1.conv = widen(model.encoder1.layer.conv1.conv)
    model.encoder1.layer.conv3.conv = widen(model.encoder1.layer.conv3.conv)


def build_focus(weights: Path) -> tuple[torch.nn.Module, bool, bool]:
    class Args:
        glance_backbone = "resnet18"
        roi_x, roi_y, roi_z = FOCUS_SHAPE
        in_channels, out_channels, feature_size = 1, 2, 48
        use_checkpoint = False
        batch_size, sw_batch_size = 1, 1

    full = GFScreen_RL(Args())
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "segmentation_model" in checkpoint:
        organ_conditioned = bool(
            checkpoint.get("args", {}).get("organ_conditioned", False)
        )
        if organ_conditioned:
            _widen_for_domains(full.segmentation_model)
        full.segmentation_model.load_state_dict(
            checkpoint["segmentation_model"], strict=True
        )
    elif "state_dict" in checkpoint:
        full.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        raise ValueError(f"unrecognized Focus checkpoint: {weights}")
    scale_normalized = bool(
        checkpoint.get("args", {}).get("roi_cache", False)
    ) if "segmentation_model" in checkpoint else False
    return (full.segmentation_model,
            organ_conditioned if "segmentation_model" in checkpoint else False,
            scale_normalized)


def domain_from_name(name: str) -> int | None:
    lower = name.lower().replace("pancreatic", "pancreas")
    for index, group in enumerate(DOMAIN_GROUPS[:-1]):
        if lower.startswith(group):
            return index
    return None


def domain_inputs(image: torch.Tensor, domain: int | None) -> torch.Tensor:
    indices = (list(range(len(DOMAIN_GROUPS))) if domain is None else [domain])
    expanded = image.expand(len(indices), -1, -1, -1, -1)
    token = F.one_hot(torch.tensor(indices), num_classes=len(DOMAIN_GROUPS)).float()
    token = token[:, :, None, None, None].expand(-1, -1, *image.shape[2:])
    return torch.cat((expanded, token), dim=1)


def crop_geometry(shape: tuple[int, int, int], center_zyx: np.ndarray,
                  size_zyx: tuple[int, int, int]):
    starts = [int(round(float(c) - (size - 1) / 2))
              for c, size in zip(center_zyx, size_zyx)]
    ends = [start + size for start, size in zip(starts, size_zyx)]
    source = tuple(slice(max(0, start), min(n, end))
                   for start, end, n in zip(starts, ends, shape))
    pads = tuple((max(0, -start), max(0, end - n))
                 for start, end, n in zip(starts, ends, shape))
    return source, pads


def extract_input(imgs: np.ndarray, recist: np.ndarray, lesion_id: int,
                  spacing_xyz: np.ndarray):
    marker = np.argwhere(recist == lesion_id)
    center = marker.mean(axis=0)
    physical_zyx = (FOCUS_SHAPE[2] * FOCUS_SPACING[2],
                    FOCUS_SHAPE[0] * FOCUS_SPACING[0],
                    FOCUS_SHAPE[1] * FOCUS_SPACING[1])
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]),
                   float(spacing_xyz[0]))
    size_zyx = tuple(max(1, int(math.ceil(mm / sp)))
                     for mm, sp in zip(physical_zyx, spacing_zyx))
    source, pads = crop_geometry(imgs.shape, center, size_zyx)
    crop = np.pad(imgs[source], pads, mode="constant")
    image = torch.from_numpy(np.ascontiguousarray(crop.transpose(1, 2, 0))).float()
    image = F.interpolate(image[None, None], size=FOCUS_SHAPE,
                          mode="trilinear", align_corners=False) / 255.0
    return image, source, pads, size_zyx


def extract_scale_input(imgs: np.ndarray, recist: np.ndarray, lesion_id: int,
                        spacing_xyz: np.ndarray):
    geometry = next(g for g in recist_geometries(recist, spacing_xyz)
                    if g.lesion_id == lesion_id)
    crop = sample_native(
        imgs, geometry, spacing_xyz, V2_CACHE_SHAPE_ZYX, order=1
    )
    image = torch.from_numpy(np.ascontiguousarray(crop.transpose(1, 2, 0))).float()
    image = F.interpolate(image[None, None], size=FOCUS_SHAPE,
                          mode="trilinear", align_corners=False) / 255.0
    return image, geometry


def restore_scale_probability(probability: torch.Tensor, geometry,
                              spacing_xyz: np.ndarray, full_shape) -> np.ndarray:
    size_yxz = (V2_CACHE_SHAPE_ZYX[1], V2_CACHE_SHAPE_ZYX[2],
                V2_CACHE_SHAPE_ZYX[0])
    restored = F.interpolate(probability[None, None], size=size_yxz,
                             mode="trilinear", align_corners=False)[0, 0]
    restored = restored.permute(2, 0, 1).float().cpu().numpy()
    return scatter_roi(restored, geometry, spacing_xyz, full_shape, order=1)


def restore_probability(probability: torch.Tensor, source, pads,
                        size_zyx: tuple[int, int, int], full_shape) -> np.ndarray:
    # Model spatial order y,x,z -> challenge order z,y,x.
    size_yxz = (size_zyx[1], size_zyx[2], size_zyx[0])
    restored = F.interpolate(probability[None, None], size=size_yxz,
                             mode="trilinear", align_corners=False)[0, 0]
    restored = restored.permute(2, 0, 1).float().cpu().numpy()
    unpadded = tuple(slice(before, size - after if after else None)
                     for size, (before, after) in zip(size_zyx, pads))
    full = np.zeros(full_shape, dtype=np.float32)
    full[source] = restored[unpadded]
    return full


def restore_probability_crop(probability: torch.Tensor, pads,
                             size_zyx: tuple[int, int, int]) -> np.ndarray:
    """Restore model output only to the valid native-resolution crop."""
    size_yxz = (size_zyx[1], size_zyx[2], size_zyx[0])
    restored = F.interpolate(probability[None, None], size=size_yxz,
                             mode="trilinear", align_corners=False)[0, 0]
    restored = restored.permute(2, 0, 1).float().cpu().numpy()
    unpadded = tuple(slice(before, size - after if after else None)
                     for size, (before, after) in zip(size_zyx, pads))
    return restored[unpadded]


def anchored_component(mask: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    # Label only the occupied lesion neighborhood.  Labeling a complete CT
    # volume for every calibration threshold is equivalent but needlessly
    # expensive (and made large public cases take minutes).
    occupied = np.argwhere(mask | prompt)
    if not len(occupied):
        return mask
    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0) + 1
    region = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    local_mask = mask[region]
    local_prompt = prompt[region]
    components, count = cc_label(
        local_mask, structure=generate_binary_structure(3, 1)
    )
    if count == 0:
        return mask
    touching = np.unique(components[local_prompt])
    touching = touching[touching != 0]
    if len(touching):
        selected = np.isin(components, touching)
        result = np.zeros_like(mask, dtype=bool)
        result[region] = selected
        return result
    # Focus can leave a sub-voxel gap at the rasterized marker. Retain the
    # component whose voxel is nearest the marker center rather than an arbitrary
    # large false positive elsewhere in the physical crop.
    center = np.argwhere(local_prompt).mean(axis=0)
    best_label, best_distance = 0, float("inf")
    for component_id in range(1, count + 1):
        coords = np.argwhere(components == component_id)
        distance = float(((coords - center) ** 2).sum(axis=1).min())
        if distance < best_distance:
            best_label, best_distance = component_id, distance
    result = np.zeros_like(mask, dtype=bool)
    result[region] = components == best_label
    return result


def fused_mask(probability: np.ndarray, sam_mask: np.ndarray | None,
               prompt: np.ndarray, inside: float, outside: float) -> np.ndarray:
    if sam_mask is None:
        candidate = probability >= outside
    else:
        candidate = ((sam_mask & (probability >= inside)) |
                     (probability >= outside))
    return anchored_component(candidate, prompt)


def dice(mask: np.ndarray, target: np.ndarray) -> float:
    denominator = int(mask.sum()) + int(target.sum())
    return (2.0 * int(np.count_nonzero(mask & target)) / denominator
            if denominator else 1.0)


def cropped_dice(mask: np.ndarray, target_crop: np.ndarray,
                 target_size: int) -> float:
    """Dice for a prediction confined to a crop and a full-volume target."""
    denominator = int(mask.sum()) + target_size
    return (2.0 * int(np.count_nonzero(mask & target_crop)) / denominator
            if denominator else 1.0)


def additive_dice(selected_crop: np.ndarray, sam_crop: np.ndarray,
                  sam_full: np.ndarray, target_crop: np.ndarray,
                  target_full: np.ndarray) -> float:
    """Score a conservative fusion that preserves SAM and only adds Focus."""
    added = selected_crop & ~sam_crop
    intersection = (int(np.count_nonzero(sam_full & target_full)) +
                    int(np.count_nonzero(added & target_crop)))
    denominator = int(sam_full.sum()) + int(added.sum()) + int(target_full.sum())
    return 2.0 * intersection / denominator if denominator else 1.0


def input_files(root: Path, file_list: Path | None) -> list[Path]:
    if file_list is None:
        return sorted(root.rglob("*.npz"))
    names = [line.strip() for line in file_list.read_text().splitlines() if line.strip()]
    return [root / (name if name.endswith(".npz") else f"{name}.npz")
            for name in names]


def main() -> None:
    args = parse_args()
    if not 0 <= args.inside_threshold <= 1 or not 0 <= args.outside_threshold <= 1:
        raise SystemExit("thresholds must be in [0,1]")
    device = torch.device(args.device)
    model, organ_conditioned, scale_normalized = build_focus(args.weights)
    model = model.eval().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    calibration: dict[str, list[float]] = defaultdict(list)
    timings = []

    for path in input_files(args.input, args.file_list):
        started = time.time()
        with np.load(path) as case:
            imgs = case["imgs"]
            recist = case["recist"]
            spacing = np.asarray(case["spacing"])
            gts = case["gts"] if "gts" in case else None
        sam = None
        if args.sam_dir:
            with np.load(args.sam_dir / path.name) as prediction:
                sam = prediction["segs"]

        segs = np.zeros_like(imgs, dtype=np.uint8)
        lesion_ids = np.unique(recist)
        lesion_ids = lesion_ids[lesion_ids != 0].astype(int)
        for lesion_id in lesion_ids:
            if scale_normalized:
                image, geometry = extract_scale_input(
                    imgs, recist, int(lesion_id), spacing
                )
                source = tuple(slice(0, n) for n in imgs.shape)
            else:
                image, source, pads, size_zyx = extract_input(
                    imgs, recist, int(lesion_id), spacing
                )
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16
            ):
                if organ_conditioned:
                    model_image = domain_inputs(
                        image, domain_from_name(path.stem)
                    ).to(device)
                    logits = model(model_image)
                    probability = logits.softmax(1)[:, 1].mean(0)
                else:
                    logits = model(image.to(device))
                    probability = logits.softmax(1)[0, 1]
            if scale_normalized:
                probability = restore_scale_probability(
                    probability, geometry, spacing, imgs.shape
                )
            else:
                probability = restore_probability_crop(probability, pads, size_zyx)
            prompt = recist[source] == lesion_id
            sam_mask = (sam[source] == lesion_id) if sam is not None else None
            selected = fused_mask(
                probability, sam_mask, prompt,
                args.inside_threshold, args.outside_threshold
            )
            output_crop = segs[source]
            output_crop[selected & (output_crop == 0)] = lesion_id

            if gts is not None and args.calibration_json:
                target = gts == lesion_id
                target_crop = target[source]
                target_size = int(target.sum())
                calibration["sam"].append(
                    dice(sam == lesion_id, target)
                    if sam is not None else 0
                )
                calibration["focus_0.50"].append(
                    cropped_dice(
                        fused_mask(probability, None, prompt, 0.5, 0.5),
                        target_crop, target_size
                    )
                )
                if sam is not None:
                    sam_full = sam == lesion_id
                    for inside in INSIDE_THRESHOLDS:
                        for outside in OUTSIDE_THRESHOLDS:
                            if outside < inside:
                                continue
                            key = f"inside_{inside:.2f}_outside_{outside:.2f}"
                            calibration[key].append(cropped_dice(
                                fused_mask(probability, sam_mask, prompt,
                                           inside, outside),
                                target_crop, target_size
                            ))
                    for outside in OUTSIDE_THRESHOLDS:
                        selected_add = fused_mask(
                            probability, sam_mask, prompt, 0.0, outside
                        )
                        calibration[f"add_outside_{outside:.2f}"].append(
                            additive_dice(selected_add, sam_mask, sam_full,
                                          target_crop, target)
                        )

        np.savez_compressed(args.output / path.name, segs=segs)
        timings.append({"case": path.stem, "seconds": time.time() - started})
        print(f"{path.name}: {timings[-1]['seconds']:.2f}s", flush=True)

    (args.output / "inference_times.json").write_text(json.dumps(timings, indent=2))
    if args.calibration_json:
        summary = {key: {"mean_dsc": float(np.mean(values)), "lesions": len(values)}
                   for key, values in calibration.items()}
        summary = dict(sorted(summary.items(),
                              key=lambda item: item[1]["mean_dsc"], reverse=True))
        args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_json.write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(list(summary.items())[:10]), indent=2))


if __name__ == "__main__":
    main()
