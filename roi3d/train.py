from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from roi3d.geometry import V2_CACHE_SHAPE_ZYX, V2_SHAPE_ZYX, recist_geometries
from roi3d.model import (
    DOMAIN_GROUPS, GlanceFocusMoE, RECISTUNet3D, parameter_count,
)


TARGET_W = {
    "liver": 0.20, "pancreas": 0.20, "kidney": 0.20,
    "lung": 0.18, "colon": 0.08,
}
SOURCE_ALIASES = {"pancreatic": "pancreas"}
SIZE_BINS = np.array([0, 10, 15, 20, 30, 40, 60, 90, 130, np.inf])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the v2 prompt-conditioned 3D U-Net")
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden", type=Path)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--batch-size", "--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--steps-per-epoch", type=int, default=250)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size-alpha", type=float, default=0.75)
    p.add_argument("--rot-z", type=float, default=0.35)
    p.add_argument("--rot-xy", type=float, default=0.10)
    p.add_argument("--scale-aug", type=float, default=0.15)
    p.add_argument("--shift-aug", type=float, default=6.0)
    p.add_argument("--band-w", type=float, default=3.0)
    p.add_argument("--band-k", type=int, default=2)
    p.add_argument("--slice-thickness-prob", type=float, default=0.5)
    p.add_argument("--slice-thickness-max", type=float, default=3.0)
    p.add_argument(
        "--intensity-affinity", action="store_true",
        help="add prompt-conditioned intensity-similarity as a fifth input channel",
    )
    p.add_argument("--init", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--organ-experts", action="store_true",
                   help="train organ-routed residual adapters on the winning model")
    p.add_argument("--expert-rank", type=int, default=8)
    p.add_argument("--router-loss-w", type=float, default=0.10)
    p.add_argument("--boundary-loss-w", type=float, default=0.15)
    p.add_argument("--freeze-shared-epochs", type=int, default=10)
    p.add_argument("--shared-lr", type=float, default=1e-5)
    p.add_argument("--glance-focus-moe", action="store_true",
                   help="train soft phenotype-routed coarse-to-fine Focus experts")
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--router-temperature", type=float, default=1.0)
    p.add_argument("--glance-loss-w", type=float, default=0.25)
    p.add_argument("--router-balance-w", type=float, default=0.02)
    return p.parse_args()


class ROICacheDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.records = json.loads((root / "manifest.json").read_text())
        self.n = len(self.records)
        done_path = root / "done.uint8"
        if not done_path.exists():
            raise RuntimeError(f"cache completion bitmap is missing: {done_path}")
        done = np.memmap(done_path, mode="r", dtype=np.uint8, shape=(self.n,))
        completed = int(np.count_nonzero(done))
        if completed != self.n:
            raise RuntimeError(
                f"cache is incomplete: {completed}/{self.n} entries are ready; "
                "finish roi3d.build_cache before training"
            )
        shape = V2_CACHE_SHAPE_ZYX
        self.images = np.memmap(root / "images.uint8", mode="r", dtype=np.uint8,
                                shape=(self.n, *shape))
        self.masks = np.memmap(
            root / "masks.packbits", mode="r", dtype=np.uint8,
            shape=(self.n, shape[0], shape[1], (shape[2] + 7) // 8),
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = torch.from_numpy(np.array(self.images[index], copy=True))
        packed = np.array(self.masks[index], copy=True)
        mask = np.unpackbits(packed, axis=-1, count=V2_CACHE_SHAPE_ZYX[-1])
        return {
            "image": image,
            "mask": torch.from_numpy(mask),
            "diameter": torch.tensor(record["diameter_true_mm"], dtype=torch.float32),
            "spacing": torch.tensor(record["roi_spacing_mm"], dtype=torch.float32),
            "axis": torch.tensor(record["axis_xy"], dtype=torch.float32),
            "domain": torch.tensor(
                DOMAIN_GROUPS.index(source_group(record["source"])), dtype=torch.long
            ),
        }


def source_group(source: str) -> str:
    lower = source.lower()
    for alias, group in SOURCE_ALIASES.items():
        if alias in lower:
            return group
    for key in TARGET_W:
        if key in lower:
            return key
    return "other"


def hidden_diameters(root: Path | None) -> np.ndarray:
    values = []
    if root is None:
        return np.empty(0)
    cache = root / ".roi3d_v2_diameters.npy"
    if cache.exists():
        return np.load(cache, allow_pickle=False)
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as case:
            recist = case["recist"]
            spacing = np.asarray(case["spacing"], dtype=np.float64)
        values.extend(g.diameter_true_mm for g in recist_geometries(recist, spacing))
    result = np.asarray(values, dtype=np.float64)
    temporary = root / ".roi3d_v2_diameters.tmp.npy"
    np.save(temporary, result, allow_pickle=False)
    temporary.replace(cache)
    return result


def sampling_weights(records: list[dict], hidden: np.ndarray,
                     size_alpha: float) -> torch.Tensor:
    groups = [source_group(r["source"]) for r in records]
    counts = Counter(groups)
    other_weight = 1.0 - sum(TARGET_W.values())
    domain_targets = dict(TARGET_W, other=other_weight)
    domain = np.array([domain_targets[g] / counts[g] for g in groups])
    train_d = np.asarray([r["diameter_true_mm"] for r in records])
    if len(hidden):
        train_hist = np.histogram(train_d, SIZE_BINS)[0].astype(float) + 1
        target_hist = np.histogram(hidden, SIZE_BINS)[0].astype(float) + 1
        train_hist /= train_hist.sum(); target_hist /= target_hist.sum()
        ratio = (target_hist / train_hist) ** size_alpha
        size = ratio[np.clip(np.digitize(train_d, SIZE_BINS) - 1, 0, len(ratio) - 1)]
    else:
        # Useful fallback matching the documented correction toward large lesions.
        size = np.clip(train_d / 20.0, 0.5, 4.0) ** size_alpha
    weights = domain * size
    weights /= weights.mean()
    return torch.from_numpy(weights).double()


def rotation_matrix(z_angle: torch.Tensor, y_angle: torch.Tensor,
                    x_angle: torch.Tensor) -> torch.Tensor:
    b = len(z_angle); device = z_angle.device
    matrices = []
    for angle, axes in ((z_angle, (1, 2)), (y_angle, (0, 2)), (x_angle, (0, 1))):
        result = torch.eye(3, device=device).repeat(b, 1, 1)
        c, s = angle.cos(), angle.sin(); a, d = axes
        result[:, a, a] = c; result[:, d, d] = c
        result[:, a, d] = -s; result[:, d, a] = s
        matrices.append(result)
    return matrices[0] @ matrices[1] @ matrices[2]


def augment_batch(batch: dict, device: torch.device, args: argparse.Namespace):
    image = batch["image"].to(device, non_blocking=True).float().unsqueeze(1) / 255.0
    target = batch["mask"].to(device, non_blocking=True).float().unsqueeze(1)
    diameter = batch["diameter"].to(device, non_blocking=True)
    spacing = batch["spacing"].to(device, non_blocking=True)
    axis = batch["axis"].to(device, non_blocking=True)
    domain = batch["domain"].to(device, non_blocking=True)
    b = image.shape[0]
    rand = lambda scale: (torch.rand(b, device=device) * 2 - 1) * scale
    rotation = rotation_matrix(rand(args.rot_z), rand(args.rot_xy), rand(args.rot_xy))
    scale = 1.0 + rand(args.scale_aug)
    flip = torch.where(torch.rand(b, 3, device=device) < 0.5, -1.0, 1.0)
    linear = rotation @ torch.diag_embed(scale[:, None] * flip)
    shift = ((torch.rand(b, 3, device=device) * 2 - 1) * args.shift_aug)

    in_shape = torch.tensor(V2_CACHE_SHAPE_ZYX, device=device, dtype=torch.float32)
    out_shape = torch.tensor(V2_SHAPE_ZYX, device=device, dtype=torch.float32)
    theta = torch.zeros(b, 3, 4, device=device)
    theta[:, :, :3] = (torch.diag(2 / (in_shape - 1))[None] @ linear @
                       torch.diag((out_shape - 1) / 2)[None])
    theta[:, :, 3] = shift * (2 / (in_shape - 1))
    grid = F.affine_grid(theta, (b, 1, *V2_SHAPE_ZYX), align_corners=True)
    image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)
    target = F.grid_sample(target, grid, mode="nearest", padding_mode="zeros",
                           align_corners=True)

    # The training corpus is substantially thinner-sliced than hidden data.
    # Blur and re-discretize image intensities along z while retaining the fine
    # target, matching the measured acquisition shift in REPRODUCE.md.
    max_factor = max(1, int(round(args.slice_thickness_max)))
    factors = torch.ones(b, dtype=torch.int64, device=device)
    selected = torch.rand(b, device=device) < args.slice_thickness_prob
    if max_factor > 1 and selected.any():
        factors[selected] = torch.randint(
            2, max_factor + 1, (int(selected.sum()),), device=device
        )
    for factor in range(2, max_factor + 1):
        indices = torch.where(factors == factor)[0]
        if not len(indices):
            continue
        coarse = F.avg_pool3d(
            image[indices], kernel_size=(factor, 1, 1),
            stride=(factor, 1, 1), ceil_mode=True,
        )
        image[indices] = F.interpolate(
            coarse, size=V2_SHAPE_ZYX, mode="nearest"
        )

    # Transform the physical RECIST endpoints by the inverse sampled affine.
    input_delta = torch.zeros(b, 3, device=device)
    input_delta[:, 1] = axis[:, 1] * diameter / spacing / 2
    input_delta[:, 2] = axis[:, 0] * diameter / spacing / 2
    inverse = torch.linalg.inv(linear)
    midpoint = (out_shape - 1) / 2 - torch.einsum("bij,bj->bi", inverse, shift)
    endpoint_delta = torch.einsum("bij,bj->bi", inverse, input_delta)
    start, end = midpoint - endpoint_delta, midpoint + endpoint_delta

    zz = torch.arange(V2_SHAPE_ZYX[0], device=device)[None, :, None, None]
    yy = torch.arange(V2_SHAPE_ZYX[1], device=device)[None, None, :, None]
    xx = torch.arange(V2_SHAPE_ZYX[2], device=device)[None, None, None, :]
    vector = end - start
    denom = vector.square().sum(1).clamp_min(1e-6)
    projection = (
        (zz - start[:, 0, None, None, None]) * vector[:, 0, None, None, None] +
        (yy - start[:, 1, None, None, None]) * vector[:, 1, None, None, None] +
        (xx - start[:, 2, None, None, None]) * vector[:, 2, None, None, None]
    ) / denom[:, None, None, None]
    projection.clamp_(0, 1)
    distance2 = (
        (zz - (start[:, 0, None, None, None] + projection * vector[:, 0, None, None, None])).square() +
        (yy - (start[:, 1, None, None, None] + projection * vector[:, 1, None, None, None])).square() +
        (xx - (start[:, 2, None, None, None] + projection * vector[:, 2, None, None, None])).square()
    )
    distance = distance2.sqrt()
    line = (distance <= 1.0).float().unsqueeze(1)
    prior = torch.exp(
        -(distance * spacing[:, None, None, None]) /
        (0.4 * diameter[:, None, None, None]).clamp_min(1e-3)
    ).unsqueeze(1)
    size_channel = (torch.log(diameter / 30.0) / 1.5)[:, None, None, None, None]
    size_channel = size_channel.expand(-1, 1, *V2_SHAPE_ZYX)
    channels = [image, line, prior, size_channel]
    if args.intensity_affinity:
        # RECIST supplies a sparse sample through the lesion. Give the network
        # an explicit, label-independent cue for voxels with similar CT
        # appearance. Initializing this channel at zero in old checkpoints lets
        # fine-tuning learn whether and where the cue is useful.
        reference = (image * line).flatten(2).sum(2) / line.flatten(2).sum(2).clamp_min(1)
        affinity = torch.exp(-torch.abs(image - reference[:, :, None, None, None]) / 0.12)
        channels.append(affinity)
    inputs = torch.cat(channels, 1)
    return (inputs.contiguous(memory_format=torch.channels_last_3d),
            target.contiguous(memory_format=torch.channels_last_3d), domain)


def dice_bce(logits: torch.Tensor, target: torch.Tensor,
             band_w: float, band_k: int) -> torch.Tensor:
    target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    if band_w:
        kernel = 2 * band_k + 1
        dilated = F.max_pool3d(target, kernel, stride=1, padding=band_k)
        eroded = 1 - F.max_pool3d(1 - target, kernel, stride=1, padding=band_k)
        weight = 1 + band_w * (dilated - eroded).clamp(0, 1)
    else:
        weight = None
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    probability = logits.sigmoid()
    intersection = (probability * target).flatten(1).sum(1)
    dice = 1 - ((2 * intersection + 1) /
                (probability.flatten(1).sum(1) + target.flatten(1).sum(1) + 1)).mean()
    return bce + dice


def boundary_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable one-voxel surface Dice loss for NSD-oriented tuning."""
    target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    probability = logits.sigmoid()
    target_boundary = (
        F.max_pool3d(target, 3, stride=1, padding=1)
        + F.max_pool3d(-target, 3, stride=1, padding=1)
    )
    probability_boundary = (
        F.max_pool3d(probability, 3, stride=1, padding=1)
        + F.max_pool3d(-probability, 3, stride=1, padding=1)
    )
    intersection = (probability_boundary * target_boundary).flatten(1).sum(1)
    denominator = (
        probability_boundary.flatten(1).sum(1)
        + target_boundary.flatten(1).sum(1)
    )
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def phenotype_targets(inputs: torch.Tensor, experts: int) -> torch.Tensor:
    """Derive reproducible size phenotypes from the supplied RECIST diameter."""
    log_size = inputs[:, 3, 0, 0, 0] * 1.5
    diameter = 30.0 * torch.exp(log_size)
    if experts == 4:
        boundaries = diameter.new_tensor((20.0, 40.0, 70.0))
    else:
        # Log-spaced bins cover 10--130 mm without using segmentation labels.
        boundaries = torch.logspace(
            math.log10(12.0), math.log10(100.0), experts - 1,
            device=diameter.device, dtype=diameter.dtype,
        )
    return torch.bucketize(diameter, boundaries)


def routing_balance_loss(logits: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
    """Match predicted batch usage to target usage without forcing uniformity."""
    probability = logits.softmax(1).mean(0).clamp_min(1e-6)
    target_frequency = F.one_hot(
        targets, num_classes=logits.shape[1]
    ).float().mean(0).clamp_min(1e-6)
    return F.kl_div(probability.log(), target_frequency, reduction="sum")


def main() -> None:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark = True
    dataset = ROICacheDataset(args.cache_root)
    hidden = hidden_diameters(args.hidden)
    weights = sampling_weights(dataset.records, hidden, args.size_alpha)
    samples = args.steps_per_epoch * args.batch_size
    sampler = WeightedRandomSampler(weights, samples, replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    in_channels = 5 if args.intensity_affinity else 4
    if args.organ_experts and args.glance_focus_moe:
        raise ValueError("choose either --organ-experts or --glance-focus-moe")
    if args.glance_focus_moe:
        raw_model = GlanceFocusMoE(
            in_channels=in_channels, base=args.base, experts=args.experts,
            expert_rank=args.expert_rank,
            router_temperature=args.router_temperature,
        )
    else:
        raw_model = RECISTUNet3D(
            in_channels=in_channels, base=args.base,
            organ_experts=args.organ_experts, expert_rank=args.expert_rank,
        )
    raw_model = raw_model.to(
        device, memory_format=torch.channels_last_3d
    )
    if args.organ_experts or args.glance_focus_moe:
        expert_parameters = []
        shared_parameters = []
        for name, parameter in raw_model.named_parameters():
            if name.startswith(("router.", "expert_", "focus_expert_", "glance_head.")):
                expert_parameters.append(parameter)
            elif name.startswith(("up1.", "up0.", "head.", "ds1.")):
                shared_parameters.append(parameter)
            else:
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW([
            {"params": expert_parameters, "lr": args.lr},
            {"params": shared_parameters, "lr": args.shared_lr},
        ], weight_decay=1e-5, fused=True)
    else:
        shared_parameters = []
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr,
                                      weight_decay=1e-5, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch, eta_min=args.lr / 100
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch = 0; global_step = 0
    checkpoint_path = args.resume or (args.output / "last.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"])
            global_step = int(checkpoint["global_step"])
    elif args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        first = "enc0.0.weight"
        if (args.intensity_affinity and first in state and
                state[first].shape[1] == 4 and raw_model.state_dict()[first].shape[1] == 5):
            widened = raw_model.state_dict()[first].clone()
            widened[:, :4] = state[first]
            widened[:, 4].zero_()
            state = dict(state)
            state[first] = widened
        missing, unexpected = raw_model.load_state_dict(
            state, strict=not (args.organ_experts or args.glance_focus_moe)
        )
        if args.organ_experts or args.glance_focus_moe:
            allowed = ("router.", "expert_", "focus_expert_", "glance_head.")
            if unexpected or any(not key.startswith(allowed) for key in missing):
                raise RuntimeError(f"incompatible expert initializer: missing={missing}, "
                                   f"unexpected={unexpected}")
    model = (torch.compile(raw_model, mode="reduce-overhead")
             if args.compile else raw_model)
    print(json.dumps({"parameters": parameter_count(raw_model), "lesions": len(dataset),
                      "batch_size": args.batch_size, "samples_per_epoch": samples,
                      "hidden_size_samples": len(hidden)}), flush=True)
    metrics = args.output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        if args.organ_experts or args.glance_focus_moe:
            train_shared = epoch >= args.freeze_shared_epochs
            for parameter in shared_parameters:
                parameter.requires_grad_(train_shared)
        model.train(); running = 0.0; router_correct = 0; router_total = 0
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader):
            inputs, target, domain = augment_batch(batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = (model(inputs, domain) if args.organ_experts
                           else model(inputs))
                segmentation_outputs = outputs[:3]
                loss = sum(
                    weight * dice_bce(output, target, args.band_w, args.band_k)
                    for weight, output in zip((1.0, 0.5, 0.25), segmentation_outputs)
                )
                if args.organ_experts:
                    loss = (
                        loss
                        + args.router_loss_w * F.cross_entropy(outputs[3], domain)
                        + args.boundary_loss_w * boundary_dice(outputs[0], target)
                    )
                    router_correct += int((outputs[3].argmax(1) == domain).sum())
                    router_total += len(domain)
                elif args.glance_focus_moe:
                    phenotype = phenotype_targets(inputs, args.experts)
                    router_logits, glance_logits = outputs[3], outputs[4]
                    loss = (
                        loss
                        + args.router_loss_w * F.cross_entropy(
                            router_logits, phenotype
                        )
                        + args.router_balance_w * routing_balance_loss(
                            router_logits, phenotype
                        )
                        + args.glance_loss_w * dice_bce(
                            glance_logits, target, args.band_w, args.band_k
                        )
                        + args.boundary_loss_w * boundary_dice(
                            outputs[0], target
                        )
                    )
                    router_correct += int(
                        (router_logits.argmax(1) == phenotype).sum()
                    )
                    router_total += len(phenotype)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running += float(loss.detach()); global_step += 1
            if step % 10 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} "
                      f"loss={float(loss.detach()):.5f} lr={scheduler.get_last_lr()[0]:.3g} "
                      f"shared={int(not (args.organ_experts or args.glance_focus_moe) or epoch >= args.freeze_shared_epochs)}",
                      flush=True)
        record = {"epoch": epoch + 1, "global_step": global_step,
                  "loss": running / len(loader), "seconds": time.time() - started,
                  "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
        if router_total:
            record["router_accuracy"] = router_correct / router_total
        print(json.dumps(record), flush=True)
        with metrics.open("a") as stream: stream.write(json.dumps(record) + "\n")
        state = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                 "global_step": global_step, "args": vars(args)}
        torch.save(state, args.output / "last.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, args.output / f"epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
