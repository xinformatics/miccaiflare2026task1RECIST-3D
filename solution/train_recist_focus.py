#!/usr/bin/env python3
"""Fine-tune GF-Screen's 3D Focus model on RECIST-centered lesion crops.

The ICLR 2026 GF-Screen model learns semantic pan-cancer segmentation from
sliding-window CT crops. Task 2 already supplies a RECIST location and instance
ID, so its Glance policy is unnecessary here. This trainer extracts a physical
96 x 96 x 192 mm crop around one prompted lesion, resamples it to the Focus
model's native 96 x 96 x 64 grid, and supervises only that lesion instance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT / "gf-screen"
sys.path.insert(0, str(GF_ROOT))

from models.GF_Screen_RL import GFScreen_RL  # noqa: E402


FOCUS_SHAPE = (96, 96, 64)  # model axes: in-plane y, in-plane x, axial z
FOCUS_SPACING = (1.0, 1.0, 3.0)
DOMAIN_GROUPS = ("liver", "pancreas", "kidney", "lung", "colon", "other")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "dataset/RECIST_train_npz")
    p.add_argument("--roi-cache", type=Path,
                   help="optional completed roi3d memmap cache for fast lesion loading")
    p.add_argument("--train-list", type=Path, default=ROOT / "splits/train.txt")
    p.add_argument("--val-list", type=Path, default=ROOT / "splits/holdout.txt")
    p.add_argument("--init", type=Path, default=GF_ROOT / "model_official.pt")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/recist-focus")
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val-limit", type=int, default=256,
                   help="maximum deterministic validation cases per epoch; 0 uses all")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="limit sampled training batches; 0 visits one sample per case")
    p.add_argument("--organ-conditioned", action="store_true",
                   help="append a constant one-hot anatomical-domain token to the image")
    return p.parse_args()


def domain_index(path: Path) -> int:
    source = path.parent.name.lower().replace("pancreatic", "pancreas")
    for index, group in enumerate(DOMAIN_GROUPS[:-1]):
        if group in source:
            return index
    return len(DOMAIN_GROUPS) - 1


def _read_split(root: Path, split: Path) -> list[Path]:
    names = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    paths = [root / (name if name.endswith(".npz") else f"{name}.npz") for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} split files are missing; first: {missing[0]}")
    return paths


def _centered_crop(array: np.ndarray, center_zyx: tuple[float, float, float],
                   size_zyx: tuple[int, int, int]) -> np.ndarray:
    """Extract a centered crop and zero-pad only where it crosses the volume."""
    starts = [int(round(c - (size - 1) / 2)) for c, size in zip(center_zyx, size_zyx)]
    ends = [start + size for start, size in zip(starts, size_zyx)]
    slices = tuple(slice(max(0, start), min(n, end))
                   for start, end, n in zip(starts, ends, array.shape))
    crop = array[slices]
    pads = tuple((max(0, -start), max(0, end - n))
                 for start, end, n in zip(starts, ends, array.shape))
    return np.pad(crop, pads, mode="constant")


def recist_focus_crop(path: Path, lesion_id: int | None = None,
                      augment: bool = False) -> tuple[torch.Tensor, torch.Tensor, int]:
    with np.load(path) as case:
        imgs = case["imgs"]
        gts = case["gts"]
        recist = case["recist"]
        spacing = np.asarray(case["spacing"], dtype=np.float32)

    ids = np.unique(recist)
    ids = ids[ids != 0].astype(int)
    if len(ids) == 0:
        raise ValueError(f"{path}: no RECIST lesion")
    if lesion_id is None:
        lesion_id = int(random.choice(ids.tolist()))
    elif lesion_id not in ids:
        raise ValueError(f"{path}: RECIST lesion {lesion_id} is absent")

    marker = np.argwhere(recist == lesion_id)
    center_zyx = tuple(marker.mean(axis=0).tolist())
    # NPZ spacing follows SimpleITK (x, y, z); arrays follow (z, y, x).
    physical_zyx = (FOCUS_SHAPE[2] * FOCUS_SPACING[2],
                    FOCUS_SHAPE[0] * FOCUS_SPACING[0],
                    FOCUS_SHAPE[1] * FOCUS_SPACING[1])
    spacing_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    size_zyx = tuple(max(1, int(math.ceil(mm / sp)))
                     for mm, sp in zip(physical_zyx, spacing_zyx))

    image_crop = _centered_crop(imgs, center_zyx, size_zyx)
    label_crop = _centered_crop((gts == lesion_id).astype(np.uint8), center_zyx, size_zyx)

    # GF-Screen/MONAI was trained in (x, y, z) spatial order. The two in-plane
    # axes are symmetric here, so transpose challenge (z,y,x) to (y,x,z).
    image = torch.from_numpy(np.ascontiguousarray(image_crop.transpose(1, 2, 0))).float()
    label = torch.from_numpy(np.ascontiguousarray(label_crop.transpose(1, 2, 0))).float()
    image = F.interpolate(image[None, None], size=FOCUS_SHAPE,
                          mode="trilinear", align_corners=False)[0]
    label = F.interpolate(label[None, None], size=FOCUS_SHAPE, mode="nearest")[0]
    image = image / 255.0

    if augment:
        for dim in (1, 2, 3):
            if random.random() < 0.5:
                image = torch.flip(image, (dim,))
                label = torch.flip(label, (dim,))
        if random.random() < 0.2:
            image = (image * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp_(0, 1)

    return image.contiguous(), label.long().contiguous(), lesion_id


class RECISTFocusDataset(Dataset):
    def __init__(self, paths: list[Path], training: bool):
        self.paths = paths
        self.training = training

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image, label, lesion_id = recist_focus_crop(
            self.paths[index], augment=self.training
        )
        return {"image": image, "label": label,
                "case": self.paths[index].stem, "lesion": lesion_id,
                "domain": domain_index(self.paths[index])}


class ROICacheFocusDataset(Dataset):
    """Read scale-normalized lesion crops from the existing ROI3D memmap."""
    SHAPE = (128, 192, 192)

    def __init__(self, root: Path, split: Path, training: bool):
        all_records = json.loads((root / "manifest.json").read_text())
        allowed = {line.strip().removesuffix(".npz")
                   for line in split.read_text().splitlines() if line.strip()}
        self.records = [r for r in all_records
                        if str(r["path"]).removesuffix(".npz") in allowed]
        self.training = training
        self.sources = [r["source"] for r in self.records]
        n = len(all_records)
        self.images = np.memmap(root / "images.uint8", mode="r", dtype=np.uint8,
                                shape=(n, *self.SHAPE))
        self.masks = np.memmap(
            root / "masks.packbits", mode="r", dtype=np.uint8,
            shape=(n, self.SHAPE[0], self.SHAPE[1], (self.SHAPE[2] + 7) // 8),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        source_index = int(record["index"])
        image = torch.from_numpy(np.array(self.images[source_index], copy=True))[None]
        packed = np.array(self.masks[source_index], copy=True)
        mask = np.unpackbits(packed, axis=-1, count=self.SHAPE[-1])
        label = torch.from_numpy(mask)[None]
        return {"image": image, "label": label,
                "case": Path(record["path"]).stem,
                "lesion": int(record["lesion_id"]),
                "domain": domain_index(Path(record["source"]))}


def prepare_cached_batch(image: torch.Tensor, target: torch.Tensor,
                         training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert cached Z,Y,X uint8 crops to Focus Y,X,Z tensors on the GPU."""
    image = image.float().permute(0, 1, 3, 4, 2) / 255.0
    target = target.float().permute(0, 1, 3, 4, 2)
    image = F.interpolate(image, size=FOCUS_SHAPE, mode="trilinear",
                          align_corners=False)
    target = F.interpolate(target, size=FOCUS_SHAPE, mode="nearest").long()
    if training:
        for dim in (2, 3, 4):
            selected = torch.rand(len(image), device=image.device) < 0.5
            image[selected] = torch.flip(image[selected], (dim,))
            target[selected] = torch.flip(target[selected], (dim,))
    return image, target


def _build_focus(init_path: Path, organ_conditioned: bool = False) -> torch.nn.Module:
    class Args:
        glance_backbone = "resnet18"
        roi_x, roi_y, roi_z = FOCUS_SHAPE
        in_channels, out_channels, feature_size = 1, 2, 48
        use_checkpoint = True
        batch_size, sw_batch_size = 1, 1

    full_model = GFScreen_RL(Args())
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    full_model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = full_model.segmentation_model
    if organ_conditioned:
        def widen(old: torch.nn.Conv3d) -> torch.nn.Conv3d:
            result = torch.nn.Conv3d(
                1 + len(DOMAIN_GROUPS), old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, dilation=old.dilation,
                groups=old.groups, bias=old.bias is not None,
            )
            with torch.no_grad():
                result.weight.zero_()
                result.weight[:, :1].copy_(old.weight)
                if old.bias is not None:
                    result.bias.copy_(old.bias)
            return result
        model.swinViT.patch_embed.proj = widen(model.swinViT.patch_embed.proj)
        model.encoder1.layer.conv1.conv = widen(model.encoder1.layer.conv1.conv)
        model.encoder1.layer.conv3.conv = widen(model.encoder1.layer.conv3.conv)
    return model


def add_domain_channels(image: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    tokens = F.one_hot(domain, num_classes=len(DOMAIN_GROUPS)).to(image.dtype)
    tokens = tokens[:, :, None, None, None].expand(-1, -1, *image.shape[2:])
    return torch.cat((image, tokens), dim=1)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             limit: int, organ_conditioned: bool = False) -> float:
    model.eval()
    dices: list[float] = []
    for index, batch in enumerate(loader):
        if limit and index * loader.batch_size >= limit:
            break
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        if image.dtype == torch.uint8:
            image, target = prepare_cached_batch(image, target, training=False)
        if organ_conditioned:
            image = add_domain_channels(
                image, batch["domain"].to(device, non_blocking=True)
            )
        target = target.bool()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(image).argmax(1, keepdim=True).bool()
        intersection = (pred & target).flatten(1).sum(1).float()
        denominator = pred.flatten(1).sum(1) + target.flatten(1).sum(1)
        dices.extend(((2 * intersection + 1) / (denominator + 1)).cpu().tolist())
    return float(np.mean(dices))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    args.output.mkdir(parents=True, exist_ok=True)

    if args.roi_cache:
        train_ds = ROICacheFocusDataset(args.roi_cache, args.train_list, training=True)
        val_ds = ROICacheFocusDataset(args.roi_cache, args.val_list, training=False)
        sources = train_ds.sources
    else:
        train_paths = _read_split(args.data, args.train_list)
        val_paths = _read_split(args.data, args.val_list)
        train_ds = RECISTFocusDataset(train_paths, training=True)
        val_ds = RECISTFocusDataset(val_paths, training=False)
        sources = [path.parent.name for path in train_paths]
    counts = Counter(sources)
    weights = torch.tensor([counts[source] ** -0.5 for source in sources])
    samples = (args.steps_per_epoch * args.batch_size
               if args.steps_per_epoch else len(weights))
    sampler = WeightedRandomSampler(weights, num_samples=samples, replacement=True)
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, **common)

    device = torch.device("cuda:0")
    model = _build_focus(args.init, args.organ_conditioned).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["segmentation_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_dice = float(checkpoint.get("best_patch_dice", -1))

    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            if image.dtype == torch.uint8:
                image, target = prepare_cached_batch(image, target, training=True)
            if args.organ_conditioned:
                image = add_domain_channels(
                    image, batch["domain"].to(device, non_blocking=True)
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(image)
                loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            if step % 50 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(train_loader)} "
                      f"loss={float(loss):.5f}", flush=True)

        patch_dice = validate(
            model, val_loader, device, args.val_limit, args.organ_conditioned
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": running / len(train_loader),
            "val_patch_dice": patch_dice,
            "seconds": time.time() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        print(json.dumps(record), flush=True)
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        state = {
            "epoch": epoch + 1,
            "best_patch_dice": max(best_dice, patch_dice),
            "segmentation_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(state, args.output / f"checkpoint_{epoch + 1}.pt")
        torch.save(state, args.output / "checkpoint.pt")
        if patch_dice > best_dice:
            best_dice = patch_dice
            torch.save(state, args.output / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
