"""EfficientMedSAM2 inference for CT lesion segmentation from RECIST prompts.

Input NPZ keys: `imgs` (D,H,W uint8), `recist` (D,H,W labelled line), and either
`gts` or `gts_hidden` (evaluation only -- never a prompt source).

Quick start
-----------
    # local weights, GPU, box prompt
    python eff_medsam2_infer_CT_lesion_npz_recist.py \
        --weights /path/to/checkpoint.pt --model-size tiny --device cuda \
        -i /data/npz -o /out --prompt box
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from glob import glob
from os.path import basename, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from scipy.ndimage import generate_binary_structure, label as cc_label
# huggingface_hub is imported lazily in build_predictor(): it is only needed to
# download a baseline, and the benchmark docker images do not ship it -- a
# top-level import made every `--weights` run fail there with ModuleNotFoundError.

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor_npz,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════ configuration ══════════════════════════════
# Everything tunable lives in this block. Three kinds of entry:
#   DEFAULT_*  the default of a CLI flag with the same name; the flag overrides it.
#   plain name a setting with NO flag -- edit it here. The geometry knobs are
#              deliberately not on the CLI: they are properties of the RECIST
#              raster, identical for every dataset we have, and were each
#              validated once (see the block comments and the module docstring).
#   the rest   fixed contracts of the model or the NPZ format; do not tune.

# ── reproducibility / runtime ──
SEED = 2026                       # torch, numpy, random (the overlay slice pick)

# ── model ──
DEFAULT_MODEL_SIZE = "tiny"      # when neither --model-size nor --checkpoint says
DEFAULT_DEVICE = "cuda"           # the original script hard-forced cpu
# Where the per-size configs live when --cfg is omitted; relative to THIS file, so
# it resolves the same however you invoke the script. --cfg overrides both.
CFG_ROOT = "efficient_track_anything/configs"
CFG_BY_SIZE = {"small": "efficienttam_s_512x512.yaml",
               "tiny": "efficienttam_ti_512x512.yaml"}
HF_REPO_ID = "wanglab/MedSAM2"
HF_BY_SIZE = {"small": "eff_medsam2_small_FLARE25_RECIST_baseline.pt",
              "tiny": "eff_medsam2_tiny_FLARE25_RECIST_baseline.pt"}
# Script-relative (not cwd-relative), so downloads land in the repo's own
# checkpoints/ dir -- same place as `bash download.sh` -- however you invoke this.
CKPT_CACHE_DIR = join(_HERE, "checkpoints")

# ── image preprocessing: the EfficientTAM input contract, not a tunable ──
IMAGE_SIZE = 512                  # default network input side
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ── prompt geometry ──
DEFAULT_PROMPT = "box"
PROMPT_CHOICES = ("box", "box3points", "1point", "3points", "5points", "5points_10_90",
                  "npoints", "grid9")
DEFAULT_SHIFT = 0                 # extra px added to each box side after scaling

# ── geometry settings: NO CLI flags, edit here ─────────────────────────────
# These describe the RECIST raster, not the run. Every dataset we have measures the
# same values, so exposing them as flags only invited a wrong value on the command
# line; the numbers below are the validated ones.
N_POINTS = 5                      # `--prompt npoints` only; named modes ignore it
BOX_SCALE = 1.0                   # box side = BOX_SCALE x diameter. 1.0 = upstream.
Z_SLAB_SCALE = 1.6                # z extent = scale x diameter in slices (was 1.0).
                                  # A lesion usually spans more slices than its
                                  # longest in-plane diameter implies, and the slab
                                  # hard-bounds how far propagation can reach.
# How the z half-extent comes from the in-plane diameter. NPZ spacing is sitk
# (x, y, z), which is what the two wrong variants get backwards:
#   correct = diameter * (mean(sp[0],sp[1]) / sp[2]) / 2
#             dimensionally right: physical length / slice thickness
#   repo    = diameter * (mean(sp[1],sp[2]) / sp[0]) / 2
#             this file's original; wrong axes, and on 5-6 mm slices the slab
#             covers the WHOLE volume
#   docker  = diameter / (mean(sp[1],sp[2]) / sp[0]) / 2
#             the variant baked into the benchmark docker images, i.e. the one
#             that produced every historical result. Set this ONLY to reproduce
#             a historical run.
SLAB_FORMULA = "correct"
SLAB_FORMULA_CHOICES = ("correct", "repo", "docker")

# Axial over-measure removed from the diameter, in px -- a number, or "auto".
# DESPITE THE HISTORICAL NAME this is NOT the stroke width: cv2.line caps both ends
# of a thick segment with a filled disc of radius r = ceil(thickness/2), so the
# raster's axial span exceeds the clinical diameter by the two caps, 2r. It equals
# the nominal thickness only at thickness=2 (2r = 2); a thickness-3 line needs 4.
# 2.0 is calibrated for our generators (RECIST_LINE_THICKNESS=2): measured
# over-measure +1.86 px median on 221 EAY labels, every one > 1 px too long, and
# subtracting 2.0 puts 100% of labels within +/-1 px of the generator's own
# longest-diameter value, midpoint untouched (max shift 0.013 px). 0 disables.
SUBTRACT_THICKNESS_SETTING = 2.0

# CONTROL ONLY. True reproduces the pre-fix geometry exactly -- raster-order
# endpoints (coords[0]/coords[-1]) and the old truncating box math, for BOTH the box
# and the diameter the z-slab is derived from -- and forces the over-measure to 0.
# For A/B-ing the endpoint fix against a historical run; never for a real result.
LEGACY_GEOMETRY = False

# SUBTRACT_THICKNESS_SETTING = "auto": recover the over-measure from the raster
# itself, for NPZs whose generator (and thus line thickness) we do not control.
#
# What is measured: the rendered band width `w`, as area/length over the middle 30%
# of the line (the two round end caps are excluded -- they are what inflate a naive
# global max-min). Then `over_measure = w - 1`.
#
# Why w - 1, exactly: cv2.line rounds the half-width UP to a whole pixel row,
# r = ceil(t/2), fills rows -r..+r, and caps both ends with a filled disc of radius
# r. So the rendered width is 2r+1 px (always odd; t=3 and t=4 render identically)
# and the axial over-measure is the two caps = 2r = w - 1. The -1 is the centre row:
# geometry, not a fitted constant. Verified on synthetic rasters t=1..10:
#   t          1     2     3     4     5     6     7     8     9    10
#   w       0.95  3.00  5.00  5.00  7.00  7.00  9.00  9.00 10.95 10.95
#   w-1     0.00  2.00  4.00  4.00  6.00  6.00  8.00  8.00  9.95  9.95
#   true   -0.01  1.90  3.59  3.62  5.61  5.73  7.44  7.54  9.82  9.88
# so the error is +0.1..+0.56 px, always over-subtracting slightly.
#
# `w - 1` is then SNAPPED to the nearest even integer, because 2r is even by
# construction -- that is a constraint, not a fudge, and it absorbs the residual
# measurement wobble. Our three datasets measure w = 2.87 / 2.90 / 2.95, i.e.
# 1.87..1.95 raw, all snapping to 2.0: auto is a strict NO-OP on data we own.
#
# MUST be aggregated over the dataset, never used per label: the rendered width
# genuinely oscillates with orientation (t=2 reads 2.15 at 25 deg and 3.60 at 45 deg;
# t=3 reads 3.50 at 45 deg and 5.30 at 20 deg) and with the sub-pixel phase. Those
# are real rasterisation differences, not measurement noise, and no observable
# predicts them -- 80% of the per-label variance comes from the int() truncation of
# the endpoints, which the raster no longer carries. The median over many labels
# averages the wobble out and returns 2r+1 exactly.
AUTO_THICKNESS = "auto"           # the literal accepted by the setting above
AUTO_THICKNESS_OFFSET = 1.0       # the centre row
AUTO_THICKNESS_PROBE = 50         # NPZs pre-scanned to build the median
AUTO_THICKNESS_MIN_LEN = 6.0      # px; shorter labels have no usable middle slab
AUTO_THICKNESS_CORE = (0.35, 0.65)  # the along-axis slab, as a fraction of length
AUTO_THICKNESS_MIN_LABELS = 20     # below this the median is too noisy to trust
AUTO_THICKNESS_FALLBACK = 2.0      # used if fewer labels than that are measurable

# Along-line percentiles per prompt mode. These MATCH
# Interctive_Model_benchmarking/scripts/prompts/prompt_utils.py so numbers from this
# script are comparable with the benchmark leaderboard:
#   3points -> benchmark `3pt_25`     (25 / 50 / 75)
#   5points -> benchmark `5pt_25_42`  (7.5 / 25 / 50 / 75 / 92.5)  <- its default
#   5points_10_90 -> benchmark `5pt_10_90` (more interior, avoids the near-edge)
# 0 / 100 are deliberately never used: raw RECIST endpoints sit on the lesion
# boundary and leak into surrounding tissue (benchmark measured endpoint 3 pt at
# 0.644 vs center-out 0.749 on nnInteractive).
PROMPT_PERCENTILES = {
    "1point": (50.0,),
    "3points": (25.0, 50.0, 75.0),
    "5points": (7.5, 25.0, 50.0, 75.0, 92.5),
    "5points_10_90": (10.0, 30.0, 50.0, 70.0, 90.0),
}

# ── output ──
DEFAULT_NUM_WORKERS = 2
OVERLAY_DPI = 300
OVERLAY_MASK_COLOR = (251 / 255, 252 / 255, 30 / 255)
OVERLAY_MASK_ALPHA = 0.5

# ── legacy point-mode names -> --prompt ──
_LEGACY_SAMPLE_POINTS = {
    "from_recist_center": "1point",
    "from_recist_3": "3points",
    "from_recist_n": "5points",
    "from_box": "grid9",
}

# ── resolved in main() from SUBTRACT_THICKNESS_SETTING (which may say "auto") and
#    read by the geometry helpers, so the control path is the SAME code path,
#    differing only in endpoint/rounding rules ──
SUBTRACT_THICKNESS = 0.0          # px, resolved at startup

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("high")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def validate_settings() -> None:
    """Fail fast on a bad edit to the configuration block."""
    if BOX_SCALE <= 0:
        raise SystemExit(f"BOX_SCALE must be > 0, got {BOX_SCALE!r}")
    if Z_SLAB_SCALE <= 0:
        raise SystemExit(f"Z_SLAB_SCALE must be > 0, got {Z_SLAB_SCALE!r}")
    if N_POINTS < 1:
        raise SystemExit(f"N_POINTS must be >= 1, got {N_POINTS!r}")
    if SLAB_FORMULA not in SLAB_FORMULA_CHOICES:
        raise SystemExit(f"SLAB_FORMULA must be one of {SLAB_FORMULA_CHOICES}, "
                         f"got {SLAB_FORMULA!r}")
    if SUBTRACT_THICKNESS_SETTING != AUTO_THICKNESS:
        try:
            v = float(SUBTRACT_THICKNESS_SETTING)
        except (TypeError, ValueError):
            raise SystemExit("SUBTRACT_THICKNESS_SETTING must be a number or "
                             f"{AUTO_THICKNESS!r}, got "
                             f"{SUBTRACT_THICKNESS_SETTING!r}") from None
        if v < 0:
            raise SystemExit(f"SUBTRACT_THICKNESS_SETTING must be >= 0, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    m = p.add_argument_group("model")
    m.add_argument("--weights", type=str, default=None,
                   help="path to a local .pt checkpoint. Accepts a raw training "
                        "checkpoint (a dict with a 'model' key) or an inference "
                        "checkpoint. If omitted, the published baseline for "
                        "--model-size is downloaded from HuggingFace.")
    m.add_argument("--model-size", "--size", dest="model_size",
                   choices=("small", "tiny"), default=None,
                   help="EfficientTAM variant; selects the config. Required with "
                        "--weights unless --checkpoint is given (default: small).")
    m.add_argument("--cfg", type=str, default=None, metavar="YAML",
                   help="EfficientTAM config to compose: a path to a .yaml, "
                        "absolute or relative to the cwd or to this file. Use it to "
                        "test a config that is not one of the two shipped ones. "
                        "Omit it and the config for --model-size is used "
                        "(CFG_ROOT/CFG_BY_SIZE, i.e. "
                        "efficient_track_anything/configs/efficienttam_{s,ti}_512x512"
                        ".yaml). Must be a file, not a directory. The path actually "
                        "composed is echoed on the [model] line. NOTE the "
                        "config must match the checkpoint: a config whose trunk "
                        "differs from the weights fails in load_state_dict."),
    m.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                   help="cuda | cuda:N | cpu  (default cuda; the original script "
                        "hard-forced cpu)")
    m.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile (useful for short validation runs)")
    m.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help="square network input side (default: 512). Must be a "
                        "multiple of the 16-pixel ViT patch size. Lower values "
                        "are an explicit speed/accuracy tradeoff.")

    io = p.add_argument_group("data")
    io.add_argument("-i", "--imgs-path", "--imgs_path", dest="imgs_path",
                    type=str, required=True, help="directory of input NPZs")
    io.add_argument("-o", "--pred-save-dir", "--pred_save_dir", dest="pred_save_dir",
                    type=str, required=True, help="output directory")
    io.add_argument("--gts-path", "--gts_path", dest="gts_path", default=None,
                    help="optional separate directory holding the GT NPZs")
    io.add_argument("--anchor-mask-dir", type=str, default=None,
                    help="optional directory of NPZ predictions whose `segs` mask "
                         "on each RECIST slice is used as SAM2's propagation anchor")
    io.add_argument("--gt-key", dest="gt_key", default="auto",
                    choices=("auto", "gts", "gts_hidden"),
                    help="GT key to copy through; 'auto' takes whichever exists. "
                         "GT is never used as a prompt source.")
    io.add_argument("--file-list", type=str, default=None,
                    help="optional list of paths relative to --imgs-path, with or "
                         "without .npz (useful for fixed holdout evaluation)")

    pr = p.add_argument_group("prompt")
    pr.add_argument("--prompt", choices=PROMPT_CHOICES, default=None,
                    help="box | box3points | 1point | 3points | 5points | 5points_10_90 | npoints | "
                         "grid9 (default: box). Point geometry matches the "
                         "benchmark: 3points=25/50/75, 5points=7.5/25/50/75/92.5.")
    pr.add_argument("--shift", type=int, default=DEFAULT_SHIFT,
                    help="extra pixels added to each box side after scaling")
    pr.add_argument("--box-scale", type=float, default=BOX_SCALE,
                    help="box side as a multiple of the RECIST diameter")
    pr.add_argument("--z-slab-scale", type=float, default=Z_SLAB_SCALE,
                    help="physical z extent as a multiple of RECIST diameter")
    pr.add_argument("--max-planned-steps", type=int, default=120,
                    help="if a case exceeds this deterministic propagation "
                         "budget, use --fallback-z-slab-scale; 0 disables")
    pr.add_argument("--fallback-z-slab-scale", type=float, default=1.0,
                    help="physical slab scale for over-budget cases")
    pr.add_argument("--slab-formula", choices=SLAB_FORMULA_CHOICES,
                    default=SLAB_FORMULA,
                    help="spacing conversion; use 'correct' except for controls")
    pr.add_argument("--subtract-thickness", default=str(SUBTRACT_THICKNESS_SETTING),
                    help="RECIST line-cap over-measure in pixels, or 'auto'")
    pr.add_argument("--state-mode", choices=("shared", "cropped"), default="shared",
                    help="shared reuses one full-volume state and image features "
                         "across lesions; cropped reproduces the slower legacy "
                         "per-lesion state construction")
    pr.add_argument("--empty-stop-patience", type=int, default=3,
                    help="stop one propagation direction after this many "
                         "consecutive empty masks; 0 disables early stopping")

    out = p.add_argument_group("output")
    out.add_argument("--save-nifti", "--save_nifti", dest="save_nifti",
                     action="store_true")
    out.add_argument("--save-overlay", "--save_overlay", dest="save_overlay",
                     action="store_true")
    out.add_argument("--num-workers", "--num_workers", dest="num_workers",
                     type=int, default=DEFAULT_NUM_WORKERS)
    out.add_argument("--debug-output", action="store_true",
                     help="include GT/boxes/spacing in NPZs for local analysis; "
                          "the default writes the required 'segs' key only")
    out.add_argument("--no-component-filter", action="store_true",
                     help="disable prompt-anchored 3D connected-component cleanup")

    leg = p.add_argument_group("legacy (kept for backward compatibility)")
    leg.add_argument("--checkpoint", type=str, default=None,
                     help="old form of --model-size ('small' / 'tiny'). A path is "
                          "also accepted and treated as --weights.")
    leg.add_argument("--propagate_with_box", action="store_true",
                     help="old flag; equivalent to --prompt box")
    leg.add_argument("--sample_points", type=str, default=None,
                     choices=tuple(_LEGACY_SAMPLE_POINTS),
                     help="old point-mode names, mapped onto --prompt")
    return p


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fold the legacy flags into the new ones and fill defaults."""
    # --checkpoint may be a size or a path
    if args.checkpoint:
        if args.checkpoint in ("small", "tiny"):
            args.model_size = args.model_size or args.checkpoint
        elif os.path.exists(args.checkpoint):
            args.weights = args.weights or args.checkpoint
        else:
            raise SystemExit(f"--checkpoint {args.checkpoint!r} is neither "
                             f"'small'/'tiny' nor an existing path")

    # prompt: explicit --prompt wins, then legacy flags, then default box.
    if args.prompt is None:
        if args.sample_points and not args.propagate_with_box:
            args.prompt = _LEGACY_SAMPLE_POINTS[args.sample_points]
        elif args.sample_points and args.propagate_with_box:
            raise SystemExit(
                "both --propagate_with_box and --sample_points were given, which is "
                "ambiguous. In the original script the box branch silently won. "
                "Pass --prompt explicitly.")
        elif args.propagate_with_box:
            args.prompt = "box"
        else:
            args.prompt = DEFAULT_PROMPT

    if args.model_size is None:
        args.model_size = DEFAULT_MODEL_SIZE
    if args.empty_stop_patience < 0:
        raise SystemExit("--empty-stop-patience must be >= 0")
    if args.max_planned_steps < 0:
        raise SystemExit("--max-planned-steps must be >= 0")
    if args.z_slab_scale <= 0 or args.fallback_z_slab_scale <= 0:
        raise SystemExit("z slab scales must be > 0")
    if args.image_size <= 0 or args.image_size % 16:
        raise SystemExit("--image-size must be a positive multiple of 16")
    return args


# ───────────────────────────── RECIST geometry ─────────────────────────────
def recist_endpoints(recist_per_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True endpoints of the rasterised RECIST line, as float32 [x, y].

    Direction and extent come from the line's PRINCIPAL AXIS (SVD), which is
    robust to two separate raster artefacts:

    1. `coords[0]` / `coords[-1]` of `np.where` are raster-scan order, not
       geometry. The raster is ~2-3 px thick, so on a shallow-slope line the
       extremal-y rows are partial segments whose inner ends sit MID-LINE -- the
       measured diameter collapses (3.2 px for a true 57.0 px line). Projecting
       onto the principal axis cannot do that.
    2. `cv2.line(thickness=2)` extends past the true endpoints by ~the thickness,
       so the axial span OVER-measures the clinical diameter by a near-constant
       +1.86 px (median over 221 EAY labels; every label > 1 px long). We remove
       it via SUBTRACT_THICKNESS, which brings 100% of labels within +/-1 px of
       the generator's own value.

    The midpoint is taken on-axis (not as the mean of two extremal pixels, whose
    orthogonal offsets would tilt it).
    """
    ys, xs = np.where(recist_per_lab > 0)
    if len(xs) < 2:
        raise ValueError("RECIST line must contain at least two pixels")
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if LEGACY_GEOMETRY:                      # pre-fix behaviour, for the control run
        return coords[0], coords[-1]
    if len(coords) == 2:
        p1, p2 = coords[0], coords[1]
    else:
        centroid = coords.mean(0)
        c0 = coords - centroid
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        u = vt[0].astype(np.float32)          # unit principal direction
        proj = c0 @ u
        lo, hi = float(proj.min()), float(proj.max())
        mid = centroid + ((hi + lo) / 2.0) * u
        half = (hi - lo) / 2.0
        p1, p2 = mid - half * u, mid + half * u

    if SUBTRACT_THICKNESS > 0:
        v = p2 - p1
        length = float(np.linalg.norm(v))
        if length > SUBTRACT_THICKNESS:       # never invert a very short line
            mid = (p1 + p2) / 2.0
            u = v / length
            half = (length - SUBTRACT_THICKNESS) / 2.0
            p1, p2 = mid - half * u, mid + half * u
    return p1.astype(np.float32), p2.astype(np.float32)


def recist_diameter(recist_per_lab: np.ndarray) -> float:
    p1, p2 = recist_endpoints(recist_per_lab)
    return float(np.linalg.norm(p1 - p2))


def raster_width(recist_per_lab: np.ndarray) -> float | None:
    """Rendered band width of one rasterised RECIST line, in px, or None if unusable.

    area / length over the middle AUTO_THICKNESS_CORE of the principal axis, so the
    two round end caps are excluded. Returns 2r+1 for a cv2.line of thickness t
    (r = ceil(t/2)) when averaged over orientations -- see AUTO_THICKNESS_OFFSET.
    """
    ys, xs = np.where(recist_per_lab > 0)
    if len(xs) < 8:
        return None
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    c0 = coords - coords.mean(0)
    _, _, vt = np.linalg.svd(c0, full_matrices=False)
    along = c0 @ vt[0]
    length = float(along.max() - along.min())
    if length < AUTO_THICKNESS_MIN_LEN:
        return None
    lo, hi = AUTO_THICKNESS_CORE
    s = along - along.min()
    core = (s > lo * length) & (s < hi * length)
    n_core = int(core.sum())
    if n_core < 3:
        return None
    return n_core / ((hi - lo) * length)


def estimate_axial_over_measure(files) -> tuple[float, float, int]:
    """Dataset-level `--subtract-RECIST-thickness auto`.

    Returns (over_measure, median_width, n_labels). Falls back to 2.0 (the value
    calibrated for our own generators) if too few labels are measurable.
    """
    widths = []
    for f in files[:AUTO_THICKNESS_PROBE]:
        try:
            recist = np.load(f, "r", allow_pickle=True)["recist"]
        except (KeyError, OSError, ValueError) as e:
            print(f"[auto] skipping {basename(f)}: {e}")
            continue
        for ulab in np.unique(recist):
            if ulab == 0:
                continue
            z_indices = np.unique(np.where(recist == ulab)[0])
            for zi in z_indices:
                w = raster_width(recist[int(zi)] == ulab)
                if w is not None:
                    widths.append(w)
    if len(widths) < AUTO_THICKNESS_MIN_LABELS:
        print(f"[auto] only {len(widths)} measurable label(s) (need "
              f"{AUTO_THICKNESS_MIN_LABELS}); falling back to "
              f"{AUTO_THICKNESS_FALLBACK}")
        return AUTO_THICKNESS_FALLBACK, float("nan"), len(widths)
    w_med = float(np.median(widths))
    # The answer is the two end caps, 2r, so it is an EVEN integer by construction
    # (r = ceil(thickness/2) whole pixel rows). Snapping to it removes the residual
    # measurement wobble instead of passing it through: our datasets measure
    # w = 2.87..2.95 -> 1.87..1.95 raw -> 2.0 snapped, i.e. exactly the default.
    # floor(x + 0.5) rather than round(), which is banker's rounding in Python.
    raw = max(0.0, w_med - AUTO_THICKNESS_OFFSET)
    return 2.0 * math.floor(raw / 2.0 + 0.5), w_med, len(widths)


def recist_bbox(recist_per_lab: np.ndarray, scale: float = 1.6,
                shift: int = 0) -> np.ndarray:
    """Square box centred on the RECIST midpoint, side = scale x diameter.

    Returns [x_min, y_min, x_max, y_max] clipped to the image. The box is square by
    construction (both axes use the same half-side), so its size never depended on
    the line's orientation -- only on the diameter, which is why the endpoint fix
    above matters here.
    """
    h, w = recist_per_lab.shape
    p1, p2 = recist_endpoints(recist_per_lab)
    if LEGACY_GEOMETRY:                      # old truncating math
        c = ((p1 + p2) / 2).astype(int)
        hs = int(float(np.linalg.norm(p1 - p2)) * scale / 2)
        return np.array([max(0, c[0] - hs - shift), max(0, c[1] - hs - shift),
                         min(w - 1, c[0] + hs + shift), min(h - 1, c[1] + hs + shift)])
    center = (p1 + p2) / 2.0
    half = (float(np.linalg.norm(p1 - p2)) * scale) / 2.0
    x_min = int(round(center[0] - half)) - shift
    x_max = int(round(center[0] + half)) + shift
    y_min = int(round(center[1] - half)) - shift
    y_max = int(round(center[1] + half)) + shift
    return np.array([max(0, x_min), max(0, y_min),
                     min(w - 1, x_max), min(h - 1, y_max)])


def recist_points_at(recist_per_lab: np.ndarray, percentiles) -> np.ndarray:
    """Points at the given along-line percentiles (0 = endpoint1, 50 = mid, 100 = endpoint2).

    Ordered centre-first (by |p - 50|), matching the benchmark helper -- irrelevant
    for SAM2 (one forward pass consumes all points) but kept so the two
    implementations are literally comparable. Returns (n, 2) float32 [x, y].
    """
    p1, p2 = recist_endpoints(recist_per_lab)
    mid = (p1 + p2) / 2.0
    pts = []
    for p in sorted(percentiles, key=lambda q: abs(q - 50.0)):
        f = (p - 50.0) / 50.0                 # -1 .. +1
        end = p2 if f >= 0 else p1
        pts.append(mid + abs(f) * (end - mid))
    return np.stack(pts).astype(np.float32)


def recist_points(recist_per_lab: np.ndarray, n: int) -> np.ndarray:
    """`n` evenly spaced interior points -- the generic `--prompt npoints` rule.

    Percentiles are `linspace(0, 100, n+2)[1:-1]`, so n=3 gives 25/50/75 (identical
    to the named `3points` mode). For n=5 this is NOT the same as `5points`: the
    named mode follows the benchmark's 7.5/25/50/75/92.5 instead.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return recist_points_at(recist_per_lab, np.linspace(0.0, 100.0, n + 2)[1:-1])


def sample_points_in_bbox_grid(bbox: np.ndarray, n: int) -> np.ndarray:
    """Uniform grid of n (x, y) points inside the bbox (legacy `from_box` mode)."""
    x_min, y_min, x_max, y_max = bbox
    g = int(np.ceil(np.sqrt(n)))
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, g, dtype=int),
                         np.linspace(y_min, y_max, g, dtype=int))
    return np.stack([xv.ravel(), yv.ravel()], axis=1)[:n]


def prompt_points(recist_per_lab: np.ndarray, prompt: str) -> np.ndarray:
    if prompt in PROMPT_PERCENTILES:
        return recist_points_at(recist_per_lab, PROMPT_PERCENTILES[prompt])
    if prompt == "npoints":
        return recist_points(recist_per_lab, N_POINTS)
    if prompt == "grid9":
        return sample_points_in_bbox_grid(
            recist_bbox(recist_per_lab, scale=BOX_SCALE), n=9)
    raise ValueError(f"not a point prompt: {prompt}")


# ─────────────────────────────── image utils ───────────────────────────────
def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    """(d, h, w) grayscale -> (d, 3, image_size, image_size)."""
    d = array.shape[0]
    out = np.empty((d, 3, image_size, image_size), dtype=np.uint8)
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        out[i] = np.array(img.resize((image_size, image_size))).transpose(2, 0, 1)
    return out


def show_mask(mask, ax, mask_color=None, alpha=OVERLAY_MASK_ALPHA):
    color = (np.concatenate([mask_color, np.array([alpha])], axis=0)
             if mask_color is not None
             else np.array([*OVERLAY_MASK_COLOR, alpha]))
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_box(box, ax, edgecolor="blue"):
    x0, y0 = box[0], box[1]
    ax.add_patch(plt.Rectangle((x0, y0), box[2] - box[0], box[3] - box[1],
                               edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))


def load_gt(npz_data, gt_key: str, npz_name: str, gts_path: str | None):
    src = npz_data
    if gts_path is not None:
        src = np.load(join(gts_path, npz_name), "r", allow_pickle=True)
    keys = getattr(src, "files", list(src.keys()))
    if gt_key == "auto":
        for k in ("gts", "gts_hidden"):
            if k in keys:
                return src[k], k
        # Hidden test inputs legitimately contain no labels. Inference must never
        # depend on GT, so absence is represented explicitly instead of failing.
        return None, None
    if gt_key not in keys:
        raise KeyError(f"{npz_name}: missing GT key {gt_key!r}; found {keys}")
    return src[gt_key], gt_key


def keep_prompt_component(mask: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    """Keep 3D components touching the RECIST marker (or the nearest component).

    Video propagation occasionally produces disconnected false positives.  The
    marker is guaranteed to identify the target lesion, making this a fully
    automatic and rules-compliant filter. The raster is not forced into the mask:
    its rounded caps can extend just outside the manual lesion boundary.
    """
    prompt = prompt.astype(bool, copy=False)
    components, count = cc_label(mask, structure=generate_binary_structure(3, 2))
    if count <= 1 or count == 0:
        return mask
    touching = np.unique(components[prompt])
    touching = touching[touching != 0]
    if touching.size:
        return np.isin(components, touching)
    # Defensive fallback for malformed/empty prompts: retain the largest object.
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(sizes.argmax())


# ──────────────────────────────── inference ────────────────────────────────
def resolve_cfg(cfg: str | None, model_size: str) -> str:
    """Absolute hydra config name for the config to compose.

    `cfg` is a path to a .yaml -- absolute, or relative (tried against the cwd
    first, then against this file's directory). None -> the shipped config for
    `model_size`, CFG_ROOT/CFG_BY_SIZE[model_size].

    hydra's compose() reads a leading '/' as an absolute config name, which is why
    the result starts with one -- preserved from the original script. The shipped
    configs are self-contained (no `defaults:`), so a yaml outside the package
    directory composes fine.
    """
    if cfg is None:
        path = join(_HERE, CFG_ROOT, CFG_BY_SIZE[model_size])
    else:
        tried = [cfg] if os.path.isabs(cfg) else [os.path.abspath(cfg),
                                                 join(_HERE, cfg)]
        path = next((c for c in tried if os.path.exists(c)), None)
        if path is None:
            raise SystemExit(f"--cfg not found: {cfg!r} (tried "
                             f"{', '.join(tried)})")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"--cfg is not a file: {path}")
    if not path.endswith((".yaml", ".yml")):
        raise SystemExit(f"--cfg must be a YAML config, got {path}")
    return "/" + path


def build_predictor(args) -> tuple[object, str]:
    cfg = resolve_cfg(args.cfg, args.model_size)

    # Several upstream EfficientTAM modules create cached tensors on the current
    # CUDA device during construction. Merely calling model.to("cuda:1") later is
    # insufficient and silently changes predictions. Pin the current device first.
    if args.device.startswith("cuda"):
        index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(index)

    if args.weights:
        ckpt = os.path.abspath(args.weights)
        if not os.path.isfile(ckpt):
            raise SystemExit(f"--weights not found: {ckpt}")
        print(f"[model] {args.model_size} | local weights: {ckpt}")
    else:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=HF_REPO_ID,
                               filename=HF_BY_SIZE[args.model_size],
                               cache_dir=CKPT_CACHE_DIR)
        print(f"[model] {args.model_size} | baseline: {ckpt}")

    print(f"[model] config: {cfg}")
    print(f"[model] device: {args.device}")
    feature_side = args.image_size // 16
    overrides = [
        f"++model.image_size={args.image_size}",
        "++model.memory_attention.layer.self_attention.feat_sizes="
        f"[{feature_side},{feature_side}]",
        "++model.memory_attention.layer.cross_attention.feat_sizes="
        f"[{feature_side},{feature_side}]",
    ]
    if args.no_compile or args.device.startswith("cpu"):
        overrides.append("++model.compile_image_encoder=False")
    if args.device.startswith("cpu"):
        # PositionEmbeddingSine otherwise warms a CUDA cache during model
        # construction whenever CUDA is visible, even though inference is CPU.
        # Besides wasting VRAM, this can OOM a concurrent full-GPU trainer.
        overrides.extend([
            "++model.image_encoder.neck.position_encoding.warmup_cache=False",
            "++model.memory_encoder.position_encoding.warmup_cache=False",
        ])
    print(f"[model] input: {args.image_size}x{args.image_size}")
    return build_efficienttam_video_predictor_npz(
        cfg, ckpt, device=args.device, hydra_overrides_extra=overrides), ckpt


@torch.inference_mode()
def infer_3d(img_npz_file: str, predictor, args) -> tuple[str, float]:
    t0 = time.time()
    npz_name = basename(img_npz_file)
    print(f"processing {npz_name}")
    npz_data = np.load(img_npz_file, "r", allow_pickle=True)
    spacing = npz_data["spacing"]
    gts_3D_ori, _ = load_gt(npz_data, args.gt_key, npz_name, args.gts_path)
    img_3D_ori = npz_data["imgs"]                      # (D, H, W)
    assert np.max(img_3D_ori) < 256, "input data should be in range [0, 255]"
    recist = npz_data["recist"]
    anchor_segs = None
    if args.anchor_mask_dir:
        anchor_path = join(args.anchor_mask_dir, npz_name)
        if not os.path.isfile(anchor_path):
            raise FileNotFoundError(f"anchor prediction is missing: {anchor_path}")
        with np.load(anchor_path, allow_pickle=False) as anchor_data:
            anchor_segs = anchor_data["segs"]
        if anchor_segs.shape != img_3D_ori.shape:
            raise ValueError(
                f"anchor shape mismatch for {npz_name}: "
                f"{anchor_segs.shape} != {img_3D_ori.shape}"
            )

    segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
    unique_labs = np.unique(recist)
    unique_labs = unique_labs[unique_labs != 0]
    print(f"unique_labs: {unique_labs}")

    # Predictable worst-case work can be calculated from RECIST geometry before
    # model inference. Only unusually expensive cases use a smaller fallback
    # slab; empty-mask stopping remains content-adaptive within that bound.
    effective_z_slab_scale = args.z_slab_scale
    if args.max_planned_steps and len(unique_labs):
        planned_steps = 0
        for label in unique_labs:
            prompt_z = int(np.where(recist == label)[0][0])
            prompt_diameter = recist_diameter(recist[prompt_z] == label)
            if args.slab_formula == "correct":
                prompt_span = prompt_diameter * (
                    ((spacing[0] + spacing[1]) / 2) / spacing[2]
                )
            else:
                wrong = ((spacing[1] + spacing[2]) / 2) / spacing[0]
                prompt_span = (
                    prompt_diameter * wrong
                    if args.slab_formula == "repo"
                    else prompt_diameter / wrong
                )
            planned_half = int(prompt_span * args.z_slab_scale / 2)
            planned_steps += 2 * (planned_half + 1)
        if planned_steps > args.max_planned_steps:
            effective_z_slab_scale = min(
                args.z_slab_scale, args.fallback_z_slab_scale
            )
            print(
                f"[budget] planned_steps={planned_steps} > "
                f"{args.max_planned_steps}; slab "
                f"{args.z_slab_scale} -> {effective_z_slab_scale}"
            )

    video_height, video_width = img_3D_ori.shape[1], img_3D_ori.shape[2]
    if (video_height, video_width) != (args.image_size, args.image_size):
        img_resized = resize_grayscale_to_rgb_and_resize(
            img_3D_ori, args.image_size
        )
    else:
        img_resized = img_3D_ori[:, None].repeat(3, axis=1)
    # Keep the resident volume float32. The previous NumPy division promoted it
    # to float64, wasting several GB on deep CTs under the 8 GB Docker limit.
    img_resized = torch.from_numpy(img_resized).float().div_(255.0)
    img_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    img_resized -= img_mean
    img_resized /= img_std

    boxes_3D_ori, z_mids = [], []
    shared_state = None
    if args.state_mode == "shared" and len(unique_labs):
        shared_state = predictor.init_state(
            img_resized, video_height, video_width
        )
    for ulab in unique_labs:
        recist_per_lab = (recist == ulab) * ulab
        if not recist_per_lab.any():
            print(f"  no recist for label {ulab}, skipping")
            continue

        z_indices = np.unique(np.where(recist == ulab)[0])
        assert len(z_indices) == 1, \
            f"expected one z index for recist=={ulab}, got {z_indices}"
        z_mid_orig = int(z_indices[0])

        # z-slab: convert the in-plane diameter (px) into a number of slices.
        # NPZ spacing is sitk (x, y, z) -- see --slab-formula for why there are
        # three variants and which one is dimensionally correct.
        diameter = recist_diameter(recist_per_lab[z_mid_orig])
        if args.slab_formula == "correct":
            z_span = diameter * (((spacing[0] + spacing[1]) / 2) / spacing[2])
        else:
            wrong = ((spacing[1] + spacing[2]) / 2) / spacing[0]
            z_span = (diameter * wrong if args.slab_formula == "repo"
                      else diameter / wrong)
        half_z = int(z_span * effective_z_slab_scale / 2)
        z_lo = max(0, z_mid_orig - half_z)
        z_hi = min(img_3D_ori.shape[0] - 1, z_mid_orig + half_z)

        if args.state_mode == "shared":
            inference_state = shared_state
            state_frame = z_mid_orig
            frame_offset = 0
            recist_prompt = recist_per_lab[z_mid_orig]
        else:
            img_crop = img_resized[z_lo:z_hi + 1]
            inference_state = predictor.init_state(
                img_crop, video_height, video_width
            )
            state_frame = z_mid_orig - z_lo
            frame_offset = z_lo
            recist_prompt = recist_per_lab[z_mid_orig]
        z_mids.append(z_mid_orig)

        # Always record the derived box: it is the provenance of the prompt and
        # keeps the `boxes` output well-formed for point modes too (the original
        # crashed in np.stack when no box branch ran).
        box_2d = recist_bbox(recist_prompt, scale=args.box_scale,
                             shift=args.shift)
        boxes_3D_ori.append([box_2d[0], box_2d[1], z_mid_orig,
                             box_2d[2], box_2d[3], z_mid_orig])

        if anchor_segs is not None:
            mask_prompt = (anchor_segs[z_mid_orig] == ulab).astype(np.uint8)
            if not mask_prompt.any():
                raise ValueError(
                    f"empty anchor for {npz_name}, lesion {int(ulab)}, "
                    f"RECIST slice {z_mid_orig}"
                )
            print(f"  label {ulab}: external mask anchor "
                  f"({int(mask_prompt.sum())} pixels)")
        elif args.prompt == "box":
            print(f"  label {ulab}: box {box_2d.tolist()} "
                  f"(diameter {diameter:.1f} x {args.box_scale})")
            _, _, logits = predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=state_frame, obj_id=1,
                box=box_2d)
        elif args.prompt == "box3points":
            pts = prompt_points(recist_prompt, "3points")
            print(f"  label {ulab}: box3points {box_2d.tolist()} + "
                  f"{[tuple(np.round(p, 1)) for p in pts]}")
            _, _, logits = predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=state_frame, obj_id=1,
                points=pts, labels=np.ones(len(pts)), box=box_2d)
        else:
            pts = prompt_points(recist_prompt, args.prompt)
            print(f"  label {ulab}: {args.prompt} -> "
                  f"{[tuple(np.round(p, 1)) for p in pts]}")
            _, _, logits = predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=state_frame, obj_id=1,
                points=pts, labels=np.ones(len(pts)))
        if anchor_segs is None:
            mask_prompt = (logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        lesion_mask = np.zeros_like(segs_3D, dtype=bool)

        # forward, then reset and go reverse from the same anchor mask
        _, _, masks = predictor.add_new_mask(inference_state, frame_idx=state_frame,
                                            obj_id=1, mask=mask_prompt)
        lesion_mask[z_mid_orig, (masks[0] > 0.0).cpu().numpy()[0]] = True
        empty_run = 0
        for fi, _, lg in predictor.propagate_in_video(
                inference_state, start_frame_idx=state_frame,
                max_frame_num_to_track=half_z, reverse=False):
            binary = (lg[0] > 0.0).cpu().numpy()[0]
            lesion_mask[frame_offset + fi, binary] = True
            empty_run = empty_run + 1 if not binary.any() else 0
            if args.empty_stop_patience and empty_run >= args.empty_stop_patience:
                break

        # reset_state deliberately preserves images and the shared feature cache.
        # Reusing the state removes repeated volume setup while keeping the two
        # propagation directions independent at the mask-memory level.
        predictor.reset_state(inference_state)
        predictor.add_new_mask(inference_state, frame_idx=state_frame, obj_id=1,
                               mask=mask_prompt)
        empty_run = 0
        for fi, _, lg in predictor.propagate_in_video(
                inference_state, start_frame_idx=state_frame,
                max_frame_num_to_track=half_z, reverse=True):
            binary = (lg[0] > 0.0).cpu().numpy()[0]
            lesion_mask[frame_offset + fi, binary] = True
            empty_run = empty_run + 1 if not binary.any() else 0
            if args.empty_stop_patience and empty_run >= args.empty_stop_patience:
                break
        predictor.reset_state(inference_state)

        prompt_3d = recist == ulab
        if not args.no_component_filter:
            lesion_mask = keep_prompt_component(lesion_mask, prompt_3d)
        # With filtering disabled, retain the model output exactly.
        # Never destroy an already assigned instance in an overlap. Each RECIST
        # marker is then restored to its own ID so the prompt/instance contract is
        # preserved even when lesions touch.
        segs_3D[lesion_mask & (segs_3D == 0)] = ulab
        segs_3D[prompt_3d & lesion_mask] = ulab

    output = {"segs": segs_3D}
    if args.debug_output:
        if gts_3D_ori is not None:
            output["gts"] = gts_3D_ori
        output["boxes"] = (np.stack(boxes_3D_ori) if boxes_3D_ori
                           else np.zeros((0, 6), dtype=int))
        output["spacing"] = spacing
    np.savez_compressed(join(args.pred_save_dir, npz_name), **output)
    dur = time.time() - t0
    print(f"Finished processing {npz_name} in {dur:.2f} seconds")

    if args.save_nifti:
        nifti_path = join(args.pred_save_dir, "segs_nifti")
        arrays = [(img_3D_ori, "_imgs"), (segs_3D, "_segs")]
        if gts_3D_ori is not None:
            arrays.insert(1, (gts_3D_ori, "_gts"))
        for arr, tag in arrays:
            im = sitk.GetImageFromArray(arr)
            im.SetSpacing([float(s) for s in spacing])
            sitk.WriteImage(im, join(nifti_path,
                                     npz_name.replace(".npz", f"{tag}.nii.gz")))

    if args.save_overlay and z_mids:
        png_dir = join(args.pred_save_dir, "png_overlay")
        idx = random.choice(z_mids)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        left_title = "Image + GT" if gts_3D_ori is not None else "Image + RECIST"
        for a, title in ((ax[0], left_title), (ax[1], "EfficientMedSAM2")):
            a.imshow(img_3D_ori[idx], cmap="gray")
            a.set_title(title)
            a.axis("off")
        for box_idx, label_id in enumerate(unique_labs):
            if not (segs_3D[idx] == label_id).any():
                print(f"  no mask for {npz_name} label {label_id}")
                continue
            color = np.random.rand(3)
            if box_idx < len(boxes_3D_ori):
                x0, y0, z0, x1, y1, z1 = boxes_3D_ori[box_idx]
                if z0 <= idx <= z1:
                    show_box(np.array([x0, y0, x1, y1]), ax[1], edgecolor=color)
                    show_box(np.array([x0, y0, x1, y1]), ax[0], edgecolor=color)
            show_mask(segs_3D[idx] == label_id, ax[1], mask_color=color)
            reference = ((gts_3D_ori[idx] == label_id) if gts_3D_ori is not None
                         else (recist[idx] == label_id))
            show_mask(reference, ax[0], mask_color=color)
        plt.tight_layout()
        plt.savefig(join(png_dir, f"{npz_name.split('.')[0]}_{idx}.png"),
                    dpi=OVERLAY_DPI)
        plt.close()

    return npz_name, dur


def main() -> int:
    global SUBTRACT_THICKNESS
    validate_settings()
    args = resolve_args(build_parser().parse_args())
    if args.box_scale <= 0 or args.z_slab_scale <= 0:
        raise SystemExit("--box-scale and --z-slab-scale must be positive")
    if LEGACY_GEOMETRY:
        print("[geometry] LEGACY (pre-fix raster endpoints + truncating box) "
              "-- control run, not a real result")
    os.makedirs(args.pred_save_dir, exist_ok=True)
    if args.save_overlay:
        os.makedirs(join(args.pred_save_dir, "png_overlay"), exist_ok=True)
    if args.save_nifti:
        os.makedirs(join(args.pred_save_dir, "segs_nifti"), exist_ok=True)

    if args.file_list:
        with open(args.file_list) as stream:
            rels = [line.strip() for line in stream if line.strip()]
        files = [join(args.imgs_path, r if r.endswith(".npz") else r + ".npz")
                 for r in rels]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise SystemExit(f"{len(missing)} files from --file-list are missing; "
                             f"first: {missing[0]}")
    else:
        files = sorted(glob(join(args.imgs_path, "**", "*.npz"), recursive=True))
    print(f"number of files to process: {len(files)}")
    if not files:
        raise SystemExit(f"no NPZ files in {args.imgs_path}")

    # Resolve the axial over-measure. 'auto' needs the file list, so it happens
    # here rather than next to the other flags; the value is logged below so a run
    # is reproducible from its own log.
    auto_note = ""
    if LEGACY_GEOMETRY:
        SUBTRACT_THICKNESS = 0.0
    elif args.subtract_thickness == AUTO_THICKNESS:
        SUBTRACT_THICKNESS, w_med, n_lab = estimate_axial_over_measure(files)
        auto_note = (f" (auto: median width {w_med:.2f} px over {n_lab} labels "
                     f"from {min(len(files), AUTO_THICKNESS_PROBE)} NPZ, "
                     f"minus {AUTO_THICKNESS_OFFSET} centre row)")
        print(f"[prompt] subtract_thickness{auto_note}")
    else:
        try:
            SUBTRACT_THICKNESS = float(args.subtract_thickness)
        except ValueError:
            raise SystemExit("--subtract-thickness must be a number or 'auto'") from None
        if SUBTRACT_THICKNESS < 0:
            raise SystemExit("--subtract-thickness must be non-negative")

    predictor, ckpt = build_predictor(args)
    print(f"[prompt] {args.prompt}"
          + (f" (box_scale={args.box_scale})" if args.prompt in ("box", "box3points", "grid9")
             else "")
          + f" | slab={args.slab_formula} x{args.z_slab_scale}"
          + f" | subtract_thickness={SUBTRACT_THICKNESS}{auto_note}"
          + f" | shift={args.shift}")
    if args.z_slab_scale != 1.0:
        print(f"[prompt] z-slab is {args.z_slab_scale}x the original width")

    names, durations = [], []
    for f in files:
        name, dur = infer_3d(f, predictor, args)
        names.append(name)
        durations.append(dur)

    pd.DataFrame(OrderedDict(image=names, duration=durations)).to_csv(
        join(args.pred_save_dir, "inference_times.csv"), index=False)
    print(f"wrote {join(args.pred_save_dir, 'inference_times.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
