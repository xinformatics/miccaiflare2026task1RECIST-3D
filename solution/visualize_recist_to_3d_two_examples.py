#!/usr/bin/env python3
"""Render two FLARE Task 2 RECIST-to-3D qualitative examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from solution.evaluate import dice, nsd
from solution.visualize_recist_to_3d import (
    GT_COLOR, PRED_COLOR, RECIST_COLOR, add_overlay, bounds, image_panel, surface,
)


@dataclass
class Example:
    case_path: Path
    prediction_path: Path
    lesion: int
    image: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    recist: np.ndarray
    spacing_xyz: np.ndarray
    anchor_z: int
    center_y: int
    score_dsc: float
    score_nsd: float
    z_slices: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--lesion", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vertical", action="store_true",
                        help="use a portrait two-case manuscript layout")
    args = parser.parse_args()
    if not (len(args.case) == len(args.prediction) == len(args.lesion) == 2):
        parser.error("provide exactly two --case, --prediction, and --lesion values")
    return args


def load_example(case_path: Path, prediction_path: Path, lesion: int) -> Example:
    with np.load(case_path, allow_pickle=False) as case:
        image = case["imgs"]
        gt = case["gts"] == lesion
        recist = case["recist"] == lesion
        spacing_xyz = np.asarray(case["spacing"], dtype=float)
    with np.load(prediction_path, allow_pickle=False) as prediction:
        pred = prediction["segs"] == lesion

    crop = bounds(gt | pred | recist, padding=(4, 18, 18))
    image, gt, pred, recist = image[crop], gt[crop], pred[crop], recist[crop]
    prompt = np.argwhere(recist)
    anchor_z = int(np.median(prompt[:, 0]))
    center_y = int(np.round(np.argwhere(gt).mean(0)[1]))
    support = np.where(np.any(gt | pred, axis=(1, 2)))[0]
    low, high = int(support.min()), int(support.max())
    return Example(
        case_path=case_path,
        prediction_path=prediction_path,
        lesion=lesion,
        image=image,
        gt=gt,
        pred=pred,
        recist=recist,
        spacing_xyz=spacing_xyz,
        anchor_z=anchor_z,
        center_y=center_y,
        score_dsc=dice(gt, pred),
        score_nsd=nsd(gt, pred, spacing_xyz),
        z_slices=(low, high),
    )


def prompt_panel(ax, example: Example) -> None:
    image_panel(ax, example.image[example.anchor_z], "")
    rgba = np.zeros((*example.recist[example.anchor_z].shape, 4), dtype=float)
    rgba[example.recist[example.anchor_z]] = (0.133, 0.827, 0.933, 1.0)
    ax.imshow(rgba, interpolation="nearest")
    points = np.argwhere(example.recist[example.anchor_z])
    ax.scatter([points[0, 1], points[-1, 1]], [points[0, 0], points[-1, 0]],
               s=30, facecolors="white", edgecolors=RECIST_COLOR,
               linewidths=1.6, zorder=5)


def render_row(fig, grid, row: int, example: Example) -> None:
    ax = fig.add_subplot(grid[row, 0])
    prompt_panel(ax, example)

    ax = fig.add_subplot(grid[row, 1])
    image_panel(ax, example.image[example.anchor_z], "")
    add_overlay(ax, example.gt[example.anchor_z], example.pred[example.anchor_z],
                example.recist[example.anchor_z])

    for column, z_index in zip((2, 3), example.z_slices):
        ax = fig.add_subplot(grid[row, column])
        image_panel(ax, example.image[z_index], "")
        add_overlay(ax, example.gt[z_index], example.pred[z_index])

    ax = fig.add_subplot(grid[row, 4])
    aspect = example.spacing_xyz[2] / example.spacing_xyz[0]
    image_panel(ax, example.image[:, example.center_y, :], "", aspect=aspect)
    add_overlay(ax, example.gt[:, example.center_y, :],
                example.pred[:, example.center_y, :],
                example.recist[:, example.center_y, :])

    ax3d = fig.add_subplot(grid[row, 5], projection="3d")
    spacing_zyx = example.spacing_xyz[::-1]
    surface(ax3d, example.gt, spacing_zyx, GT_COLOR, 0.25)
    surface(ax3d, example.pred, spacing_zyx, PRED_COLOR, 0.36)
    line = np.argwhere(example.recist).astype(float) * spacing_zyx
    if len(line):
        axis = 2 if np.ptp(line[:, 2]) >= np.ptp(line[:, 1]) else 1
        line = line[np.argsort(line[:, axis])]
        ax3d.plot(line[:, 2], line[:, 1], line[:, 0], color=RECIST_COLOR,
                  linewidth=3.5)
    shape_mm = np.asarray(example.gt.shape) * spacing_zyx
    ax3d.set_xlim(0, shape_mm[2]); ax3d.set_ylim(0, shape_mm[1]); ax3d.set_zlim(0, shape_mm[0])
    ax3d.set_box_aspect((shape_mm[2], shape_mm[1], shape_mm[0]))
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_axis_off(); ax3d.set_facecolor("white")


def render_vertical_block(fig, grid, start_row: int, example: Example,
                          panel_offset: int) -> None:
    """Render one case as a two-row by three-column portrait block."""
    letters = [chr(ord("a") + panel_offset + index) for index in range(6)]

    ax = fig.add_subplot(grid[start_row, 0])
    prompt_panel(ax, example)
    ax.set_title(f"({letters[0]}) RECIST prompt", loc="left", fontsize=16,
                 fontweight="bold", pad=7)

    ax = fig.add_subplot(grid[start_row, 1])
    image_panel(ax, example.image[example.anchor_z], "")
    add_overlay(ax, example.gt[example.anchor_z], example.pred[example.anchor_z],
                example.recist[example.anchor_z])
    ax.set_title(f"({letters[1]}) Anchor slice", loc="left", fontsize=16,
                 fontweight="bold", pad=7)

    ax = fig.add_subplot(grid[start_row, 2])
    aspect = example.spacing_xyz[2] / example.spacing_xyz[0]
    image_panel(ax, example.image[:, example.center_y, :], "", aspect=aspect)
    add_overlay(ax, example.gt[:, example.center_y, :],
                example.pred[:, example.center_y, :],
                example.recist[:, example.center_y, :])
    ax.set_title(f"({letters[2]}) Coronal view", loc="left", fontsize=16,
                 fontweight="bold", pad=7)

    for column, z_index, label in zip((0, 1), example.z_slices,
                                      ("Inferior slice", "Superior slice")):
        ax = fig.add_subplot(grid[start_row + 1, column])
        image_panel(ax, example.image[z_index], "")
        add_overlay(ax, example.gt[z_index], example.pred[z_index])
        ax.set_title(f"({letters[3 + column]}) {label}", loc="left",
                     fontsize=16, fontweight="bold", pad=7)

    ax3d = fig.add_subplot(grid[start_row + 1, 2], projection="3d")
    spacing_zyx = example.spacing_xyz[::-1]
    surface(ax3d, example.gt, spacing_zyx, GT_COLOR, 0.25)
    surface(ax3d, example.pred, spacing_zyx, PRED_COLOR, 0.36)
    line = np.argwhere(example.recist).astype(float) * spacing_zyx
    if len(line):
        axis = 2 if np.ptp(line[:, 2]) >= np.ptp(line[:, 1]) else 1
        line = line[np.argsort(line[:, axis])]
        ax3d.plot(line[:, 2], line[:, 1], line[:, 0], color=RECIST_COLOR,
                  linewidth=3.5)
    shape_mm = np.asarray(example.gt.shape) * spacing_zyx
    ax3d.set_xlim(0, shape_mm[2]); ax3d.set_ylim(0, shape_mm[1]); ax3d.set_zlim(0, shape_mm[0])
    ax3d.set_box_aspect((shape_mm[2], shape_mm[1], shape_mm[0]))
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_title(f"({letters[5]}) 3D surface", loc="left", fontsize=16,
                   fontweight="bold", pad=7)
    ax3d.set_axis_off(); ax3d.set_facecolor("white")


def render_vertical(examples: list[Example], output: Path) -> None:
    fig = plt.figure(figsize=(10, 16), dpi=200, facecolor="white")
    grid = fig.add_gridspec(
        5, 3, left=0.055, right=0.975, top=0.86, bottom=0.05,
        width_ratios=(1, 1, 1), height_ratios=(1, 1, 0.28, 1, 1),
        wspace=0.12, hspace=0.24,
    )
    render_vertical_block(fig, grid, 0, examples[0], 0)
    render_vertical_block(fig, grid, 3, examples[1], 6)

    legend = [
        Line2D([0], [0], color=RECIST_COLOR, lw=4, label="RECIST prompt"),
        Line2D([0], [0], color=GT_COLOR, lw=3, label="Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=3, label="Prediction"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=16, labelcolor="#111827",
               columnspacing=2.4, handletextpad=0.7)

    for index, (example, y) in enumerate(zip(examples, (0.895, 0.443)), 1):
        slices = int(np.count_nonzero(np.any(example.gt, axis=(1, 2))))
        fig.text(0.055, y, f"Example {index}", fontsize=19, fontweight="bold",
                 color="#111827", va="center")
        fig.text(0.975, y,
                 f"DSC {example.score_dsc:.3f}    NSD {example.score_nsd:.3f}    {slices} slices",
                 fontsize=16, color="#111827", ha="right", va="center")
    fig.add_artist(plt.Line2D([0.055, 0.975], [0.485, 0.485],
                              transform=fig.transFigure, color="#d1d5db",
                              linewidth=0.8))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    examples = [
        load_example(case, prediction, lesion)
        for case, prediction, lesion in zip(args.case, args.prediction, args.lesion)
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "#111827",
        "axes.titlecolor": "#111827",
    })
    if args.vertical:
        render_vertical(examples, args.output)
        print(f"wrote {args.output}")
        for index, example in enumerate(examples, 1):
            print(f"example {index}: DSC={example.score_dsc:.6f} NSD={example.score_nsd:.6f}")
        return
    fig = plt.figure(figsize=(18, 9), dpi=180, facecolor="white")
    grid = fig.add_gridspec(2, 6, left=0.095, right=0.985, top=0.86,
                            bottom=0.12, wspace=0.11, hspace=0.34)

    render_row(fig, grid, 0, examples[0])
    render_row(fig, grid, 1, examples[1])

    column_titles = (
        "(a) RECIST prompt", "(b) Anchor slice", "(c) Inferior slice",
        "(d) Superior slice", "(e) Coronal view", "(f) 3D surface",
    )
    column_x = (0.164, 0.318, 0.472, 0.626, 0.780, 0.934)
    for x, title in zip(column_x, column_titles):
        fig.text(x, 0.91, title, ha="center", va="bottom", fontsize=13,
                 fontweight="bold", color="#111827")

    legend = [
        Line2D([0], [0], color=RECIST_COLOR, lw=4, label="RECIST prompt"),
        Line2D([0], [0], color=GT_COLOR, lw=3, label="Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=3, label="Prediction"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.54, 0.985), fontsize=12, labelcolor="#111827",
               columnspacing=2.8, handletextpad=0.8)

    row_y = (0.665, 0.275)
    for index, (example, y) in enumerate(zip(examples, row_y), start=1):
        slices = int(np.count_nonzero(np.any(example.gt, axis=(1, 2))))
        fig.text(0.018, y,
                 f"Example {index}\n\nDSC  {example.score_dsc:.3f}\nNSD  {example.score_nsd:.3f}\n\n{slices} slices",
                 fontsize=12, color="#111827", va="center", ha="left",
                 linespacing=1.25)

    fig.add_artist(plt.Line2D([0.095, 0.985], [0.49, 0.49],
                              transform=fig.transFigure, color="#d1d5db",
                              linewidth=0.8))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    for index, example in enumerate(examples, 1):
        print(f"example {index}: DSC={example.score_dsc:.6f} NSD={example.score_nsd:.6f}")


if __name__ == "__main__":
    main()
