#!/usr/bin/env python3
"""Create a publication-quality RECIST-to-3D qualitative segmentation figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import find_contours, marching_cubes

from solution.evaluate import dice, nsd


GT_COLOR = "#39d98a"
PRED_COLOR = "#ff4d6d"
RECIST_COLOR = "#22d3ee"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--lesion", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bounds(mask: np.ndarray, padding=(4, 24, 24)) -> tuple[slice, ...]:
    coords = np.argwhere(mask)
    lo = coords.min(0)
    hi = coords.max(0) + 1
    result = []
    for axis, (start, stop, pad) in enumerate(zip(lo, hi, padding)):
        result.append(slice(max(0, int(start) - pad),
                            min(mask.shape[axis], int(stop) + pad)))
    return tuple(result)


def contour(ax, mask: np.ndarray, color: str, linewidth: float = 2.0) -> None:
    for curve in find_contours(mask.astype(float), 0.5):
        ax.plot(curve[:, 1], curve[:, 0], color=color, linewidth=linewidth)


def image_panel(ax, image: np.ndarray, title: str, *, aspect="equal") -> None:
    ax.imshow(image, cmap="gray", vmin=15, vmax=235, interpolation="nearest",
              aspect=aspect)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_overlay(ax, gt: np.ndarray, pred: np.ndarray,
                recist: np.ndarray | None = None) -> None:
    contour(ax, gt, GT_COLOR, 2.2)
    contour(ax, pred, PRED_COLOR, 2.0)
    if recist is not None and recist.any():
        overlay = np.zeros((*recist.shape, 4), dtype=float)
        overlay[recist] = (0.133, 0.827, 0.933, 0.95)
        ax.imshow(overlay, interpolation="nearest")


def surface(ax, mask: np.ndarray, spacing_zyx: np.ndarray, color: str,
            alpha: float) -> None:
    if min(mask.shape) < 2 or not mask.any():
        return
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5,
                                            spacing=spacing_zyx)
    xyz = vertices[:, [2, 1, 0]]
    mesh = Poly3DCollection(xyz[faces], alpha=alpha, linewidth=0.12)
    mesh.set_facecolor(color)
    mesh.set_edgecolor(color)
    ax.add_collection3d(mesh)


def main() -> None:
    args = parse_args()
    with np.load(args.case, allow_pickle=False) as case:
        image = case["imgs"]
        recist = case["recist"] == args.lesion
        gt = case["gts"] == args.lesion
        spacing_xyz = np.asarray(case["spacing"], dtype=float)
    with np.load(args.prediction, allow_pickle=False) as prediction:
        pred = prediction["segs"] == args.lesion

    if not gt.any() or not recist.any():
        raise ValueError(f"lesion {args.lesion} is absent from GT or RECIST")
    if pred.shape != gt.shape:
        raise ValueError("prediction and case shapes differ")

    score_dsc = dice(gt, pred)
    score_nsd = nsd(gt, pred, spacing_xyz)
    crop = bounds(gt | pred | recist)
    z0, y0, x0 = (part.start for part in crop)
    image_c = image[crop]
    gt_c, pred_c, recist_c = gt[crop], pred[crop], recist[crop]
    spacing_zyx = spacing_xyz[::-1]

    recist_points = np.argwhere(recist_c)
    anchor_z = int(np.median(recist_points[:, 0]))
    center = np.round(np.argwhere(gt_c).mean(0)).astype(int)
    center_y, center_x = int(center[1]), int(center[2])

    z_support = np.where(np.any(gt_c | pred_c, axis=(1, 2)))[0]
    montage_z = np.unique(np.round(np.linspace(z_support.min(), z_support.max(), 3)).astype(int))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#07111f",
        "axes.facecolor": "#07111f",
        "text.color": "#eaf2ff",
        "axes.titlecolor": "#eaf2ff",
    })
    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="#07111f")
    grid = fig.add_gridspec(2, 4, left=0.035, right=0.98, top=0.86,
                            bottom=0.08, wspace=0.12, hspace=0.25)

    # A: the information supplied to Task 2.
    ax = fig.add_subplot(grid[0, 0])
    image_panel(ax, image_c[anchor_z], "A  Input CT + 2D RECIST prompt")
    prompt_rgba = np.zeros((*recist_c[anchor_z].shape, 4), dtype=float)
    prompt_rgba[recist_c[anchor_z]] = (0.133, 0.827, 0.933, 1.0)
    ax.imshow(prompt_rgba, interpolation="nearest")
    points = np.argwhere(recist_c[anchor_z])
    if len(points):
        ax.scatter([points[0, 1], points[-1, 1]], [points[0, 0], points[-1, 0]],
                   s=38, facecolors="#07111f", edgecolors=RECIST_COLOR,
                   linewidths=1.8, zorder=5)

    # B: comparison on the prompted axial slice.
    ax = fig.add_subplot(grid[0, 1])
    image_panel(ax, image_c[anchor_z], "B  Anchor-slice segmentation")
    add_overlay(ax, gt_c[anchor_z], pred_c[anchor_z], recist_c[anchor_z])

    # C-D: orthogonal evidence that the output is volumetric.
    ax = fig.add_subplot(grid[0, 2])
    aspect_cor = spacing_zyx[0] / spacing_zyx[2]
    image_panel(ax, image_c[:, center_y, :], "C  Coronal view", aspect=aspect_cor)
    add_overlay(ax, gt_c[:, center_y, :], pred_c[:, center_y, :],
                recist_c[:, center_y, :])

    ax = fig.add_subplot(grid[0, 3])
    aspect_sag = spacing_zyx[0] / spacing_zyx[1]
    image_panel(ax, image_c[:, :, center_x], "D  Sagittal view", aspect=aspect_sag)
    add_overlay(ax, gt_c[:, :, center_x], pred_c[:, :, center_x],
                recist_c[:, :, center_x])

    # E-G: axial propagation away from the single prompted slice.
    for column, z_index in enumerate(montage_z[:3]):
        ax = fig.add_subplot(grid[1, column])
        offset_mm = (z_index - anchor_z) * spacing_zyx[0]
        image_panel(ax, image_c[z_index],
                    f"{chr(69 + column)}  3D propagation  z {offset_mm:+.0f} mm")
        add_overlay(ax, gt_c[z_index], pred_c[z_index])

    # H: surface rendering in physical coordinates.
    ax3d = fig.add_subplot(grid[1, 3], projection="3d")
    surface(ax3d, gt_c, spacing_zyx, GT_COLOR, 0.26)
    surface(ax3d, pred_c, spacing_zyx, PRED_COLOR, 0.34)
    local_line = recist_points.astype(float) * spacing_zyx
    if len(local_line):
        # Sort the raster points along the dominant in-plane direction.
        axis = 2 if np.ptp(local_line[:, 2]) >= np.ptp(local_line[:, 1]) else 1
        local_line = local_line[np.argsort(local_line[:, axis])]
        ax3d.plot(local_line[:, 2], local_line[:, 1], local_line[:, 0],
                  color=RECIST_COLOR, linewidth=4.0)
    shape_mm = np.asarray(gt_c.shape) * spacing_zyx
    ax3d.set_xlim(0, shape_mm[2]); ax3d.set_ylim(0, shape_mm[1]); ax3d.set_zlim(0, shape_mm[0])
    ax3d.set_box_aspect((shape_mm[2], shape_mm[1], shape_mm[0]))
    ax3d.view_init(elev=23, azim=-58)
    ax3d.set_title("H  3D lesion surface", loc="left", fontsize=13,
                   fontweight="bold", pad=9)
    ax3d.set_axis_off()
    ax3d.set_facecolor("#07111f")

    fig.text(0.04, 0.955, "FLARE Task 2 · RECIST-to-3D lesion segmentation",
             fontsize=24, fontweight="bold", color="#ffffff", va="top")
    fig.text(0.04, 0.905,
             "A single axial RECIST diameter conditions a volumetric prediction across unprompted slices",
             fontsize=14, color="#a8bad2", va="top")
    fig.text(0.975, 0.945, f"DSC  {score_dsc:.3f}\nNSD  {score_nsd:.3f}",
             fontsize=16, fontweight="bold", color="#ffffff", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#14263f",
                       edgecolor="#29476d", linewidth=1.2))

    legend = [
        Line2D([0], [0], color=RECIST_COLOR, lw=4, label="RECIST prompt"),
        Line2D([0], [0], color=GT_COLOR, lw=3, label="Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=3, label="Prediction"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=12, labelcolor="#dbe8f8")
    fig.text(0.98, 0.018,
             f"{args.case.stem} · lesion {args.lesion} · crop origin (z,y,x)=({z0},{y0},{x0})",
             ha="right", va="bottom", fontsize=9, color="#7388a4")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output} | DSC={score_dsc:.6f} NSD={score_nsd:.6f}")


if __name__ == "__main__":
    main()
