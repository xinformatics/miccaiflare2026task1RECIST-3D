#!/usr/bin/env python3
"""Render a clean 3D lesion, axial CT plane, and RECIST marker pipeline asset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageChops
from skimage.measure import marching_cubes

from solution.visualize_recist_to_3d import bounds


SURFACE_COLOR = "#f43f5e"
RECIST_COLOR = "#06b6d4"


def trim_to_canvas(path: Path, transparent: bool, padding: int = 36,
                   aspect: float = 1.0) -> None:
    """Remove renderer whitespace and center content on a chosen width/height."""
    image = Image.open(path).convert("RGBA")
    if transparent:
        box = image.getchannel("A").getbbox()
        background = (255, 255, 255, 0)
    else:
        rgb = image.convert("RGB")
        white = Image.new("RGB", image.size, (255, 255, 255))
        box = ImageChops.difference(rgb, white).getbbox()
        background = (255, 255, 255, 255)
    if box is None:
        return
    left = max(0, box[0] - padding); top = max(0, box[1] - padding)
    right = min(image.width, box[2] + padding); bottom = min(image.height, box[3] + padding)
    cropped = image.crop((left, top, right, bottom))
    if cropped.width / cropped.height > aspect:
        canvas_width = cropped.width
        canvas_height = int(round(canvas_width / aspect))
    else:
        canvas_height = cropped.height
        canvas_width = int(round(canvas_height * aspect))
    canvas = Image.new("RGBA", (canvas_width, canvas_height), background)
    canvas.paste(cropped, ((canvas_width - cropped.width) // 2,
                           (canvas_height - cropped.height) // 2), cropped)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--lesion", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transparent-output", type=Path)
    parser.add_argument("--azimuth", type=float, default=-58)
    parser.add_argument("--elevation", type=float, default=27)
    parser.add_argument("--vertical", action="store_true",
                        help="export on a portrait 2:3 canvas for vertical pipelines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.case, allow_pickle=False) as case:
        image = case["imgs"]
        recist = case["recist"] == args.lesion
        spacing_xyz = np.asarray(case["spacing"], dtype=float)
    with np.load(args.prediction, allow_pickle=False) as prediction:
        mask = prediction["segs"] == args.lesion
    if not mask.any() or not recist.any():
        raise ValueError("requested lesion is absent from prediction or RECIST")

    crop = bounds(mask | recist, padding=(4, 15, 15))
    image = image[crop]
    mask = mask[crop]
    recist = recist[crop]
    spacing_zyx = spacing_xyz[::-1]

    vertices, faces, _, _ = marching_cubes(
        mask.astype(np.uint8), level=0.5, spacing=spacing_zyx
    )
    vertices_xyz = vertices[:, [2, 1, 0]]

    prompt = np.argwhere(recist)
    anchor_z_index = int(np.median(prompt[:, 0]))
    anchor_z_mm = anchor_z_index * spacing_zyx[0]

    # Crop the displayed CT plane a little more tightly around the lesion.
    yx = np.argwhere(np.any(mask | recist, axis=0))
    y_lo, x_lo = np.maximum(yx.min(0) - 7, 0)
    y_hi, x_hi = np.minimum(yx.max(0) + 8, mask.shape[1:])
    axial = image[anchor_z_index, y_lo:y_hi, x_lo:x_hi].astype(float) / 255.0
    yy = np.arange(y_lo, y_hi) * spacing_zyx[1]
    xx = np.arange(x_lo, x_hi) * spacing_zyx[2]
    plane_x, plane_y = np.meshgrid(xx, yy)
    plane_z = np.full_like(plane_x, anchor_z_mm)
    plane_colors = cm.gray(np.clip(axial, 0, 1))
    plane_colors[..., 3] = 0.72

    figure_size = (6, 9) if args.vertical else (8, 7)
    fig = plt.figure(figsize=figure_size, dpi=300, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # The CT plane provides anatomical context without rendering the full volume.
    ax.plot_surface(
        plane_x, plane_y, plane_z,
        rstride=1, cstride=1, facecolors=plane_colors,
        shade=False, linewidth=0, antialiased=False,
    )

    lesion = Poly3DCollection(vertices_xyz[faces], alpha=0.30, linewidth=0.18)
    lesion.set_facecolor(SURFACE_COLOR)
    lesion.set_edgecolor("#be123c")
    ax.add_collection3d(lesion)

    # Order the rasterized RECIST pixels along their dominant in-plane axis.
    prompt_mm = prompt.astype(float) * spacing_zyx
    dominant = 2 if np.ptp(prompt_mm[:, 2]) >= np.ptp(prompt_mm[:, 1]) else 1
    prompt_mm = prompt_mm[np.argsort(prompt_mm[:, dominant])]
    # Lift the line slightly above the CT plane so it remains crisp when composited.
    line_z = prompt_mm[:, 0] + max(0.6, spacing_zyx[0] * 0.18)
    ax.plot(prompt_mm[:, 2], prompt_mm[:, 1], line_z,
            color="white", linewidth=8.0, solid_capstyle="round", zorder=20)
    ax.plot(prompt_mm[:, 2], prompt_mm[:, 1], line_z,
            color=RECIST_COLOR, linewidth=5.2, solid_capstyle="round", zorder=21)
    endpoints = prompt_mm[[0, -1]]
    endpoint_z = line_z[[0, -1]]
    ax.scatter(endpoints[:, 2], endpoints[:, 1], endpoint_z,
               s=105, facecolors="white", edgecolors=RECIST_COLOR,
               linewidths=2.3, depthshade=False, zorder=22)

    physical = np.asarray(mask.shape) * spacing_zyx
    ax.set_xlim(0, physical[2])
    ax.set_ylim(0, physical[1])
    ax.set_zlim(0, physical[0])
    ax.set_box_aspect((physical[2], physical[1], physical[0]))
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    ax.set_axis_off()
    ax.margins(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    if args.transparent_output:
        args.transparent_output.parent.mkdir(parents=True, exist_ok=True)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        fig.savefig(args.transparent_output, dpi=300, transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    output_aspect = 2 / 3 if args.vertical else 1.0
    trim_to_canvas(args.output, transparent=False, aspect=output_aspect)
    if args.transparent_output:
        trim_to_canvas(args.transparent_output, transparent=True,
                       aspect=output_aspect)
    print(f"wrote {args.output}")
    if args.transparent_output:
        print(f"wrote {args.transparent_output}")


if __name__ == "__main__":
    main()
