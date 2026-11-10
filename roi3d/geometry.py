from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform


LINE_CAP_PX = 1.75
V2_SHAPE_ZYX = (112, 176, 176)
V2_CACHE_SHAPE_ZYX = (128, 192, 192)


@dataclass(frozen=True)
class RECISTGeometry:
    lesion_id: int
    center_zyx: np.ndarray
    axis_xy: np.ndarray
    diameter_mm: float
    diameter_true_mm: float
    roi_spacing_mm: float


def _principal_line(recist: np.ndarray, lesion_id: int,
                    spacing_xyz: np.ndarray) -> RECISTGeometry:
    zyx = np.argwhere(recist == lesion_id)
    if not len(zyx):
        raise ValueError(f"RECIST id {lesion_id} is absent")
    z_values = np.unique(zyx[:, 0])
    if len(z_values) != 1:
        raise ValueError(f"RECIST id {lesion_id} occupies z={z_values.tolist()}")

    xy = zyx[:, [2, 1]].astype(np.float64)
    center_xy = xy.mean(0)
    centered = xy - center_xy
    if len(xy) == 1:
        axis_xy = np.array([1.0, 0.0])
        extent_px = 1.0
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_xy = vh[0]
        projection = centered @ axis_xy
        extent_px = float(projection.max() - projection.min())

    # Convert distance along the raster line to millimetres. In-plane spacing is
    # usually isotropic, but using the projected spacing keeps oblique lines exact.
    physical_axis = axis_xy * np.asarray(spacing_xyz[:2], dtype=np.float64)
    projected_spacing = float(np.linalg.norm(physical_axis))
    physical_axis /= max(projected_spacing, 1e-12)
    diameter_mm = max(extent_px * projected_spacing, projected_spacing)
    diameter_true_mm = max(
        (extent_px - LINE_CAP_PX) * projected_spacing, projected_spacing
    )
    spacing = max(
        float(np.clip(2.9 * diameter_true_mm / 192.0, 0.45, 1.10)),
        1.6 * diameter_true_mm / 192.0,
    )
    center_zyx = np.array(
        [float(z_values[0]), center_xy[1], center_xy[0]], dtype=np.float64
    )
    return RECISTGeometry(
        lesion_id=int(lesion_id), center_zyx=center_zyx, axis_xy=physical_axis,
        diameter_mm=diameter_mm, diameter_true_mm=diameter_true_mm,
        roi_spacing_mm=spacing,
    )


def recist_geometries(recist: np.ndarray,
                      spacing_xyz: np.ndarray) -> list[RECISTGeometry]:
    ids = np.unique(recist)
    return [_principal_line(recist, int(i), spacing_xyz)
            for i in ids if i != 0]


def geometry_from_mask(mask: np.ndarray, lesion_id: int,
                       spacing_xyz: np.ndarray) -> RECISTGeometry:
    """Fallback for legacy training cases whose RECIST raster is absent."""
    binary = mask == lesion_id
    areas = np.count_nonzero(binary, axis=(1, 2))
    if not np.any(areas):
        raise ValueError(f"GT id {lesion_id} is absent")
    z = int(np.argmax(areas))
    yx = np.argwhere(binary[z])
    xy = yx[:, [1, 0]].astype(np.float64)
    center = xy.mean(0)
    centered = xy - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pixel_axis = vh[0]
    projection = centered @ pixel_axis
    extent_px = float(projection.max() - projection.min())
    physical_axis = pixel_axis * np.asarray(spacing_xyz[:2], dtype=np.float64)
    projected_spacing = float(np.linalg.norm(physical_axis))
    physical_axis /= max(projected_spacing, 1e-12)
    diameter = max(extent_px * projected_spacing, projected_spacing)
    roi_spacing = max(
        float(np.clip(2.9 * diameter / 192.0, 0.45, 1.10)),
        1.6 * diameter / 192.0,
    )
    return RECISTGeometry(
        int(lesion_id), np.array([z, center[1], center[0]]), physical_axis,
        diameter, diameter, roi_spacing,
    )


def native_affine(geometry: RECISTGeometry, spacing_xyz: np.ndarray,
                  output_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return scipy affine mapping output ROI indices to native z,y,x indices."""
    spacing_zyx = np.asarray(spacing_xyz, dtype=np.float64)[::-1]
    scale = geometry.roi_spacing_mm / spacing_zyx
    matrix = np.diag(scale)
    output_center = (np.asarray(output_shape, dtype=np.float64) - 1.0) / 2.0
    offset = geometry.center_zyx - matrix @ output_center
    return matrix, offset


def sample_native(array: np.ndarray, geometry: RECISTGeometry,
                  spacing_xyz: np.ndarray,
                  output_shape: tuple[int, int, int] = V2_SHAPE_ZYX,
                  order: int = 1) -> np.ndarray:
    matrix, offset = native_affine(geometry, spacing_xyz, output_shape)
    # scipy.ndimage does not implement geometric transforms for float16 (a few
    # legacy NPZs use it); avoid changing compact uint8 cases.
    if array.dtype == np.float16:
        array = array.astype(np.float32)
    return affine_transform(
        array, matrix=matrix, offset=offset, output_shape=output_shape,
        order=order, mode="constant", cval=0, prefilter=False,
    )


def scatter_roi(probability: np.ndarray, geometry: RECISTGeometry,
                spacing_xyz: np.ndarray, native_shape: tuple[int, int, int],
                order: int = 1) -> np.ndarray:
    """Resample an ROI probability map into the native grid."""
    matrix, offset = native_affine(geometry, spacing_xyz, probability.shape)
    inverse = np.linalg.inv(matrix)
    native_to_roi_offset = -inverse @ offset
    return affine_transform(
        probability, matrix=inverse, offset=native_to_roi_offset,
        output_shape=native_shape, order=order, mode="constant", cval=0,
        prefilter=False,
    )


def rasterized_line(shape_zyx: tuple[int, int, int],
                    geometry: RECISTGeometry) -> np.ndarray:
    """Create a centered, de-biased RECIST line in normalized ROI coordinates."""
    z, y, x = shape_zyx
    line = np.zeros(shape_zyx, dtype=np.uint8)
    length = geometry.diameter_true_mm / geometry.roi_spacing_mm
    center = np.array([(x - 1) / 2, (y - 1) / 2], dtype=np.float64)
    delta = geometry.axis_xy * length / 2
    samples = max(2, int(np.ceil(length * 2)))
    points = center[None] + np.linspace(-1, 1, samples)[:, None] * delta[None]
    xx = np.clip(np.rint(points[:, 0]).astype(int), 0, x - 1)
    yy = np.clip(np.rint(points[:, 1]).astype(int), 0, y - 1)
    line[(z - 1) // 2, yy, xx] = 1
    return line
