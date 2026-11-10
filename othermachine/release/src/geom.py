"""Shared geometry: RECIST endpoints, ROI crop/resample and its exact inverse."""
import numpy as np

def spacing_zyx(spacing):
    """npz 'spacing' is sitk (x,y,z); arrays are (Z,Y,X)."""
    s = np.asarray(spacing, dtype=np.float64)
    return np.array([s[2], s[1], s[0]])

def recist_endpoints(line_yx):
    """Farthest pair of pixels of a (thick) 2D line mask. -> (p1, p2) as (y,x) float."""
    ys, xs = np.nonzero(line_yx)
    pts = np.stack([ys, xs], 1).astype(np.float64)
    if len(pts) < 2:
        return pts[0], pts[0]
    if len(pts) > 400:
        pts = pts[np.linspace(0, len(pts) - 1, 400, dtype=int)]
    d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return pts[i], pts[j]

# cv2.line(thickness=2) end-caps extend the drawn stroke past the true endpoints; the
# principal-axis extent minus this many pixels recovers the GT key-slice longest diameter
# to ~0.1 mm (fitted on colon/kidney/liver).
LINE_CAP_PX = 1.75

def principal_extent_mm(line_yx, sy, sx):
    """Extent of the drawn stroke along its own principal axis, in mm."""
    ys, xs = np.nonzero(line_yx)
    p = np.stack([ys * sy, xs * sx], 1).astype(np.float64)
    if len(p) < 2:
        return 0.0
    q = p - p.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    t = q @ vt[0]
    return float(t.max() - t.min())

def lesion_prompt(recist, lid, spacing):
    """-> dict(z0, p1, p2, center_zyx, diam_mm, diam_true_mm) for one lesion id.

    `diam_mm` is the raw farthest-pair length (what the ROI scale is keyed on);
    `diam_true_mm` is the de-biased estimate of the lesion's actual RECIST diameter.
    """
    szyx = spacing_zyx(spacing)
    zs = np.unique(np.nonzero(recist == lid)[0])
    z0 = int(zs[len(zs) // 2])
    p1, p2 = recist_endpoints(recist[z0] == lid)
    dy = (p1[0] - p2[0]) * szyx[1]
    dx = (p1[1] - p2[1]) * szyx[2]
    diam_mm = float(np.hypot(dy, dx))
    c = np.array([float(z0), (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])
    ext = principal_extent_mm(recist[z0] == lid, szyx[1], szyx[2])
    diam_true = float(max(ext - LINE_CAP_PX * float(szyx[1]), 1.0))
    return dict(z0=z0, p1=p1, p2=p2, center_zyx=c, diam_mm=diam_mm,
                diam_true_mm=diam_true, spacing_zyx=szyx)

# ---- ROI grid: isotropic `s` mm, size P, centered at center_zyx ----------------

def target_spacing(diam_mm, K=32.0, smin=0.6, smax=2.2):
    return float(np.clip(diam_mm / K, smin, smax))

def as_spacing3(s):
    """Accept a scalar (isotropic) or a per-axis (z,y,x) ROI spacing."""
    a = np.atleast_1d(np.asarray(s, dtype=np.float64))
    return np.repeat(a, 3) if a.size == 1 else a

def _axis_coords(center_a, P_a, s, sp_a):
    """Native (float) indices along one axis for the ROI grid samples."""
    k = np.arange(P_a, dtype=np.float64) - (P_a - 1) / 2.0
    return center_a + k * s / sp_a

def sample_roi(vol, center_zyx, P, s, sp_zyx, order=1, cval=0.0):
    """Resample `vol` (Z,Y,X) onto the ROI grid. Separable linear interp, fast."""
    import scipy.ndimage as ndi
    s3 = as_spacing3(s)
    coords = [_axis_coords(center_zyx[a], P[a], s3[a], sp_zyx[a]) for a in range(3)]
    # tight native bbox to avoid touching the whole volume
    lo = [max(0, int(np.floor(c.min())) - 1) for c in coords]
    hi = [min(vol.shape[a], int(np.ceil(coords[a].max())) + 2) for a in range(3)]
    if any(hi[a] <= lo[a] for a in range(3)):
        return np.full(P, cval, np.float32)
    sub = np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    grid = np.meshgrid(*[coords[a] - lo[a] for a in range(3)], indexing='ij')
    out = ndi.map_coordinates(sub, np.stack(grid), order=order, mode='constant',
                              cval=cval, prefilter=False)
    return out.astype(np.float32)

def native_bbox_of_roi(shape, center_zyx, P, s, sp_zyx):
    """Native index bbox [lo, hi) that the ROI grid covers (clipped to volume)."""
    s3 = as_spacing3(s)
    lo, hi = [], []
    for a in range(3):
        c = _axis_coords(center_zyx[a], P[a], s3[a], sp_zyx[a])
        lo.append(max(0, int(np.floor(c.min()))))
        hi.append(min(shape[a], int(np.ceil(c.max())) + 1))
    return np.array(lo), np.array(hi)

def scatter_roi(roi_vol, out_shape, center_zyx, P, s, sp_zyx, order=1):
    """Inverse of sample_roi: resample ROI-grid values back onto the native grid.

    Returns (values, lo, hi) where values has shape hi-lo; caller pastes it.
    """
    import scipy.ndimage as ndi
    lo, hi = native_bbox_of_roi(out_shape, center_zyx, P, s, sp_zyx)
    if np.any(hi <= lo):
        return None, lo, hi
    s3 = as_spacing3(s)
    coords = []
    for a in range(3):
        n = np.arange(lo[a], hi[a], dtype=np.float64)
        coords.append((n - center_zyx[a]) * sp_zyx[a] / s3[a] + (P[a] - 1) / 2.0)
    grid = np.meshgrid(*coords, indexing='ij')
    vals = ndi.map_coordinates(roi_vol.astype(np.float32), np.stack(grid), order=order,
                               mode='constant', cval=0.0, prefilter=False)
    return vals, lo, hi
