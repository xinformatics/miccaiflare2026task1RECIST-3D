"""Local scorer: per-lesion Lesion_DSC and Lesion_NSD (tol=1.0mm), matching the challenge."""
import numpy as np
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

def dice(a, b):
    s = a.sum() + b.sum()
    if s == 0: return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / s)

def nsd(a, b, spacing_mm_zyx, tol=1.0):
    if not a.any() and not b.any(): return 1.0
    if not a.any() or not b.any(): return 0.0
    sd = compute_surface_distances(a, b, tuple(float(x) for x in spacing_mm_zyx))
    return float(compute_surface_dice_at_tolerance(sd, tol))

def score_case(pred, gt, spacing, ids=None):
    """spacing is npz (x,y,z). Mean over every lesion id present in gt."""
    szyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    if ids is None:
        ids = np.unique(gt); ids = ids[ids != 0]
    ds, ns = [], []
    for lid in ids:
        g = gt == lid
        p = pred == lid
        # crop to a joint bbox for speed
        idx = np.nonzero(g | p)
        if len(idx[0]) == 0:
            ds.append(1.0); ns.append(1.0); continue
        sl = tuple(slice(max(0, i.min() - 2), i.max() + 3) for i in idx)
        gc, pc = g[sl], p[sl]
        ds.append(dice(gc, pc)); ns.append(nsd(gc, pc, szyx))
    return float(np.mean(ds)), float(np.mean(ns)), ds, ns
