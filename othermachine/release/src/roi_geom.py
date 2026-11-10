"""The ROI grid definitions, in one place so training, inference and evaluation cannot drift.

v1  the original grid: P(112,128,128), s = clip(2.9*D/144, 0.5, 2.2) mm isotropic.
v2  finer grid: P(112,176,176), s = clip(2.9*D/192, 0.45, 1.10) with an FOV floor of 1.6*D.
    Motivated by NSD's 1 mm tolerance -- v1's spacing exceeds it above ~50 mm, where 29% of
    test lesions sit, and measured NSD there collapses while DSC stays flat.
"""
import numpy as np

def _v1(D):
    return float(np.clip(2.9 * D / 144.0, 0.5, 2.2))

def _v2(D):
    s = float(np.clip(2.9 * D / 192.0, 0.45, 1.10))
    return max(s, 1.6 * D / 192.0)

GEOMETRIES = {
    'v1': dict(P_INF=(112, 128, 128), P_CACHE=(128, 144, 144), spacing=_v1,
               cache_root='/media/shashank/NewVolume/c2/cache'),
    'v2': dict(P_INF=(112, 176, 176), P_CACHE=(128, 192, 192), spacing=_v2,
               cache_root='/media/shashank/NewVolume/c2/cache_geo'),
}

def get(name):
    if name not in GEOMETRIES:
        raise ValueError(f'unknown geometry {name!r}; have {sorted(GEOMETRIES)}')
    return GEOMETRIES[name]
