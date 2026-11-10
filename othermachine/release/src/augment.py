"""GPU affine augmentation: samples the training patch out of the larger cached ROI with
random rotation / scale / shift / flip, and carries the RECIST endpoints through the same
transform so the prompt channels stay registered to the image."""
import torch, torch.nn.functional as F

def _rot(ax, ang):
    c, s = torch.cos(ang), torch.sin(ang)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    if ax == 0:   # about z: rotates the axial (y,x) plane
        return torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1),
                            torch.stack([z, s, c], -1)], -2)
    if ax == 1:
        return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1),
                            torch.stack([-s, z, c], -1)], -2)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], -2)

def random_affine(B, dev, train=True, rot_z=0.35, rot_xy=0.18, scale=0.15, shift=6.0):
    """-> (R (B,3,3) output->input offsets in ZYX voxels, t (B,3), zoom (B,)).

    `zoom` is the per-sample sampling-step factor: the augmented patch has physical voxel
    size s*zoom, so callers must scale the mm-valued prompt features by it.
    """
    if not train:
        return (torch.eye(3, device=dev).expand(B, 3, 3).clone(),
                torch.zeros(B, 3, device=dev), torch.ones(B, device=dev))
    u = lambda a, b: torch.rand(B, device=dev) * (b - a) + a
    R = _rot(0, u(-rot_z, rot_z)) @ _rot(1, u(-rot_xy, rot_xy)) @ _rot(2, u(-rot_xy, rot_xy))
    zoom = u(1.0 - scale, 1.0 + scale)
    R = R * zoom.view(B, 1, 1)
    flip = (torch.rand(B, 3, device=dev) < 0.5).float() * (-2) + 1     # +1 / -1 per axis
    R = R * flip.view(B, 1, 3)
    t = (torch.rand(B, 3, device=dev) * 2 - 1) * shift
    return R, t, zoom

def apply_affine(vol, R, t, P_out, mode='bilinear'):
    """vol: (B,1,Z,Y,X) float. Samples P_out from vol under o -> C_in + R(o - C_out) + t."""
    B = vol.shape[0]; dev = vol.device
    P_in = torch.tensor(vol.shape[2:], device=dev, dtype=torch.float32)
    P_o = torch.tensor(P_out, device=dev, dtype=torch.float32)
    grids = [torch.arange(P_out[a], device=dev, dtype=torch.float32) - (P_o[a] - 1) / 2
             for a in range(3)]
    oz, oy, ox = torch.meshgrid(*grids, indexing='ij')
    o = torch.stack([oz, oy, ox], -1).view(1, -1, 3)                    # (1,N,3)
    i = torch.einsum('bij,bnj->bni', R, o.expand(B, -1, -1)) + t[:, None, :]
    i = i + ((P_in - 1) / 2).view(1, 1, 3)
    n = 2 * i / (P_in - 1).view(1, 1, 3) - 1                            # normalized ZYX
    grid = torch.stack([n[..., 2], n[..., 1], n[..., 0]], -1)           # grid_sample wants XYZ
    grid = grid.view(B, P_out[0], P_out[1], P_out[2], 3)
    return F.grid_sample(vol, grid, mode=mode, padding_mode='zeros', align_corners=True)

def transform_points(p, R, t, P_in, P_out):
    """Cache-frame points (B,3) -> output-frame coords, the inverse of apply_affine."""
    dev = p.device
    Pi = torch.tensor(P_in, device=dev, dtype=torch.float32)
    Po = torch.tensor(P_out, device=dev, dtype=torch.float32)
    d = p - (Pi - 1).view(1, 3) / 2 - t
    o = torch.linalg.solve(R, d.unsqueeze(-1)).squeeze(-1)
    return o + (Po - 1).view(1, 3) / 2


def simulate_slice_thickness(im, s_mm, native_z, p=0.5, tmin=2.5, tmax=6.5):
    """Simulate a thicker-slice acquisition on the image only.

    The training corpus is far thinner-sliced than the hidden test set (median 2.5 mm vs
    3.7 mm; for lung, no held-out case exceeds 3.5 mm while 45% of hidden lung does), so
    without this the model is tested out of distribution. We blur and re-discretize along z
    to the target thickness, leaving the mask at true anatomical resolution: the metric is
    computed by resampling our prediction onto the native grid, so the correct target is
    still the fine mask, and the model learns to recover it from thick-slice appearance.

    im: (B,Z,Y,X); s_mm: (B,) ROI voxel size; native_z: (B,) acquisition z spacing.
    """
    B, Z = im.shape[0], im.shape[1]
    dev = im.device
    do = (torch.rand(B, device=dev) < p)
    if not bool(do.any()):
        return im
    target = torch.rand(B, device=dev) * (tmax - tmin) + tmin
    # only thicken; never pretend a coarse acquisition was finer than it was
    target = torch.maximum(target, native_z)
    factor = (target / s_mm.clamp_min(1e-6)).clamp(1.0, Z / 4.0)
    out = im
    for f in factor[do].unique():
        sel = do & (factor == f)
        if not bool(sel.any()):
            continue
        zt = max(int(round(Z / float(f))), 4)
        x = im[sel][:, None]
        x = F.interpolate(x, size=(zt, im.shape[2], im.shape[3]), mode='trilinear',
                          align_corners=False)
        x = F.interpolate(x, size=(Z, im.shape[2], im.shape[3]), mode='trilinear',
                          align_corners=False)
        out = out.clone() if out is im else out
        out[sel] = x[:, 0]
    return out
