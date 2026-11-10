"""Hybrid: our 3D model supplies SAM2's anchor mask, SAM2 propagates it through slices.

The stock pipeline turns the RECIST box into an anchor mask via SAM2's own prompt encoder,
then propagates. But our 3D model is measurably better at the prompted slice (it sees the line,
a distance prior and absolute scale), while SAM2's strength is slice-to-slice propagation from
a foundation-model prior. Swapping only the anchor keeps each where it is strong.
"""
import os, sys, glob, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, roi_geom, model as M
def load_net(ckpt, device='cpu'):
    """Load the released 3D checkpoint without optional training utilities."""
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    net = M.UNet3D(in_ch=4, base=st.get('base', 16))
    net.load_state_dict(st['net'])
    return net.eval().to(device)

REPO = os.environ.get(
    'MEDSAM2_REPO',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'baseline')),
)
sys.path.insert(0, REPO)

@torch.inference_mode()
def ours_prob(net, imgs, recist, spacing, lid, geo, device):
    G = roi_geom.get(geo); P = G['P_INF']
    szyx = geom.spacing_zyx(spacing)
    pr = geom.lesion_prompt(recist, lid, spacing)
    D = max(pr['diam_mm'], 5.0); s = G['spacing'](D); c = pr['center_zyx']
    ctr = np.array([(P[a]-1)/2. for a in range(3)])
    roi = geom.sample_roi(imgs, c, P, s, szyx, order=1, cval=0.0)
    to_roi = lambda pyx: ctr + (np.array([c[0], pyx[0], pyx[1]]) - c) * szyx / s
    x = M.build_input(torch.from_numpy(roi)[None].to(device),
                      torch.tensor(to_roi(pr['p1']), dtype=torch.float32)[None].to(device),
                      torch.tensor(to_roi(pr['p2']), dtype=torch.float32)[None].to(device),
                      torch.tensor([D], device=device), torch.tensor([s], device=device), P)
    prob = torch.sigmoid(net(x).float())[0, 0].cpu().numpy()
    vals, lo, hi = geom.scatter_roi(prob, imgs.shape, c, P, s, szyx, order=1)
    out = np.zeros(imgs.shape, np.float32)
    if vals is not None:
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = vals
    return out

@torch.inference_mode()
def hybrid_case(eff, predictor, net, imgs, recist, spacing, geo, device, use_ours_anchor=True,
                anchor_masks=None, anchor_w=0.4, multi_anchor=0, anchor_step=3,
                external_anchor=None):
    segs = np.zeros(imgs.shape, np.uint8)
    ids = np.unique(recist); ids = ids[ids != 0]
    H, W = imgs.shape[1], imgs.shape[2]
    if (H, W) != (eff.IMAGE_SIZE, eff.IMAGE_SIZE):
        img_r = eff.resize_grayscale_to_rgb_and_resize(imgs, eff.IMAGE_SIZE)
    else:
        img_r = imgs[:, None].repeat(3, axis=1)
    img_r = torch.from_numpy(img_r).float().div_(255.0)
    img_r -= torch.tensor(eff.IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    img_r /= torch.tensor(eff.IMAGENET_STD, dtype=torch.float32)[:, None, None]
    for ulab in ids:
        per = (recist == ulab) * ulab
        if not per.any(): continue
        z_mid_orig = int(np.unique(np.where(recist == ulab)[0])[0])
        diameter = eff.recist_diameter(per[z_mid_orig])
        z_span = diameter * (((spacing[0] + spacing[1]) / 2) / spacing[2])
        half_z = int(z_span * eff.Z_SLAB_SCALE / 2)
        z_lo = max(0, z_mid_orig - half_z); z_hi = min(imgs.shape[0] - 1, z_mid_orig + half_z)
        crop = per[z_lo:z_hi + 1]; img_crop = img_r[z_lo:z_hi + 1]
        z_mid = z_mid_orig - z_lo
        state = predictor.init_state(img_crop, H, W)
        if use_ours_anchor:
            # A complete external 3D/fused prediction can be used directly as
            # SAM2's key-slice mask. This differs from ``anchor_masks``, which
            # merely votes binary masks into the released model's probability.
            # Fall back to the released 3D probability if the external member
            # misses this lesion on the RECIST slice.
            p3 = None
            if external_anchor is not None:
                anchor = (external_anchor[z_mid_orig] == ulab).astype(np.uint8)
            else:
                anchor = np.zeros((H, W), dtype=np.uint8)
            if external_anchor is None or anchor.sum() < 4:
                p3 = ours_prob(net, imgs, recist, spacing, ulab, geo, device)
            if external_anchor is not None and anchor.sum() >= 4:
                pass
            elif anchor_masks:
                sc = anchor_w * p3[z_mid_orig]
                wsum = anchor_w
                wi = (1.0 - anchor_w) / len(anchor_masks)
                for am in anchor_masks:
                    sc = sc + wi * (am[z_mid_orig] == ulab).astype(np.float32)
                    wsum += wi
                anchor = (sc / wsum > 0.5).astype(np.uint8)
            else:
                anchor = (p3[z_mid_orig] > 0.5).astype(np.uint8)
            if anchor.sum() < 4:                      # our model found nothing: fall back
                box = eff.recist_bbox(crop[z_mid], scale=eff.BOX_SCALE, shift=0)
                _, _, lg = predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=z_mid, obj_id=1, box=box)
                anchor = (lg[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            p3 = None
            box = eff.recist_bbox(crop[z_mid], scale=eff.BOX_SCALE, shift=0)
            _, _, lg = predictor.add_new_points_or_box(
                inference_state=state, frame_idx=z_mid, obj_id=1, box=box)
            anchor = (lg[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
        def seed(st, reverse):
            predictor.add_new_mask(st, frame_idx=z_mid, obj_id=1, mask=anchor)
            if multi_anchor and use_ours_anchor:
                sgn = -1 if reverse else 1
                for k in range(1, multi_anchor + 1):
                    zk = z_mid + sgn * k * anchor_step
                    zk_o = z_mid_orig + sgn * k * anchor_step
                    if zk < 0 or zk >= img_crop.shape[0] or zk_o < 0 or zk_o >= imgs.shape[0]:
                        break
                    extra = (p3[zk_o] > 0.5).astype(np.uint8)
                    if extra.sum() < 4:      # nothing there: stop seeding further out
                        break
                    predictor.add_new_mask(st, frame_idx=zk, obj_id=1, mask=extra)

        _, _, masks = predictor.add_new_mask(state, frame_idx=z_mid, obj_id=1, mask=anchor)
        segs[z_mid_orig, (masks[0] > 0.0).cpu().numpy()[0]] = ulab
        predictor.reset_state(state)
        for reverse in (False, True):
            state = predictor.init_state(img_crop, H, W)
            seed(state, reverse)
            for fi, _, lg in predictor.propagate_in_video(state, start_frame_idx=z_mid,
                                                          reverse=reverse):
                segs[z_lo + fi, (lg[0] > 0.0).cpu().numpy()[0]] = ulab
            predictor.reset_state(state)
    return segs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', required=True); ap.add_argument('-o', required=True)
    ap.add_argument('--ours', required=True); ap.add_argument('--size', default='small')
    ap.add_argument('--sam-weights', default='',
                    help='override the Efficient MedSAM2 checkpoint path')
    ap.add_argument('--geo', default='v2'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--box-scale', type=float, default=0.9)
    ap.add_argument('--stock-anchor', action='store_true')
    ap.add_argument('--anchor-dirs', nargs='*', default=[],
                    help='member mask dirs to vote into the anchor alongside our model; the '
                         'ensemble is a better key-slice estimator than our model alone '
                         '(0.8479 vs 0.8262), and the anchor is what propagation amplifies')
    ap.add_argument('--anchor-w', type=float, default=0.4)
    ap.add_argument('--external-anchor-dir', default='',
                    help='use this complete prediction directly as the SAM2 key-slice mask; '
                         'missing/empty lesion anchors fall back to the released 3D model')
    ap.add_argument('--multi-anchor', type=int, default=0,
                    help='anchor SAM2 at this many slices either side of the key slice, not '
                         'just one. Propagation amplifies whatever the anchor got wrong, and '
                         'our 3D model has a mask for every slice, so constraining more of '
                         'the path should bound the drift.')
    ap.add_argument('--anchor-step', type=int, default=3)
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'eff', os.path.join(REPO, 'Efficient_MedSAM2_infer_CT_lesion_npz_recist.py'))
    eff = importlib.util.module_from_spec(spec); eff.__dict__['__name__'] = 'eff'
    spec.loader.exec_module(eff)
    eff.BOX_SCALE = a.box_scale
    ck = a.sam_weights or os.path.join(
        REPO, 'checkpoints', f'eff_medsam2_{a.size}_FLARE25_RECIST_baseline.pt')
    ns = argparse.Namespace(cfg=None, model_size=a.size, weights=ck, device=a.device,
                            image_size=512, no_compile=True)
    predictor, _ = eff.build_predictor(ns)
    net = load_net(a.ours, a.device)
    os.makedirs(a.o, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.i, '*.npz')))
    t0 = time.time()
    for k, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        ams = None
        if a.anchor_dirs:
            n = os.path.basename(f)
            ams = [np.load(os.path.join(d, n))['segs'] for d in a.anchor_dirs
                   if os.path.exists(os.path.join(d, n))] or None
        ext = None
        if a.external_anchor_dir:
            ep = os.path.join(a.external_anchor_dir, os.path.basename(f))
            if os.path.exists(ep):
                ext = np.load(ep)['segs']
        segs = hybrid_case(eff, predictor, net, z['imgs'], z['recist'], z['spacing'],
                           a.geo, a.device, not a.stock_anchor, ams, a.anchor_w,
                           a.multi_anchor, a.anchor_step, ext)
        np.savez_compressed(os.path.join(a.o, os.path.basename(f)), segs=segs)
        if k % 10 == 0: print(f'{k+1}/{len(files)} {time.time()-t0:.0f}s', flush=True)
    print(f'DONE {len(files)} in {time.time()-t0:.0f}s', flush=True)
