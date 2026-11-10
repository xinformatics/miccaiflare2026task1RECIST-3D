import os, sys, time, json, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
import dataset as DS, model as M, augment as AUG

def boundary_band(t, k=2):
    """Voxels within k of the target surface (max-pool dilation minus erosion)."""
    kk = 2 * k + 1
    dil = F.max_pool3d(t, kk, stride=1, padding=k)
    ero = -F.max_pool3d(-t, kk, stride=1, padding=k)
    return (dil - ero).clamp(0, 1)

def dice_bce_loss(logit, target, eps=1e-5, band_w=0.0, band_k=2):
    """Dice + BCE, optionally up-weighting BCE near the target surface.

    NSD scores surface placement rather than overlap, so plain Dice+BCE spends most of its
    gradient on interior voxels the metric is indifferent to. Our NSD/DSC ratio (0.790) trails
    the baseline's (0.815), which is the deficit this term targets.
    """
    p = torch.sigmoid(logit.float())
    t = target.float()
    dims = (2, 3, 4)
    inter = (p * t).sum(dims); den = p.sum(dims) + t.sum(dims)
    dice = 1 - (2 * inter + eps) / (den + eps)
    if band_w > 0:
        w = 1.0 + band_w * boundary_band(t, band_k)
        raw = F.binary_cross_entropy_with_logits(logit.float(), t, reduction='none')
        bce = (raw * w).sum() / w.sum().clamp_min(1.0)
    else:
        bce = F.binary_cross_entropy_with_logits(logit.float(), t)
    return dice.mean() + bce

# Sampling weights per source dataset, roughly matching the hidden-validation mix
# (Kidney/Liver/Pancreas ~equal, Lung a bit less, Colon small) plus a slice of the
# other cancer types for generalization.
TARGET_W = {
    'Dataset003_LungCancer': 0.18, 'Dataset004_LiverCancer': 0.20,
    'Dataset006_PancreaticCancer': 0.20, 'Dataset007_KidneyCancer': 0.20,
    'Dataset009_ColonCancer': 0.08, 'Dataset002_EsophagusCancer': 0.025,
    'Dataset005_AdrenalCancer': 0.02, 'Dataset008_LymphNodes': 0.04,
    'Dataset010_EndometrialCancer': 0.02, 'Dataset011_WholeBodyLesion': 0.035,
}

def ds_weights(ds_names, diams=None, size_prior=None, cap=25.0, alpha=0.75):
    """Per-lesion sampling weights: balance across source datasets, and (optionally) reweight
    within each dataset so the sampled RECIST-diameter distribution matches the test set.

    Training lesions are far smaller than test lesions (liver: 71% under 20 mm vs 13%), and
    below ~25 mm the ROI spacing clips at its floor so those lesions are not scale-normalized
    the way larger ones are. Without this the model trains mostly in a size regime the test
    set barely contains.
    """
    from collections import Counter
    cnt = Counter(ds_names)
    present = {d: TARGET_W.get(d, 0.01) for d in cnt}
    tot = sum(present.values())
    print('sampling mix: ' + ', '.join(f'{d.split("_")[-1]}={present[d]/tot:.3f}(n={cnt[d]})'
                                       for d in sorted(present)), flush=True)
    base = np.array([present[d] / tot / cnt[d] for d in ds_names], dtype=np.float64)
    if size_prior is None or diams is None:
        return base
    bins = np.array(size_prior['bins'], dtype=np.float64)
    b = np.clip(np.digitize(np.asarray(diams, np.float64), bins) - 1, 0, len(bins) - 2)
    ds_arr = np.asarray(ds_names)
    sw = np.ones(len(base))
    for d in cnt:
        sel = ds_arr == d
        tgt = np.array(size_prior['groups'].get(d, size_prior['overall']), dtype=np.float64)
        obs = np.bincount(b[sel], minlength=len(tgt)).astype(np.float64)
        obs /= max(obs.sum(), 1.0)
        ratio = np.where(obs > 0, tgt / np.maximum(obs, 1e-12), 0.0)
        # alpha<1 is a partial correction: full matching would cut effective unique
        # lesions from 7205 to 2963, and most of the gap closes well before that.
        ratio = np.clip(ratio, 1.0 / cap, cap) ** alpha
        sw[sel] = ratio[b[sel]]
        # keep each dataset's overall share unchanged; only its internal size mix shifts
        sw[sel] *= sel.sum() / max(sw[sel].sum(), 1e-12)
    w = base * sw
    eff = w.sum() ** 2 / max((w ** 2).sum(), 1e-12)
    print(f'size-stratified sampling on: effective unique lesions {eff:.0f} of {len(w)}',
          flush=True)
    return w


def intensity_aug(im):
    """im: (B,Z,Y,X) in [0,255]. Per-sample gamma / brightness / contrast / noise."""
    B = im.shape[0]; dev = im.device
    x = im / 255.0
    r = lambda: torch.rand(B, 1, 1, 1, device=dev)
    g = torch.exp((r() * 2 - 1) * 0.35)
    x = torch.where(r() < 0.3, x.clamp(0, 1) ** g, x)
    x = torch.where(r() < 0.3, x * (0.85 + 0.3 * r()) + (r() - 0.5) * 0.16, x)
    x = torch.where(r() < 0.2, x + torch.randn_like(x) * (0.005 + 0.025 * r()), x)
    return x.clamp(0, 1) * 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--steps-per-epoch', type=int, default=250)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='/media/shashank/NewVolume/c2/models/v1')
    ap.add_argument('--resume', default='auto')
    ap.add_argument('--datasets', default='')
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--balance', type=int, default=1)
    ap.add_argument('--rot-z', type=float, default=0.35)
    ap.add_argument('--rot-xy', type=float, default=0.18)
    ap.add_argument('--scale-aug', type=float, default=0.15)
    ap.add_argument('--shift-aug', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--thick-aug', type=float, default=0.5)
    ap.add_argument('--size-match', type=int, default=1)
    ap.add_argument('--size-alpha', type=float, default=0.75)
    ap.add_argument('--cache-root', default='/media/shashank/NewVolume/c2/cache')
    ap.add_argument('--p-cache', default='128,144,144')
    ap.add_argument('--p-train', default='112,128,128')
    ap.add_argument('--init', default='', help='warm-start weights from another checkpoint')
    ap.add_argument('--band-w', type=float, default=0.0)
    ap.add_argument('--band-k', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True

    DS.configure([int(v) for v in a.p_cache.split(',')],
                 [int(v) for v in a.p_train.split(',')])
    print(f'ROI grid: cache {DS.P_CACHE} train {DS.P_TRAIN} from {a.cache_root}', flush=True)
    ds_filter = a.datasets.split(',') if a.datasets else None
    ix = DS.CacheIndex(root=a.cache_root, datasets=ds_filter)
    tr, va = DS.split_rows(ix, 0.04, seed=0)
    print(f'lesions total={ix.n} train={len(tr)} val={len(va)}', flush=True)
    json.dump({'val_rows': va.tolist()[:100000]}, open(os.path.join(a.out, 'split.json'), 'w'))
    dtr = DS.LesionROIDataset(ix, tr, True)
    dva = DS.LesionROIDataset(ix, va, False)
    if a.balance:
        from torch.utils.data import WeightedRandomSampler
        prior = None
        if a.size_match:
            pp = '/media/shashank/NewVolume/c2/cache/hidden_size_prior.json'
            if os.path.exists(pp): prior = json.load(open(pp))
            else: print('WARNING: size prior missing; sampling by dataset only', flush=True)
        w = ds_weights([ix.meta[r]['ds'] for r in tr],
                       diams=[ix.meta[r]['diam'] for r in tr], size_prior=prior,
                       alpha=a.size_alpha)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=a.steps_per_epoch * a.bs, replacement=True)
        ltr = DataLoader(dtr, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                         pin_memory=True, drop_last=True, persistent_workers=True,
                         prefetch_factor=4)
    else:
        ltr = DataLoader(dtr, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                         pin_memory=True, drop_last=True, persistent_workers=True,
                         prefetch_factor=4)
    lva = DataLoader(dva, batch_size=a.bs, shuffle=False, num_workers=4, pin_memory=True)

    net = M.UNet3D(in_ch=4, base=a.base).to(dev).to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    ep0 = 0; best = 0.0
    ck = os.path.join(a.out, 'last.pt')
    if a.init and not os.path.exists(ck):
        w = torch.load(a.init, map_location='cpu')['net']
        net.load_state_dict(w)
        print(f'warm-started from {a.init}', flush=True)
    if a.resume == 'auto' and os.path.exists(ck):
        st = torch.load(ck, map_location='cpu')
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt'])
        ep0 = st['epoch'] + 1; best = st.get('best', 0.0)
        print(f'resumed from epoch {ep0}', flush=True)

    size_prior = None
    _pp = '/media/shashank/NewVolume/c2/cache/hidden_size_prior.json'
    if os.path.exists(_pp): size_prior = json.load(open(_pp))
    dsw = [1.0, 0.5, 0.25]; dsw = [w / sum(dsw) for w in dsw]
    total_steps = a.epochs * a.steps_per_epoch

    def prep(batch, train=True):
        img, msk, p1, p2, d, s, nz = [t.to(dev, non_blocking=True) for t in batch]
        B = img.shape[0]
        R, t, zoom = AUG.random_affine(B, dev, train=train, rot_z=a.rot_z, rot_xy=a.rot_xy,
                                       scale=a.scale_aug, shift=a.shift_aug)
        im = AUG.apply_affine(img.float()[:, None], R, t, DS.P_TRAIN, 'bilinear')[:, 0]
        mk = (AUG.apply_affine(msk.float()[:, None], R, t, DS.P_TRAIN, 'bilinear') > 0.5).float()
        q1 = AUG.transform_points(p1, R, t, DS.P_CACHE, DS.P_TRAIN)
        q2 = AUG.transform_points(p2, R, t, DS.P_CACHE, DS.P_TRAIN)
        if train:
            im = AUG.simulate_slice_thickness(im, s * zoom, nz, p=a.thick_aug)
            im = intensity_aug(im)
        # zoom changes the physical voxel size of the sampled patch
        x = M.build_input(im, q1, q2, d, s * zoom, DS.P_TRAIN).to(memory_format=torch.channels_last_3d)
        return x, mk

    step = ep0 * a.steps_per_epoch
    it = iter(ltr)
    for ep in range(ep0, a.epochs):
        net.train(); t0 = time.time(); losses = []
        for i in range(a.steps_per_epoch):
            try: batch = next(it)
            except StopIteration:
                it = iter(ltr); batch = next(it)
            lr = a.lr * (1 - step / total_steps) ** 0.9
            for g in opt.param_groups: g['lr'] = lr
            x, y = prep(batch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outs = net(x)
            loss = 0.0
            for k, o in enumerate(outs):
                yy = y if k == 0 else F.avg_pool3d(y, 2 ** k)
                loss = loss + dsw[k] * dice_bce_loss(o, yy, band_w=a.band_w, band_k=a.band_k)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
            opt.step(); step += 1
            losses.append(float(loss.detach()))
        msg = f'ep {ep} loss {np.mean(losses):.4f} lr {lr:.2e} {time.time()-t0:.0f}s'
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            net.eval(); ds = []; dia = []
            with torch.inference_mode():
                for batch in lva:
                    x, y = prep(batch, train=False)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        o = net(x)
                    p = (torch.sigmoid(o.float()) > 0.5).float()
                    inter = (p * y).sum((1, 2, 3, 4)); den = p.sum((1, 2, 3, 4)) + y.sum((1, 2, 3, 4))
                    ds += ((2 * inter + 1e-5) / (den + 1e-5)).tolist()
                    dia += batch[4].tolist()
            vd = float(np.mean(ds)) if ds else 0.0
            # The plain mean is uniform over held-out lesions, i.e. small-lesion heavy, which
            # is not the test distribution -- report the size-reweighted mean alongside it so
            # the log tracks the quantity we actually care about.
            vw = vd
            if ds and size_prior is not None:
                bb = np.clip(np.digitize(np.array(dia), np.array(size_prior['bins'])) - 1,
                             0, len(size_prior['bins']) - 2)
                tgt = np.array(size_prior['overall']); dsa = np.array(ds)
                num = den_ = 0.0
                for k in range(len(tgt)):
                    sel = bb == k
                    if sel.any() and tgt[k] > 0:
                        num += tgt[k] * dsa[sel].mean(); den_ += tgt[k]
                if den_ > 0: vw = float(num / den_)
            msg += f' | valROIdice {vd:.4f} sizeWt {vw:.4f}'
            if vw > best:
                best = vw
                torch.save({'net': net.state_dict(), 'base': a.base, 'epoch': ep, 'best': best},
                           os.path.join(a.out, 'best.pt'))
                msg += ' *'
        print(msg, flush=True)
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'base': a.base,
                    'epoch': ep, 'best': best}, ck)

if __name__ == '__main__':
    main()
