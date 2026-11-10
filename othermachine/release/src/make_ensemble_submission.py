"""Ensemble our probability with the organizers' baseline masks, on the hidden set.

The baseline reaches 0.7722/0.6291 by an entirely different mechanism (per-slice propagation
at native 512x512 from a foundation model), and agrees with us at only 0.765 Dice, so the two
error patterns are largely decorrelated -- the regime where averaging beats both members.

Only masks are available from the baseline, so the combination is w*p + (1-w)*b > 0.5, which
with binary b acts as an adaptive threshold: where the baseline votes yes we need p > 0.0909
(at w=0.55), where it votes no we need p > 0.909. w must exceed 0.5 or the rule collapses to
the baseline alone.
"""
import os, sys, glob, io, time, zipfile, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_anchor import load_net, ours_prob
import scipy.ndimage as ndi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--baseline-zip', default='')
    ap.add_argument('--member-dirs', nargs='*', default=[],
                    help='directories of binary member masks (one npz per case, key segs)')
    ap.add_argument('--weights', default='',
                    help='comma-separated: ours first, then one per member; must sum to 1')
    ap.add_argument('--out', required=True); ap.add_argument('--preds', required=True)
    ap.add_argument('--input', required=True, help='directory containing input npz cases')
    ap.add_argument('--w', type=float, default=0.58); ap.add_argument('--geo', default='v2')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    net = load_net(a.ckpt, a.device)
    bz = zipfile.ZipFile(a.baseline_zip) if a.baseline_zip else None
    have = set(bz.namelist()) if bz else set()
    ws = [float(v) for v in a.weights.split(',')] if a.weights else [a.w, 1 - a.w]
    nmem = len(a.member_dirs) if a.member_dirs else (1 if bz else 0)
    assert len(ws) == nmem + 1, f'need {nmem+1} weights, got {len(ws)}'
    print(f'{nmem} member(s), weights {ws}', flush=True)
    os.makedirs(a.preds, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, '*.npz')))
    ts = []
    for k, f in enumerate(files):
        t0 = time.time()
        n = os.path.basename(f)
        z = np.load(f, allow_pickle=True)
        imgs, rc, sp = z['imgs'], z['recist'], z['spacing']
        if a.member_dirs:
            mem = [np.load(os.path.join(d, n))['segs'] if os.path.exists(os.path.join(d, n))
                   else None for d in a.member_dirs]
        else:
            mem = [np.load(io.BytesIO(bz.read(n)))['segs'] if n in have else None]
        base = next((m for m in mem if m is not None), None)
        segs = np.zeros(imgs.shape, np.uint8); best = np.zeros(imgs.shape, np.uint8)
        ids = np.unique(rc); ids = ids[ids != 0]
        for lid in ids:
            p = ours_prob(net, imgs, rc, sp, lid, a.geo, a.device)
            score = ws[0] * p
            wsum = ws[0]
            for wi, m in zip(ws[1:], mem):
                if m is None: continue
                score = score + wi * (m == lid).astype(np.float32)
                wsum += wi
            if wsum > 0: score = score / wsum      # renormalize if a member is missing
            m = score > 0.5
            if m.any():
                lab, nl = ndi.label(m)
                if nl > 1:
                    zz, yy, xx = np.nonzero(rc == lid)
                    seeds = np.unique([v for v in (lab[zz, yy, xx] if len(zz) else []) if v > 0])
                    if len(seeds) == 0:
                        seeds = [int(np.argmax(ndi.sum(m, lab, range(1, nl + 1)))) + 1]
                    m = np.isin(lab, seeds)
            if not m.any() and base is not None:
                m = base == lid
            win = np.where(m, np.clip(score, 0, 1) * 255, 0).astype(np.uint8)
            take = m & (win > best)
            best = np.where(take, win, best); segs = np.where(take, np.uint8(lid), segs)
        np.savez_compressed(os.path.join(a.preds, n), segs=segs)
        ts.append(time.time() - t0)
        if k % 20 == 0: print(f'{k+1}/{len(files)} {n} {ts[-1]:.1f}s', flush=True)
    print(f'cases={len(ts)} mean {np.mean(ts):.1f}s', flush=True)
    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_STORED) as zf:
        for p in sorted(glob.glob(os.path.join(a.preds, '*.npz'))):
            zf.write(p, arcname=os.path.basename(p))
    import make_submission as MS
    print(); MS.verify(a.out)

if __name__ == '__main__':
    main()
