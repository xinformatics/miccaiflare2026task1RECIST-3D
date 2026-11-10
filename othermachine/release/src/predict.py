"""RECIST -> 3D inference. Produces {case}.npz with key `segs` (uint8, lesion ids preserved)."""
import os, sys, glob, time, argparse, zipfile
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, model as M
from calib import calibrate_threshold
from build_cache import cache_spacing
from baseline_geom import predict_case as geom_case
import roi_geom

P_INF = (112, 128, 128)

def load_net(ckpt, device='cpu'):
    """Load one checkpoint, or an ensemble from a comma-separated list of them."""
    paths = [p for p in str(ckpt).split(',') if p]
    nets = []
    for p in paths:
        st = torch.load(p, map_location='cpu')
        net = M.UNet3D(in_ch=4, base=st.get('base', 16))
        net.load_state_dict(st['net']); net.eval().to(device)
        nets.append(net)
    return Ensemble(nets) if len(nets) > 1 else nets[0]

class Ensemble(torch.nn.Module):
    """Averages member logits. Members share the input contract, so callers see one model."""
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.eval()
    def forward(self, x):
        out = None
        for n in self.nets:
            o = n(x)
            out = o if out is None else out + o
        return out / len(self.nets)

@torch.inference_mode()
def predict_case(net, imgs, recist, spacing, device='cpu', tta=0, thr=0.5, keep_cc=True,
                 calib=0.0, geo='v1'):
    G = roi_geom.get(geo); P_INF = G['P_INF']; spacing_fn = G['spacing']
    szyx = geom.spacing_zyx(spacing)
    segs = np.zeros(imgs.shape, np.uint8)
    # Overlap arbitration only needs to compare probabilities, so keep the running best as
    # uint8 -- on a 1168-slice volume a float32 copy would cost 1.2 GB of the 8 GB budget.
    bestp = np.zeros(imgs.shape, np.uint8)
    ids = np.unique(recist); ids = ids[ids != 0]
    ctr = np.array([(P_INF[0]-1)/2., (P_INF[1]-1)/2., (P_INF[2]-1)/2.])
    for lid in ids:
        pr = geom.lesion_prompt(recist, lid, spacing)
        D = max(pr['diam_mm'], 5.0)
        s = spacing_fn(D)
        c = pr['center_zyx']
        roi = geom.sample_roi(imgs, c, P_INF, s, szyx, order=1, cval=0.0)
        def to_roi(pyx):
            d = (np.array([c[0], pyx[0], pyx[1]]) - c) * szyx / s
            return ctr + d
        p1 = torch.tensor(to_roi(pr['p1']), dtype=torch.float32)[None]
        p2 = torch.tensor(to_roi(pr['p2']), dtype=torch.float32)[None]
        img = torch.from_numpy(roi)[None].to(device)
        x = M.build_input(img, p1.to(device), p2.to(device),
                          torch.tensor([D], device=device), torch.tensor([s], device=device), P_INF)
        prob = torch.sigmoid(net(x).float())
        if tta:
            axes = [(2,), (3,), (4,), (3, 4)][:tta]
            for ax in axes:
                xf = torch.flip(x, ax)
                prob = prob + torch.sigmoid(torch.flip(net(xf), ax).float())
            prob = prob / (1 + len(axes))
        prob = prob[0, 0].cpu().numpy()
        thr_l = thr
        if calib > 0:
            zk = float(ctr[0])
            r1n, r2n = to_roi(pr['p1']), to_roi(pr['p2'])   # rasterize the line for CC seeding
            nseg = max(int(np.hypot(r1n[1]-r2n[1], r1n[2]-r2n[2])) * 2, 2)
            tt = np.linspace(0, 1, nseg)
            ly = np.clip(np.round(r1n[1] + tt*(r2n[1]-r1n[1])).astype(int), 0, P_INF[1]-1)
            lx = np.clip(np.round(r1n[2] + tt*(r2n[2]-r1n[2])).astype(int), 0, P_INF[2]-1)
            thr_l = calibrate_threshold(prob, zk, pr['diam_true_mm'], s,
                                        line_yx=(ly, lx), blend=calib)
        vals, lo, hi = geom.scatter_roi(prob, imgs.shape, c, P_INF, s, szyx, order=1)
        if vals is None: continue
        sub = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        m = vals > thr_l
        if keep_cc and m.any():
            import scipy.ndimage as ndi
            lab, n = ndi.label(m)
            if n > 1:
                zz, yy, xx = np.nonzero(recist[sub] == lid)
                seeds = lab[zz, yy, xx] if len(zz) else []
                seeds = np.unique([v for v in seeds if v > 0])
                if len(seeds) == 0:
                    sizes = ndi.sum(m, lab, range(1, n + 1))
                    seeds = [int(np.argmax(sizes)) + 1]
                m = np.isin(lab, seeds)
        if not m.any():
            # Safety net: the prompt guarantees a lesion is here, so never emit nothing.
            fb = geom_case(imgs.shape, np.where(recist == lid, recist, 0), spacing)
            m = fb[sub] > 0
            vals = np.where(m, 0.5, vals).astype(np.float32)
        win = np.where(m, np.clip(vals, 0, 1) * 255, 0).astype(np.uint8)
        take = m & (win > bestp[sub])
        bestp[sub] = np.where(take, win, bestp[sub])
        segs[sub] = np.where(take, np.uint8(lid), segs[sub])
    return segs

def run(net, in_dir, out_dir, device='cpu', tta=0, thr=0.5, limit=None, threads=8, quiet=False,
        calib=0.0, geo='v1'):
    torch.set_num_threads(threads)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, '*.npz')))[:limit]
    times = []
    for i, f in enumerate(files):
        t0 = time.time()
        z = np.load(f, allow_pickle=True)
        segs = predict_case(net, z['imgs'], z['recist'], z['spacing'], device, tta, thr,
                            calib=calib, geo=geo)
        np.savez_compressed(os.path.join(out_dir, os.path.basename(f)), segs=segs)
        times.append(time.time() - t0)
        if not quiet and i % 10 == 0:
            print(f'{i+1}/{len(files)} {os.path.basename(f)} {times[-1]:.1f}s', flush=True)
    if times:
        print(f'cases={len(times)} mean {np.mean(times):.1f}s max {np.max(times):.1f}s', flush=True)
    return times

def make_zip(out_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for f in sorted(glob.glob(os.path.join(out_dir, '*.npz'))):
            zf.write(f, arcname=os.path.basename(f))
    print('zip:', zip_path, f'{os.path.getsize(zip_path)/1e6:.1f} MB')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', required=True); ap.add_argument('-o', required=True)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--tta', type=int, default=0); ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--zip', default=None); ap.add_argument('--calib', type=float, default=0.0)
    ap.add_argument('--geo', default='v1')
    a = ap.parse_args()
    net = load_net(a.ckpt, a.device)
    run(net, a.i, a.o, a.device, a.tta, a.thr, a.limit, a.threads, calib=a.calib, geo=a.geo)
    if a.zip: make_zip(a.o, a.zip)
