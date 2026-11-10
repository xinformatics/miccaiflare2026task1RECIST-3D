"""Score prediction directories against the public validation set (the honest proxy)."""
import os, sys, glob, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as S
from concurrent.futures import ProcessPoolExecutor
REF = '/media/shashank/NewVolume/c2/data/validation_npz/validation_public_npz'

def one(args):
    pf, rf = args
    p = np.load(pf)['segs']; z = np.load(rf, allow_pickle=True)
    gt = z['gts']; szyx = (float(z['spacing'][2]), float(z['spacing'][1]), float(z['spacing'][0]))
    ids = np.unique(z['recist']); ids = ids[ids != 0]
    out = []
    for lid in ids:
        g = gt == lid
        if not g.any(): continue
        q = p == lid
        idx = np.nonzero(g | q)
        if len(idx[0]) == 0: out.append((1.0, 1.0)); continue
        sl = tuple(slice(max(0, i.min()-2), i.max()+3) for i in idx)
        out.append((S.dice(g[sl], q[sl]), S.nsd(g[sl], q[sl], szyx)))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()
    for d in a.dirs:
        jobs = [(os.path.join(d, os.path.basename(r)), r)
                for r in sorted(glob.glob(os.path.join(REF, '*.npz')))
                if os.path.exists(os.path.join(d, os.path.basename(r)))]
        if not jobs: print(f'{os.path.basename(d):28s} (no predictions)'); continue
        rows = []
        with ProcessPoolExecutor(10) as ex:
            for r in ex.map(one, jobs): rows += r
        arr = np.array(rows)
        print(f'{os.path.basename(d):28s} n={len(arr):3d}  DSC {arr[:,0].mean():.4f}  NSD {arr[:,1].mean():.4f}')
