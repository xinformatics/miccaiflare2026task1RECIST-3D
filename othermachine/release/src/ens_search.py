"""Search member subsets and weights on the public validation set.

Members are ranked by solo score; subsets are taken best-first. Our model is the only soft
member, so it carries weight w0 and the binary members share (1-w0) equally -- with binary
votes, fine-grained per-member weights mostly reproduce the same decision boundary, so the
search is over subset size and w0 rather than a high-dimensional simplex.
"""
import os, sys, pickle, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom, score as S, roi_geom
from concurrent.futures import ProcessPoolExecutor
import scipy.ndimage as ndi

def native(e, shape):
    P = roi_geom.get(e.get('geo', 'v1'))['P_INF']
    v, lo, hi = geom.scatter_roi(e['prob'].astype(np.float32), shape, e['c'], P, e['s'], e['szyx'])
    o = np.zeros(shape, np.float32)
    if v is not None: o[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = v
    return o

def lcc(m):
    if not m.any(): return m
    lab, n = ndi.label(m)
    if n > 1: m = lab == (int(np.argmax(ndi.sum(m, lab, range(1, n+1)))) + 1)
    return m

def one(args):
    e, dirs, ref, plans = args
    z = np.load(os.path.join(ref, e['file']), allow_pickle=True)
    g = z['gts'] == e['lid']
    p = native(e, e['shape'])
    bs = []
    for d in dirs:
        fp = os.path.join(d, e['file'])
        bs.append((np.load(fp)['segs'] == e['lid']).astype(np.float32) if os.path.exists(fp) else None)
    out = {}
    for name, k, w0 in plans:
        use = [b for b in bs[:k] if b is not None]
        if not use:
            sc = p
        else:
            sc = w0 * p + ((1 - w0) / len(use)) * sum(use)
        m = lcc(sc > 0.5)
        idx = np.nonzero(g | m)
        if len(idx[0]) == 0: out[name] = (1.0, 1.0); continue
        sl = tuple(slice(max(0, i.min()-2), i.max()+3) for i in idx)
        out[name] = (S.dice(g[sl], m[sl]), S.nsd(g[sl], m[sl], tuple(e['szyx'])))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True); ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--ref', required=True)
    a = ap.parse_args()
    plans = [('ours alone', 0, 1.0)]
    for k in range(1, len(a.dirs) + 1):
        for w0 in (0.34, 0.40, 0.46, 0.52):
            plans.append((f'{k} member(s), w0={w0:.2f}', k, w0))
    store = pickle.load(open(a.ours, 'rb'))
    with ProcessPoolExecutor(10) as ex:
        rows = [r for r in ex.map(one, [(e, a.dirs, a.ref, plans) for e in store], chunksize=4) if r]
    print(f'{len(rows)} lesions   members in order: ' + ', '.join(os.path.basename(d) for d in a.dirs))
    best = None
    for name, _, _ in plans:
        v = np.array([r[name] for r in rows])
        d, n = v[:, 0].mean(), v[:, 1].mean()
        flag = ''
        if name != 'ours alone' and (best is None or d + n > best[1]):
            best = (name, d + n); flag = ' <-'
        print(f'  {name:22s} DSC {d:.4f}  NSD {n:.4f}{flag}')
    print(f'\nbest by DSC+NSD: {best[0]}')
