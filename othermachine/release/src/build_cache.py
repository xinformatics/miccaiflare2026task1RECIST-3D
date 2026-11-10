"""Per-lesion ROI cache: scale-normalized isotropic crops written to flat memmaps.

For every lesion (one RECIST prompt) we store an image ROI + binary mask ROI on an
isotropic grid whose spacing is proportional to the RECIST diameter, so every lesion
is seen at the same relative scale. Training crops a smaller window out of this with
jitter; inference takes the centered window.
"""
import os, sys, glob, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom

P_CACHE = (128, 144, 144)          # cached ROI grid (Z, Y, X)
FOV_MULT = 2.9                     # in-plane FOV = FOV_MULT * RECIST diameter
S_MIN, S_MAX = 0.5, 2.2            # isotropic ROI spacing bounds (mm)
NVOX = int(np.prod(P_CACHE))

def cache_spacing(diam_mm):
    return float(np.clip(FOV_MULT * diam_mm / P_CACHE[2], S_MIN, S_MAX))

def gen_recist_from_gts(gts, spacing):
    """Fallback when a training npz has no 'recist' key: redraw it like the organizers do."""
    import cv2
    from scipy.spatial.distance import pdist, squareform
    rc = np.zeros_like(gts, dtype=np.uint16)
    szyx = geom.spacing_zyx(spacing)
    for lid in np.unique(gts):
        if lid == 0: continue
        m = (gts == lid).astype(np.uint8)
        ks = int(np.argmax(m.sum(axis=(1, 2))))
        cnts, _ = cv2.findContours(m[ks], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        pts = np.vstack(cnts).squeeze()
        if pts.ndim != 2 or len(pts) < 2: continue
        if len(pts) > 500: pts = pts[np.linspace(0, len(pts) - 1, 500, dtype=int)]
        d = squareform(pdist(pts)); i, j = np.unravel_index(np.argmax(d), d.shape)
        p1, p2 = pts[i], pts[j]
        if np.hypot((p1[0]-p2[0])*szyx[2], (p1[1]-p2[1])*szyx[1]) < 10.0: continue
        cv2.line(rc[ks], (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 color=int(lid), thickness=2)
    return rc

def process_case(f):
    """-> list of (img_roi uint8, mask_roi bool, meta dict)"""
    out = []
    z = np.load(f, allow_pickle=True)
    imgs = z['imgs']; gts = z['gts']; sp = z['spacing']
    rc = z['recist'] if 'recist' in z.files else gen_recist_from_gts(gts, sp)
    szyx = geom.spacing_zyx(sp)
    ids = np.unique(rc); ids = ids[ids != 0]
    if imgs.dtype != np.uint8:
        imgs = np.clip(np.asarray(imgs, np.float32), 0, 255).astype(np.uint8)
    for lid in ids:
        m_full = (gts == lid)
        if not m_full.any():
            continue
        pr = geom.lesion_prompt(rc, lid, sp)
        D = pr['diam_mm']
        if D < 5.0: continue
        s = cache_spacing(D)
        c = pr['center_zyx']
        img_roi = geom.sample_roi(imgs, c, P_CACHE, s, szyx, order=1, cval=0.0)
        msk_roi = geom.sample_roi(m_full.astype(np.float32), c, P_CACHE, s, szyx,
                                  order=1, cval=0.0) > 0.5
        # RECIST endpoints expressed in ROI voxel coordinates
        ctr = np.array([(P_CACHE[0]-1)/2., (P_CACHE[1]-1)/2., (P_CACHE[2]-1)/2.])
        def to_roi(pyx):
            d = (np.array([c[0], pyx[0], pyx[1]]) - c) * szyx / s
            return ctr + d
        r1, r2 = to_roi(pr['p1']), to_roi(pr['p2'])
        vox_full = int(m_full.sum())
        vox_roi = int(msk_roi.sum())
        out.append((np.clip(img_roi, 0, 255).astype(np.uint8), msk_roi,
                    dict(case=os.path.basename(f)[:-4], lid=int(lid), diam=float(D),
                         s=float(s), p1y=float(r1[1]), p1x=float(r1[2]),
                         p2y=float(r2[1]), p2x=float(r2[2]),
                         cover=float(vox_roi * s**3 / max(vox_full * np.prod(szyx), 1e-6)),
                         nvox=vox_full, sz=float(szyx[0]), sxy=float(szyx[1]))))
    return out

def _worker(f):
    try:
        return f, process_case(f), None
    except Exception as e:
        return f, [], repr(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data-root', default='/media/shashank/NewVolume/c2/data/RECIST_train_npz')
    ap.add_argument('--out-root', default='/media/shashank/NewVolume/c2/cache')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    src = os.path.join(a.data_root, a.dataset)
    files = sorted(glob.glob(os.path.join(src, '*.npz')))
    out = os.path.join(a.out_root, a.dataset)
    os.makedirs(out, exist_ok=True)
    done_f = os.path.join(out, 'done.json')
    done = set(json.load(open(done_f))) if os.path.exists(done_f) else set()
    files = [f for f in files if os.path.basename(f) not in done]
    print(f'{a.dataset}: {len(files)} new files', flush=True)
    if not files: return
    fi = open(os.path.join(out, 'imgs.u8'), 'ab')
    fm = open(os.path.join(out, 'msks.u8'), 'ab')
    meta_path = os.path.join(out, 'meta.jsonl')
    fmeta = open(meta_path, 'a')
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (f, res, err) in enumerate(ex.map(_worker, files, chunksize=1)):
            if err: print('ERR', os.path.basename(f), err, flush=True)
            for img, msk, meta in res:
                fi.write(img.tobytes()); fm.write(np.packbits(msk.ravel()).tobytes())
                fmeta.write(json.dumps(meta) + '\n'); n += 1
            done.add(os.path.basename(f))
            if k % 50 == 0:
                fi.flush(); fm.flush(); fmeta.flush()
                json.dump(sorted(done), open(done_f, 'w'))
                el = time.time() - t0
                print(f'  {k+1}/{len(files)} cases, {n} lesions, {el:.0f}s, '
                      f'{(k+1)/max(el,1):.2f} case/s', flush=True)
    fi.close(); fm.close(); fmeta.close()
    json.dump(sorted(done), open(done_f, 'w'))
    print(f'{a.dataset} DONE: {n} lesions in {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
