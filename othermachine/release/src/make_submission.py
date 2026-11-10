"""Predict the hidden validation set and package a submission zip, with the format checks
that silently zero a run if violated (nested dirs, renamed cases, wrong shape/dtype)."""
import os, sys, glob, zipfile, argparse, io
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HIDDEN = '/media/shashank/NewVolume/c2/data/validation_npz/validation_hidden_npz'

def verify(zip_path, ref_dir=HIDDEN):
    """Every check the scorer can fail on, before we hand it over."""
    ok = True
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    ref = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ref_dir, '*.npz')))
    def chk(cond, msg):
        nonlocal ok
        print(('  OK   ' if cond else '  FAIL ') + msg)
        ok = ok and cond
    chk(all('/' not in n for n in names), 'flat archive (no nested directory)')
    chk(sorted(names) == ref, f'{len(names)} entries, names match the test set exactly')
    bad_shape = bad_dtype = bad_key = bad_ids = 0
    for n in names:
        d = np.load(io.BytesIO(zf.read(n)))
        if 'segs' not in d.files: bad_key += 1; continue
        segs = d['segs']
        r = np.load(os.path.join(ref_dir, n), allow_pickle=True)
        if segs.shape != r['imgs'].shape: bad_shape += 1
        if segs.dtype != np.uint8: bad_dtype += 1
        pid = set(np.unique(r['recist']).tolist()) - {0}
        sid = set(np.unique(segs).tolist()) - {0}
        if not sid.issubset(pid): bad_ids += 1
    chk(bad_key == 0, "every file stores array key 'segs'")
    chk(bad_shape == 0, 'every segs shape matches its imgs volume')
    chk(bad_dtype == 0, 'every segs is uint8')
    chk(bad_ids == 0, 'no lesion id outside the prompt ids')
    print(('PASS' if ok else 'FAIL') + f'  {zip_path}  {os.path.getsize(zip_path)/1e6:.1f} MB')
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint, or comma-separated for an ensemble')
    ap.add_argument('--out', required=True, help='output zip path')
    ap.add_argument('--preds', default=None)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--tta', type=int, default=0)
    ap.add_argument('--thr', type=float, default=0.5); ap.add_argument('--calib', type=float, default=0.0)
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--geo', default='v1')
    a = ap.parse_args()
    if a.verify_only:
        sys.exit(0 if verify(a.out) else 1)
    import predict as P
    pred_dir = a.preds or a.out.replace('.zip', '_npz')
    net = P.load_net(a.ckpt, a.device)
    P.run(net, HIDDEN, pred_dir, a.device, a.tta, a.thr, threads=8, calib=a.calib, geo=a.geo)
    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_STORED) as zf:
        for f in sorted(glob.glob(os.path.join(pred_dir, '*.npz'))):
            zf.write(f, arcname=os.path.basename(f))
    print()
    verify(a.out)

if __name__ == '__main__':
    main()
