"""Memmap-backed lesion-ROI dataset with on-the-fly crop/flip/intensity augmentation."""
import os, json, glob
import numpy as np
import torch
from torch.utils.data import Dataset

P_CACHE = (128, 144, 144)
NVOX = int(np.prod(P_CACHE))
NPACK = (NVOX + 7) // 8
P_TRAIN = (112, 128, 128)

def configure(p_cache=None, p_train=None):
    """Point the loader at a cache built with a different ROI grid."""
    global P_CACHE, P_TRAIN, NVOX, NPACK
    if p_cache is not None:
        P_CACHE = tuple(int(v) for v in p_cache)
        NVOX = int(np.prod(P_CACHE)); NPACK = (NVOX + 7) // 8
    if p_train is not None:
        P_TRAIN = tuple(int(v) for v in p_train)
    return P_CACHE, P_TRAIN

class CacheIndex:
    """Union of per-dataset caches; row -> (memmap, local row, meta)."""
    def __init__(self, root='/media/shashank/NewVolume/c2/cache', datasets=None):
        # NVOX/NPACK are read live so configure() applies to indexes built afterwards
        self.parts = []
        dirs = sorted(glob.glob(os.path.join(root, 'Dataset*')))
        if datasets: dirs = [d for d in dirs if os.path.basename(d) in datasets]
        self.meta = []
        for d in dirs:
            mp = os.path.join(d, 'meta.jsonl')
            if not os.path.exists(mp): continue
            metas = [json.loads(l) for l in open(mp)]
            ni = os.path.getsize(os.path.join(d, 'imgs.u8')) // NVOX
            nm = os.path.getsize(os.path.join(d, 'msks.u8')) // NPACK
            n = min(len(metas), ni, nm)
            if n == 0: continue
            pi = len(self.parts)
            self.parts.append(dict(dir=d, n=n, imgs=None, msks=None))
            for k in range(n):
                m = metas[k]; m['part'] = pi; m['row'] = k; m['ds'] = os.path.basename(d)
                self.meta.append(m)
        self.n = len(self.meta)

    def _open(self, pi):
        p = self.parts[pi]
        if p['imgs'] is None:
            p['imgs'] = np.memmap(os.path.join(p['dir'], 'imgs.u8'), np.uint8, 'r',
                                  shape=(p['n'],) + P_CACHE)
            p['msks'] = np.memmap(os.path.join(p['dir'], 'msks.u8'), np.uint8, 'r',
                                  shape=(p['n'], NPACK))
        return p

    def get(self, i):
        m = self.meta[i]; p = self._open(m['part']); r = m['row']
        img = np.asarray(p['imgs'][r])
        msk = np.unpackbits(np.asarray(p['msks'][r])).reshape(P_CACHE)
        return img, msk, m

class LesionROIDataset(Dataset):
    """Returns the whole cached ROI; geometric augmentation happens on GPU (see augment.py)."""
    def __init__(self, index, rows, train=True, patch=None):
        self.ix = index; self.rows = np.asarray(rows); self.train = train
        self.patch = patch or P_TRAIN
    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        img, msk, m = self.ix.get(int(self.rows[i]))
        ctr0 = (P_CACHE[0] - 1) / 2.
        p1 = np.array([ctr0, m['p1y'], m['p1x']], np.float32)
        p2 = np.array([ctr0, m['p2y'], m['p2x']], np.float32)
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(msk)),
                torch.from_numpy(p1), torch.from_numpy(p2),
                torch.tensor(float(m['diam'])), torch.tensor(float(m['s'])),
                torch.tensor(float(m.get('sz', 2.5))))

class _LegacyCropDataset(Dataset):
    def __init__(self, index, rows, train=True, patch=P_TRAIN):
        self.ix = index; self.rows = np.asarray(rows); self.train = train; self.patch = patch
    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        img, msk, m = self.ix.get(int(self.rows[i]))
        P = self.patch
        maxo = [P_CACHE[a] - P[a] for a in range(3)]
        if self.train:
            off = [np.random.randint(0, mo + 1) for mo in maxo]
        else:
            off = [mo // 2 for mo in maxo]
        sl = tuple(slice(off[a], off[a] + P[a]) for a in range(3))
        img = np.ascontiguousarray(img[sl]).astype(np.float32)
        msk = np.ascontiguousarray(msk[sl]).astype(np.float32)
        ctr = np.array([(P_CACHE[0]-1)/2., (P_CACHE[1]-1)/2., (P_CACHE[2]-1)/2.])
        p1 = np.array([ctr[0], m['p1y'], m['p1x']]) - np.array(off, float)
        p2 = np.array([ctr[0], m['p2y'], m['p2x']]) - np.array(off, float)
        if self.train:
            for a in range(3):
                if np.random.rand() < 0.5:
                    img = np.flip(img, a); msk = np.flip(msk, a)
                    p1[a] = P[a] - 1 - p1[a]; p2[a] = P[a] - 1 - p2[a]
            img = np.ascontiguousarray(img); msk = np.ascontiguousarray(msk)
            if np.random.rand() < 0.3:
                p1 += np.random.uniform(-1, 1, 3); p2 += np.random.uniform(-1, 1, 3)
            # intensity augmentation on [0,255]
            x = img / 255.0
            if np.random.rand() < 0.3:
                g = np.exp(np.random.uniform(-0.35, 0.35)); x = np.clip(x, 0, 1) ** g
            if np.random.rand() < 0.3:
                x = x * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.08, 0.08)
            if np.random.rand() < 0.2:
                x = x + np.random.normal(0, np.random.uniform(0.005, 0.03), x.shape).astype(np.float32)
            img = np.clip(x, 0, 1).astype(np.float32) * 255.0
        return (torch.from_numpy(img), torch.from_numpy(msk[None]),
                torch.tensor(p1, dtype=torch.float32), torch.tensor(p2, dtype=torch.float32),
                torch.tensor(float(m['diam'])), torch.tensor(float(m['s'])))

def case_hash(ds, case, seed=0):
    """Stable 0..1 hash of a case, so the split does not shift as datasets are added."""
    import hashlib
    h = hashlib.md5(f'{seed}|{ds}|{case}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2 ** 64

def is_val_case(ds, case, val_frac=0.04, seed=0):
    return case_hash(ds, case, seed) < val_frac

def split_rows(index, val_frac=0.04, seed=0):
    """Case-level split, stable under a growing cache."""
    tr, va = [], []
    for i, m in enumerate(index.meta):
        (va if is_val_case(m['ds'], m['case'], val_frac, seed) else tr).append(i)
    return np.array(tr), np.array(va)
