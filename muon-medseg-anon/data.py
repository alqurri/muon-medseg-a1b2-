"""
Data + the two augmentation regimes.

Expected on-disk layout (preprocess your own ACDC/Synapse into this):

    <root>/train/case0001_slice012.npz   with keys 'image' (H,W) float32
                                                   'label' (H,W) int64
    <root>/val/...
    <root>/test/...

`case` is parsed from the filename prefix and is used for patient-level
grouping in the stats -- slices from one patient are NOT independent samples,
and treating them as such is the fastest way to manufacture a fake p-value.

If root is None you get SyntheticSeg, which exists so you can verify the whole
harness end-to-end before touching real data.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Two regimes. The contrast between them IS the Stage 3 hypothesis: the
# published ViT result showed AdamW degrading as augmentation strength went up
# while Muon held or improved. If that reproduces on medical data it is a
# finding; if not, that is worth reporting too.
AUG_LIGHT = dict(flip=True, rot90=False, affine=False, elastic=False,
                 gamma=False, noise=False, blur=False)
AUG_FULL = dict(flip=True, rot90=True, affine=True, elastic=True,
                gamma=True, noise=True, blur=True)


def _rand_affine(img, lab, rng, max_rot=15.0, scale=(0.85, 1.15), shift=0.08):
    import scipy.ndimage as ndi
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale)
    H, W = img.shape
    img = ndi.rotate(img, ang, reshape=False, order=1, mode="constant")
    lab = ndi.rotate(lab, ang, reshape=False, order=0, mode="constant")
    zi = ndi.zoom(img, sc, order=1)
    zl = ndi.zoom(lab, sc, order=0)
    img, lab = _center_fit(zi, H, W, 0.0), _center_fit(zl, H, W, 0)
    sy, sx = (rng.uniform(-shift, shift) * np.array([H, W])).astype(int)
    img = np.roll(np.roll(img, sy, 0), sx, 1)
    lab = np.roll(np.roll(lab, sy, 0), sx, 1)
    return img, lab


def _center_fit(a, H, W, pad_val):
    h, w = a.shape
    if h >= H:
        t = (h - H) // 2
        a = a[t:t + H]
    else:
        p = H - h
        a = np.pad(a, ((p // 2, p - p // 2), (0, 0)), constant_values=pad_val)
    if w >= W:
        l = (w - W) // 2
        a = a[:, l:l + W]
    else:
        p = W - w
        a = np.pad(a, ((0, 0), (p // 2, p - p // 2)), constant_values=pad_val)
    return a


def _elastic(img, lab, rng, alpha=34.0, sigma=4.0):
    import scipy.ndimage as ndi
    H, W = img.shape
    dx = ndi.gaussian_filter(rng.uniform(-1, 1, (H, W)), sigma) * alpha
    dy = ndi.gaussian_filter(rng.uniform(-1, 1, (H, W)), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    crd = [np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)]
    return (ndi.map_coordinates(img, crd, order=1),
            ndi.map_coordinates(lab, crd, order=0))


def augment(img, lab, cfg, rng):
    if cfg["flip"] and rng.random() < 0.5:
        img, lab = img[:, ::-1].copy(), lab[:, ::-1].copy()
    if cfg["rot90"] and rng.random() < 0.5:
        k = rng.integers(1, 4)
        img, lab = np.rot90(img, k).copy(), np.rot90(lab, k).copy()
    if cfg["affine"] and rng.random() < 0.5:
        img, lab = _rand_affine(img, lab, rng)
    if cfg["elastic"] and rng.random() < 0.25:
        img, lab = _elastic(img, lab, rng)
    if cfg["gamma"] and rng.random() < 0.3:
        g = rng.uniform(0.7, 1.5)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img - lo) / (hi - lo)) ** g * (hi - lo) + lo
    if cfg["noise"] and rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(0.01, 0.08), img.shape)
    if cfg["blur"] and rng.random() < 0.2:
        import scipy.ndimage as ndi
        img = ndi.gaussian_filter(img, rng.uniform(0.5, 1.2))
    return img.astype(np.float32), lab.astype(np.int64)


def _resolve_keys(path):
    keys = list(np.load(path).keys())
    ik = next((k for k in ("img", "image", "data", "arr_0") if k in keys), None)
    lk = next((k for k in ("label", "gt", "mask", "seg", "arr_1") if k in keys), None)
    if ik is None or lk is None:
        raise KeyError(f"{path} has keys {keys}; edit alias lists in data.py")
    return ik, lk


def _case_id(fname):
    """case_001_sliceED_0.npz -> case_001 (patient-level id).

    The stats pair at patient level. Splitting on '_' and taking [0] would
    make every id 'case', collapsing all patients into one group and
    invalidating every p-value. Join the first two fields instead.
    """
    parts = os.path.basename(fname).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


class SliceDataset(Dataset):
    def __init__(self, root, split, size=224, aug=None, seed=0):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no .npz under {root}/{split}")
        self.size, self.aug = size, aug
        self.rng = np.random.default_rng(seed)
        self._img_key, self._lab_key = _resolve_keys(self.files[0])
        self.cases = [_case_id(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = np.squeeze(d[self._img_key].astype(np.float32))
        lab = np.squeeze(np.rint(d[self._lab_key]).astype(np.int64))
        if img.ndim != 2 or lab.shape != img.shape:
            raise ValueError(f"{self.files[i]}: img {img.shape}, lab {lab.shape} "
                             f"after squeeze (expected matching 2D)")
        img = _center_fit(img, self.size, self.size, 0.0)
        lab = _center_fit(lab, self.size, self.size, 0)
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, self.rng)
        m, s = img.mean(), img.std()
        img = (img - m) / (s + 1e-8)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()),
                self.cases[i])


class VolumeDataset(Dataset):
    """Test-time dataset: one .npz == one (N,H,W) volume.

    ACDC test data is stored per-volume, not per-slice, because segmentation
    is scored per-volume -- HD95 in particular only means anything in 3D.
    Returns the whole stack as (N,1,H,W); evaluate() iterates its slices and
    aggregates metrics per volume.

    NOTE: train/val here are 2D slices, test is 3D volumes. That is the ACDC
    convention, but it means test metrics are per-volume and NOT directly
    comparable to slice-level numbers. Both optimizer arms are scored the
    same way, so the COMPARISON is valid; the absolute values are not a
    published-baseline match.
    """

    def __init__(self, root, split, size=224):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no .npz under {root}/{split}")
        self.size = size
        self._img_key, self._lab_key = _resolve_keys(self.files[0])
        # one case id per volume file, e.g. case_008_volume_ED -- each test
        # volume is already one patient/phase, so the filename stem is the id
        self.cases = [os.path.basename(f)[:-4] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d[self._img_key].astype(np.float32)
        lab = np.rint(d[self._lab_key]).astype(np.int64)
        if img.ndim == 2:                      # tolerate a stray 2D test file
            img, lab = img[None], lab[None]
        if img.shape != lab.shape:
            raise ValueError(f"{self.files[i]}: img {img.shape} != lab {lab.shape}")
        ims, lbs = [], []
        for n in range(img.shape[0]):
            im = _center_fit(img[n], self.size, self.size, 0.0)
            lb = _center_fit(lab[n], self.size, self.size, 0)
            m, s = im.mean(), im.std()
            ims.append((im - m) / (s + 1e-8))
            lbs.append(lb)
        vol = np.stack(ims)[:, None]           # (N,1,H,W)
        msk = np.stack(lbs)                     # (N,H,W)
        return torch.from_numpy(vol), torch.from_numpy(msk), self.cases[i]


class SyntheticSeg(Dataset):
    """Smoke-test data. Random ellipses on noise, 4 classes.

    Not a benchmark -- its only job is to prove the loop, the optimizers, the
    metrics and the stats all run before you spend two days of GPU on it.
    """

    def __init__(self, n=256, size=224, n_classes=4, aug=None, seed=0, n_cases=16):
        self.n, self.size, self.k, self.aug = n, size, n_classes, aug
        self.seed = seed
        self.cases = [f"syn{i % n_cases:03d}" for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        S = self.size
        img = rng.normal(0, 0.25, (S, S)).astype(np.float32)
        lab = np.zeros((S, S), np.int64)
        yy, xx = np.mgrid[0:S, 0:S]
        for c in range(1, self.k):
            cy, cx = rng.integers(S // 4, 3 * S // 4, 2)
            ry, rx = rng.integers(S // 12, S // 6, 2)
            m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            lab[m] = c
            img[m] += 0.8 * c
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, np.random.default_rng(i))
        m_, s_ = img.mean(), img.std()
        img = (img - m_) / (s_ + 1e-8)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()), self.cases[i])


def build_datasets(root, size=224, aug_regime="full", seed=0, n_classes=4):
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    if root is None:
        return (SyntheticSeg(256, size, n_classes, cfg, seed),
                SyntheticSeg(64, size, n_classes, None, seed + 7777),
                SyntheticSeg(64, size, n_classes, None, seed + 12345))
    return (SliceDataset(root, "train", size, cfg, seed),
            SliceDataset(root, "val", size, None, seed),
            VolumeDataset(root, "test", size))
