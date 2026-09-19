"""
BraTS 2020 brain-tumor MRI loader.

A hard, NON-CT anchor for the difficulty study: multi-region brain tumor
segmentation on MRI, genuinely low baseline Dice (enhancing tumor / core are
small and hard). Breaks the "both hard tasks are abdominal CT" confound.

Config: single FLAIR modality (1-channel, matches the other
single-channel datasets), 4-class subregion labels.

Raw layout (Kaggle mirror awsaf49/brats20-dataset-training-validation):
    <root>/MICCAI_BraTS2020_TrainingData/BraTS20_Training_XXX/
        BraTS20_Training_XXX_flair.nii   (used)
        BraTS20_Training_XXX_t1.nii / _t1ce.nii / _t2.nii   (unused here)
        BraTS20_Training_XXX_seg.nii     (labels {0,1,2,4})
Files are UNCOMPRESSED .nii on this mirror (not .nii.gz); we glob *_flair.nii*
and *_seg.nii* so either works.

LABEL REMAP: raw BraTS labels are {0,1,2,4} (0 bg, 1 necrotic/non-enhancing
core, 2 edema, 4 enhancing tumor). The 4 is non-contiguous, so we map
4 -> 3, giving contiguous {0,1,2,3} => n_classes=4. This keeps the three
tumor subregions as separate classes (the hard, low-baseline task) rather
than collapsing to whole-tumor binary.

MRI NORMALIZATION (differs from FLARE's CT windowing): MRI has no absolute
intensity scale, so we z-score each volume over its BRAIN (nonzero) voxels
only -- BraTS volumes are skull-stripped, so background is exactly 0 and
including it would swamp the statistics. No HU windowing (that's CT-only).

Structure mirrors data_flare.py: foreground axial slices for train/val,
whole volumes for test (per-volume 3D scoring), fixed-seed patient-level
split, and a preprocess_brats.py fast path (see that file) via
build_brats_npz_datasets.
"""
from __future__ import annotations

import glob
import os
import json as _json
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

from data import augment, AUG_FULL, AUG_LIGHT

BRATS_ROOT = "./data/BraTS2020"

SLICE_STRIDE = 3          # brain spans ~100 axial slices; stride 3 keeps enough
MAX_SLICES_PER_VOL = 50

# raw BraTS label -> contiguous. 4 (enhancing) -> 3; others unchanged.
_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}


def _load_nii(path):
    return np.asanyarray(nib.load(path).dataobj).astype(np.float32)


def _remap_label(lab):
    out = np.zeros_like(lab, dtype=np.int64)
    out[lab == 1] = 1
    out[lab == 2] = 2
    out[lab == 4] = 3
    return out


def _znorm_brain(img):
    """Z-score over nonzero (brain) voxels; skull-stripped background stays 0-ish."""
    brain = img > 0
    if brain.sum() < 10:
        return img
    m = img[brain].mean()
    s = img[brain].std() + 1e-8
    out = (img - m) / s
    out[~brain] = out[brain].min()   # push background below brain range
    return out


def _resize2d(a, size, order):
    import cv2
    interp = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
    return cv2.resize(a, (size, size), interpolation=interp)


def _norm(x):
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-8)


def _case_id(path):
    b = os.path.basename(path)
    for suf in ("_flair.nii.gz", "_flair.nii", "_seg.nii.gz", "_seg.nii"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b.split(".")[0]


def _find_pairs(root):
    """Return [(flair_path, seg_path), ...]."""
    flairs = sorted(glob.glob(os.path.join(root, "**", "*_flair.nii*"), recursive=True))
    pairs = []
    for fp in flairs:
        sp = fp.replace("_flair.nii", "_seg.nii")
        if os.path.exists(sp):
            pairs.append((fp, sp))
    if not pairs:
        raise FileNotFoundError(f"no BraTS flair/seg pairs under {root}")
    return pairs


class BratsSliceDataset(Dataset):
    """Foreground axial slices (train/val)."""

    def __init__(self, pairs, size=224, aug=None, seed=0,
                 stride=SLICE_STRIDE, max_slices=MAX_SLICES_PER_VOL):
        self.size, self.aug = size, aug
        self.rng = np.random.default_rng(seed)
        self.index = []
        for ip, lp in pairs:
            lab = _load_nii(lp)
            fg_z = np.where((lab > 0).any(axis=(0, 1)))[0]
            if len(fg_z) == 0:
                continue
            fg_z = fg_z[::stride]
            if len(fg_z) > max_slices:
                fg_z = self.rng.choice(fg_z, max_slices, replace=False)
            cid = _case_id(ip)
            for z in fg_z:
                self.index.append((ip, lp, int(z), cid))
        if not self.index:
            raise FileNotFoundError("no foreground slices for BraTS split")
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _vol(self, path):
        if path not in self._cache:
            self._cache = {}
            self._cache[path] = _load_nii(path)
        return self._cache[path]

    def __getitem__(self, i):
        ip, lp, z, cid = self.index[i]
        img = self._vol(ip)[:, :, z]
        lab = _remap_label(np.rint(self._vol(lp)[:, :, z]).astype(np.int64))
        img = _znorm_brain(img)
        img = _resize2d(img, self.size, 1)
        lab = _resize2d(lab.astype(np.float32), self.size, 0).astype(np.int64)
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, self.rng)
        img = _norm(img)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()), cid)


class BratsVolumeDataset(Dataset):
    """Whole volumes for test, scored per-volume in 3D."""

    def __init__(self, pairs, size=224):
        self.pairs = pairs
        self.size = size
        self.cases = [_case_id(ip) for ip, lp in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = _load_nii(ip)
        lab = _remap_label(np.rint(_load_nii(lp)).astype(np.int64))
        img = np.transpose(img, (2, 0, 1))          # (D,H,W)
        lab = np.transpose(lab, (2, 0, 1))
        ims, lbs = [], []
        for z in range(img.shape[0]):
            im = _znorm_brain(img[z])
            ims.append(_norm(_resize2d(im, self.size, 1)))
            lbs.append(_resize2d(lab[z].astype(np.float32), self.size, 0).astype(np.int64))
        vol = np.stack(ims)[:, None]
        msk = np.stack(lbs)
        return torch.from_numpy(vol), torch.from_numpy(msk), self.cases[i]


def _split(pairs, val_frac, test_frac):
    rng = np.random.default_rng(0)                  # FIXED split seed
    idx = rng.permutation(len(pairs))
    n_test = max(1, int(len(pairs) * test_frac))
    n_val = max(1, int(len(pairs) * val_frac))
    test_i = set(idx[:n_test].tolist())
    val_i = set(idx[n_test:n_test + n_val].tolist())
    train = [i for i in range(len(pairs)) if i not in test_i and i not in val_i]
    return train, sorted(val_i), sorted(test_i)


def build_brats_datasets(root=None, size=224, aug_regime="full", seed=0,
                         n_classes=4, data_path=None,
                         val_frac=0.1, test_frac=0.15):
    root = root or data_path or BRATS_ROOT
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    pairs = _find_pairs(root)
    train_i, val_i, test_i = _split(pairs, val_frac, test_frac)
    tr = BratsSliceDataset([pairs[i] for i in train_i], size, cfg, seed)
    va = BratsSliceDataset([pairs[i] for i in val_i], size, None, seed)
    te = BratsVolumeDataset([pairs[i] for i in test_i], size)
    return tr, va, te


# ==================================================================
# FAST PATH: pre-extracted .npy slices (see preprocess_brats.py)
# ==================================================================
class BratsNpzSliceDataset(Dataset):
    def __init__(self, root, entries, size=224, aug=None, seed=0):
        self.root, self.entries = root, entries
        self.size, self.aug = size, aug
        self.rng = np.random.default_rng(seed)
        self.cases = [e["case"] for e in entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        arr = np.load(os.path.join(self.root, self.entries[i]["file"]))
        img = arr[0].astype(np.float32)
        lab = np.rint(arr[1]).astype(np.int64)
        img = _resize2d(img, self.size, 1)
        lab = _resize2d(lab.astype(np.float32), self.size, 0).astype(np.int64)
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, self.rng)
        img = _norm(img)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()), self.cases[i])


def build_brats_npz_datasets(root=None, size=224, aug_regime="full", seed=0,
                             n_classes=4, data_path=None):
    root = root or data_path or (BRATS_ROOT + "_npz")
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    with open(os.path.join(root, "manifest.json")) as f:
        man = _json.load(f)
    tr = BratsNpzSliceDataset(root, man["train"], size, cfg, seed)
    va = BratsNpzSliceDataset(root, man["val"], size, None, seed)
    test_pairs = [(t["flair"], t["seg"]) for t in man["test_vols"]]
    te = BratsVolumeDataset(test_pairs, size)
    return tr, va, te
