"""
AMOS22-MRI abdominal multi-organ loader.

The MANY-CLASS, NON-CT anchor for the class-count hypothesis. AMOS annotates
15 abdominal organs (labels 1..15, +bg => n_classes=16) on MRI. This is the
one cell the study lacks: high class count on a NON-CT modality. If Muon
shows a large benefit here like it does on Synapse(9)/FLARE(13) CT, that's
strong evidence class count -- not "abdominal CT" -- drives the effect.

AMOS22 layout (Zenodo / grand-challenge):
    <root>/imagesTr/amos_XXXX.nii.gz     (CT and MRI MIXED, by case number)
    <root>/labelsTr/amos_XXXX.nii.gz
    (imagesVa/labelsVa similar; imagesTs has no labels -> unused)

CRITICAL -- CT vs MRI split: AMOS mixes CT and MRI in the same imagesTr
folder, distinguished by CASE NUMBER. In AMOS22, CT cases are the lower
IDs and MRI cases are the higher IDs (MRI starts around amos_0500+). The
EXACT threshold must be confirmed against your download -- run the check
script and set MRI_MIN_ID. We keep ONLY MRI cases (>= MRI_MIN_ID). The MRI
set is small (40 train + 20 val labeled), so we pool all labeled MRI cases
and carve our own patient-level train/val/test split like FLARE/BraTS.

15 organs (label ids): 1 spleen, 2 R-kidney, 3 L-kidney, 4 gallbladder,
5 esophagus, 6 liver, 7 stomach, 8 aorta, 9 IVC, 10 pancreas,
11 R-adrenal, 12 L-adrenal, 13 duodenum, 14 bladder, 15 prostate/uterus.
Labels are contiguous 0..15 => n_classes=16, no remap needed (verify).

MRI z-score normalization over nonzero voxels (like BraTS), NOT CT
windowing. Reuses the FLARE/BraTS structure: foreground axial slices for
train/val, whole volumes for test (per-volume 3D scoring), npz fast path.
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

AMOS_ROOT = "./data/AMOS22"
MRI_MIN_ID = 500          # keep amos_XXXX with XXXX >= this as MRI (CONFIRM!)

SLICE_STRIDE = 2
MAX_SLICES_PER_VOL = 60


def _load_nii(path):
    return np.asanyarray(nib.load(path).dataobj).astype(np.float32)


def _znorm(img):
    """Z-score over nonzero voxels (MRI has no absolute scale)."""
    nz = img > 0
    if nz.sum() < 10:
        return img
    m, s = img[nz].mean(), img[nz].std() + 1e-8
    out = (img - m) / s
    out[~nz] = out[nz].min()
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
    for suf in (".nii.gz", ".nii"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def _case_num(path):
    """Extract the integer XXXX from amos_XXXX."""
    cid = _case_id(path)
    digits = "".join(ch for ch in cid if ch.isdigit())
    return int(digits) if digits else -1


def _find_pairs(root, mri_min_id=MRI_MIN_ID):
    """Return [(image, label), ...] for MRI cases only (case num >= mri_min_id)."""
    pairs = []
    for sub in ("imagesTr", "imagesVa"):
        imgs = sorted(glob.glob(os.path.join(root, sub, "amos_*.nii*")))
        lab_sub = sub.replace("images", "labels")
        for ip in imgs:
            if _case_num(ip) < mri_min_id:      # skip CT cases
                continue
            lp = os.path.join(root, lab_sub, os.path.basename(ip))
            if os.path.exists(lp):
                pairs.append((ip, lp))
    if not pairs:
        raise FileNotFoundError(
            f"no AMOS MRI pairs (case >= {mri_min_id}) under {root}; "
            f"check MRI_MIN_ID against your data")
    return pairs


class AmosSliceDataset(Dataset):
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
            raise FileNotFoundError("no AMOS foreground slices")
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
        lab = np.rint(self._vol(lp)[:, :, z]).astype(np.int64)
        img = _znorm(img)
        img = _resize2d(img, self.size, 1)
        lab = _resize2d(lab.astype(np.float32), self.size, 0).astype(np.int64)
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, self.rng)
        img = _norm(img)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()), cid)


class AmosVolumeDataset(Dataset):
    def __init__(self, pairs, size=224):
        self.pairs = pairs
        self.size = size
        self.cases = [_case_id(ip) for ip, lp in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = _load_nii(ip)
        lab = np.rint(_load_nii(lp)).astype(np.int64)
        img = np.transpose(img, (2, 0, 1))
        lab = np.transpose(lab, (2, 0, 1))
        ims, lbs = [], []
        for z in range(img.shape[0]):
            ims.append(_norm(_resize2d(_znorm(img[z]), self.size, 1)))
            lbs.append(_resize2d(lab[z].astype(np.float32), self.size, 0).astype(np.int64))
        return (torch.from_numpy(np.stack(ims)[:, None]),
                torch.from_numpy(np.stack(lbs)), self.cases[i])


def _split(pairs, val_frac, test_frac):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    n_test = max(1, int(len(pairs) * test_frac))
    n_val = max(1, int(len(pairs) * val_frac))
    test_i = set(idx[:n_test].tolist())
    val_i = set(idx[n_test:n_test + n_val].tolist())
    train = [i for i in range(len(pairs)) if i not in test_i and i not in val_i]
    return train, sorted(val_i), sorted(test_i)


def build_amos_datasets(root=None, size=224, aug_regime="full", seed=0,
                        n_classes=16, data_path=None,
                        val_frac=0.1, test_frac=0.2, mri_min_id=MRI_MIN_ID):
    root = root or data_path or AMOS_ROOT
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    pairs = _find_pairs(root, mri_min_id)
    tr_i, va_i, te_i = _split(pairs, val_frac, test_frac)
    tr = AmosSliceDataset([pairs[i] for i in tr_i], size, cfg, seed)
    va = AmosSliceDataset([pairs[i] for i in va_i], size, None, seed)
    te = AmosVolumeDataset([pairs[i] for i in te_i], size)
    return tr, va, te


# ---- npz fast path (see preprocess_amos.py) ----
class AmosNpzSliceDataset(Dataset):
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


def build_amos_npz_datasets(root=None, size=224, aug_regime="full", seed=0,
                            n_classes=16, data_path=None):
    root = root or data_path or (AMOS_ROOT + "_npz")
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    with open(os.path.join(root, "manifest.json")) as f:
        man = _json.load(f)
    tr = AmosNpzSliceDataset(root, man["train"], size, cfg, seed)
    va = AmosNpzSliceDataset(root, man["val"], size, None, seed)
    te = AmosVolumeDataset([(t["image"], t["label"]) for t in man["test_vols"]], size)
    return tr, va, te
