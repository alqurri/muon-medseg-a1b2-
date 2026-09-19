"""
Synapse (multi-organ abdominal CT) loaders, matching the standard TransUNet
preprocessing layout:

  train/val : 2D slices as <train_npz>/<name>.npz, keys 'image'/'label',
              shape (512,512); the set of names comes from a LIST FILE
              (lists_Synapse/train.txt), one 'caseXXXX_sliceYYY' per line,
              NOT from globbing the directory.
  test      : 3D volumes as <test_vol_h5>/<case>.npy.h5 (h5py), keys
              'image'/'label', shape (N,512,512); names from test_vol.txt.

Differences from the ACDC loaders that matter:
  * 9 classes (background + 8 organs), not 4  -> pass --n-classes 9
  * native 512x512, resized to `size` (224)   -> zoom, not center-crop,
    because Synapse organs can sit near the frame edge and cropping would
    clip them; label uses order=0 (nearest) to keep class ids exact.
  * organ imbalance: small organs (gallbladder, pancreas) are absent from
    many slices, so the empty-mask NaN handling in metrics.py does real
    work here. Per-organ Dice variance is higher than ACDC by nature.

This module is standalone; it does not modify the ACDC loaders in data.py.
build_synapse_datasets() returns the same (train, val, test) tuple shape as
build_datasets(), so train.py/run_study.py can consume it unchanged.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data import augment, AUG_FULL, AUG_LIGHT   # reuse the exact same augmentations


# canonical TransUNet paths (override via build_synapse_datasets args)
SYN_TRAIN_NPZ = "./data/Synapse/train_npz"
SYN_TEST_H5   = "./data/Synapse/test_vol_h5"
SYN_LISTS     = "./data/Synapse/lists_Synapse"


def _read_list(list_dir, name):
    """Read lists_Synapse/<name>.txt -> list of stripped entries."""
    path = os.path.join(list_dir, name)
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _zoom_to(a, size, order):
    """Resize (H,W) to (size,size). order=1 image (bilinear), 0 label (nearest)."""
    import scipy.ndimage as ndi
    h, w = a.shape
    if (h, w) == (size, size):
        return a
    return ndi.zoom(a, (size / h, size / w), order=order)


def _norm(img):
    m, s = img.mean(), img.std()
    return (img - m) / (s + 1e-8)


class SynapseSliceDataset(Dataset):
    """Train/val 2D slices, driven by a list file."""

    def __init__(self, npz_dir, list_dir, list_name, size=224, aug=None, seed=0, merge=None):
        self.npz_dir = npz_dir
        self.names = _read_list(list_dir, list_name)
        if not self.names:
            raise FileNotFoundError(f"empty list {list_name} in {list_dir}")
        self.size, self.aug = size, aug
        self.merge = merge
        self.rng = np.random.default_rng(seed)
        # patient id = the caseXXXX prefix; slices from one case are paired
        self.cases = [n.split("_")[0] for n in self.names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        f = os.path.join(self.npz_dir, self.names[i] + ".npz")
        d = np.load(f)
        img = d["image"].astype(np.float32)
        lab = np.rint(d["label"]).astype(np.int64)
        img = np.squeeze(img)
        lab = np.squeeze(lab)
        if img.ndim != 2 or lab.shape != img.shape:
            raise ValueError(f"{f}: img {img.shape}, lab {lab.shape}")
        lab = _apply_merge(lab, self.merge)
        img = _zoom_to(img, self.size, order=1)
        lab = _zoom_to(lab, self.size, order=0)
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, self.rng)
        img = _norm(img)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()),
                self.cases[i])


class SynapseVolumeDataset(Dataset):
    """Test 3D volumes as .h5, one file per case. Returns (N,1,H,W)+(N,H,W)."""

    def __init__(self, h5_dir, list_dir, list_name="test_vol.txt", size=224, merge=None):
        import glob
        self.h5_dir = h5_dir
        self.size = size
        self.merge = merge
        try:
            names = _read_list(list_dir, list_name)
        except FileNotFoundError:
            names = None
        # resolve each name to an actual file; TransUNet uses '<case>.npy.h5'
        self.files = []
        self.cases = []
        if names:
            for n in names:
                cand = [os.path.join(h5_dir, n + ext)
                        for ext in (".npy.h5", ".h5", "")]
                hit = next((c for c in cand if os.path.exists(c)), None)
                if hit is None:  # last resort: prefix match
                    g = glob.glob(os.path.join(h5_dir, n + "*"))
                    hit = g[0] if g else None
                if hit:
                    self.files.append(hit); self.cases.append(n)
        if not self.files:  # fall back to globbing the dir
            for hit in sorted(glob.glob(os.path.join(h5_dir, "*.h5"))):
                self.files.append(hit)
                self.cases.append(os.path.basename(hit).split(".")[0])
        if not self.files:
            raise FileNotFoundError(f"no test volumes under {h5_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import h5py
        with h5py.File(self.files[i], "r") as f:
            img = f["image"][:].astype(np.float32)     # (N,H,W)
            lab = np.rint(f["label"][:]).astype(np.int64)
        lab = _apply_merge(lab, self.merge)
        if img.ndim == 2:
            img, lab = img[None], lab[None]
        ims, lbs = [], []
        for n in range(img.shape[0]):
            im = _zoom_to(img[n], self.size, order=1)
            lb = _zoom_to(lab[n], self.size, order=0)
            ims.append(_norm(im))
            lbs.append(lb)
        vol = np.stack(ims)[:, None]                    # (N,1,H,W)
        msk = np.stack(lbs)                             # (N,H,W)
        return torch.from_numpy(vol), torch.from_numpy(msk), self.cases[i]


def build_synapse_datasets(root=None, size=224, aug_regime="full", seed=0,
                           n_classes=9,
                           train_npz=SYN_TRAIN_NPZ, test_h5=SYN_TEST_H5,
                           lists=SYN_LISTS, train_frac=1.0):
    """Same return shape as data.build_datasets: (train, val, test).

    Synapse has no official val split in the TransUNet lists; if val.txt is
    absent we carve a deterministic 10% of TRAIN cases (by patient, never by
    slice) as val, so LR selection never touches test.

    CLASS-COUNT ABLATION: n_classes in {9,6,4,2} selects a label-merge scheme
    (see _MERGE_MAPS). n_classes=9 is the identity control. The IMAGES are
    identical across all four; only label granularity changes -- this is the
    controlled test of whether Muon's benefit tracks class count.

    DIFFICULTY-AT-FIXED-CLASS-COUNT (train_frac < 1.0): after the val carve-out,
    subsample the remaining TRAIN cases at the PATIENT level to train_frac of
    them, with a FIXED seed (not per-run), holding classes, images, val, and
    test fixed. Lower train_frac induces a lower attainable baseline (harder
    task) without changing class count or the evaluation set -- the controlled
    difficulty knob. val and test are UNCHANGED across all fractions.
    """
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    if n_classes not in _MERGE_MAPS:
        raise ValueError(f"Synapse n_classes must be one of {sorted(_MERGE_MAPS)} "
                         f"(9=full, 6/4/2=merged ablation); got {n_classes}")
    merge = n_classes           # 9 is identity

    train_names = _read_list(lists, "train.txt")
    have_val = os.path.exists(os.path.join(lists, "val.txt"))

    if have_val:
        tr_names = _read_list(lists, "train.txt")
        va = SynapseSliceDataset(train_npz, lists, "val.txt", size, None, seed, merge=merge)
        _use_from_names = False
    else:
        # patient-level split of train into train/val (10% of cases to val)
        cases = sorted({n.split("_")[0] for n in train_names})
        rng = np.random.default_rng(0)                 # fixed, not per-seed
        val_cases = set(rng.choice(cases, max(1, len(cases) // 10), replace=False))
        tr_names = [n for n in train_names if n.split("_")[0] not in val_cases]
        va_names = [n for n in train_names if n.split("_")[0] in val_cases]
        va = _SynapseSliceFromNames(train_npz, va_names, size, None, seed, merge=merge)
        _use_from_names = True

    # DIFFICULTY KNOB: subsample training cases (patient-level, fixed seed).
    if train_frac < 1.0:
        tr_cases = sorted({n.split("_")[0] for n in tr_names})
        frng = np.random.default_rng(12345)            # FIXED across runs/seeds
        n_keep = max(1, int(round(len(tr_cases) * train_frac)))
        keep = set(frng.choice(tr_cases, n_keep, replace=False))
        tr_names = [n for n in tr_names if n.split("_")[0] in keep]

    if have_val and not _use_from_names and train_frac >= 1.0:
        tr = SynapseSliceDataset(train_npz, lists, "train.txt", size, cfg, seed, merge=merge)
    else:
        tr = _SynapseSliceFromNames(train_npz, tr_names, size, cfg, seed, merge=merge)

    te = SynapseVolumeDataset(test_h5, lists, "test_vol.txt", size, merge=merge)
    return tr, va, te


class _SynapseSliceFromNames(SynapseSliceDataset):
    """Same as SynapseSliceDataset but takes an explicit name list (for the
    carved val split) instead of reading a list file."""
    def __init__(self, npz_dir, names, size=224, aug=None, seed=0, merge=None):
        self.npz_dir = npz_dir
        self.names = names
        self.size, self.aug = size, aug
        self.merge = merge
        self.rng = np.random.default_rng(seed)
        self.cases = [n.split("_")[0] for n in names]


# ==================================================================
# CLASS-COUNT ABLATION: merge the 9-class labels into coarser schemes
# on the SAME images, to test whether Muon's benefit tracks class count
# causally (holding dataset/modality/images fixed).
#
# Original Synapse ids: 0 bg, 1 spleen, 2 R-kidney, 3 L-kidney,
#   4 gallbladder, 5 liver, 6 stomach, 7 aorta, 8 pancreas.
#
# Merges are anatomically/functionally grouped (not arbitrary) so a
# reviewer can't attribute a change to "which organs you dropped":
#
#   merge9 (control): identity -> 9 classes (0..8)
#   merge6: pair bilateral/adjacent organs ->  6 classes
#       0 bg
#       1 spleen
#       2 kidneys (L+R merged)          <- 2,3 -> 2
#       3 gallbladder+liver (hepatobiliary)  <- 4,5 -> 3
#       4 stomach                        <- 6 -> 4
#       5 aorta+pancreas (retroperitoneal)   <- 7,8 -> 5
#   merge4: group into 4 functional systems -> 4 classes
#       0 bg
#       1 solid organs (spleen,kidneys,liver)  <- 1,2,3,5 -> 1
#       2 hollow/GI (gallbladder,stomach)      <- 4,6 -> 2
#       3 vascular+pancreas (aorta,pancreas)   <- 7,8 -> 3
#   merge2 (binary): any organ vs background -> 2 classes
#       0 bg, 1 any organ (1..8 -> 1)
#
# The IMAGES and per-pixel foreground are identical across all four; only
# the label granularity changes. n_classes must match: 9/6/4/2.

_MERGE_MAPS = {
    9: {i: i for i in range(9)},                          # identity / control
    6: {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5},
    4: {0: 0, 1: 1, 2: 1, 3: 1, 5: 1, 4: 2, 6: 2, 7: 3, 8: 3},
    2: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
}


def _apply_merge(lab, merge):
    """Remap a 9-class label array to a coarser scheme in-place-safe."""
    if merge is None or merge == 9:
        return lab
    m = _MERGE_MAPS[merge]
    out = np.zeros_like(lab)
    for src, dst in m.items():
        if dst != 0:
            out[lab == src] = dst
    return out
