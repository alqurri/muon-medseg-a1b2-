"""
CAMUS (cardiac ultrasound / echocardiography) loader.

Layout (from the user's easy_Synapse_dataset example):
  ./data/CAMUS/training/<patient>/<patient>_2CH_<side>.mhd       (image)
  ./data/CAMUS/training/<patient>/<patient>_2CH_<side>_gt.mhd    (mask)
  ./data/CAMUS/testing/...   same structure
  side in {ED, ES}

Decisions (confirmed with the user):
  * 4 classes (bg + LV endo + LV epi/myo + LA).
  * 2CH view only. ED and ES are BOTH used, each patient contributing two
    independent samples (one per side).
  * /training vs /testing is the train/test split; we carve a patient-level
    10% val split from training (both sides of a held-out patient go to val,
    so no patient leaks across train/val).

Why CAMUS matters for the study: it is a THIRD cardiac dataset but a NEW
modality (ultrasound: speckle, low contrast, poor boundaries). It segments
the same 4 cardiac structures as ACDC but is substantially harder, so it is
expected to land BETWEEN ACDC (easy MRI) and Synapse (hard CT) on the
difficulty axis -- filling the middle of the gradient.

Test scoring is PER-IMAGE (each 2CH ED/ES frame is an independent 2D image;
CAMUS test has no 3D volume in this layout), like ISIC. Pairing is per image.

.mhd is MetaImage; read with SimpleITK. The array comes back (1,H,W) or
(H,W); we squeeze to (H,W). Masks resized with nearest-neighbor to keep
class ids exact.
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

from data import augment, AUG_FULL, AUG_LIGHT

CAMUS_TRAIN = "./data/CAMUS/training"
CAMUS_TEST = "./data/CAMUS/testing"
_SIDES = ("ED", "ES")
_VIEW = "2CH"


def _read_mhd(path):
    """Read a .mhd into a 2D float array (H,W)."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))   # (Z,H,W) or (H,W)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        # some CAMUS frames come as (1,H,W); squeeze handles it, but guard
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr


def _list_patients(root):
    return sorted(next(os.walk(root))[1])


def _sample_id(patient, side):
    return f"{patient}_{_VIEW}_{side}"


class CAMUSDataset(Dataset):
    """One item per (patient, side). image (1,H,W) float, mask (H,W) long."""

    def __init__(self, root, patients, size=224, aug=None, seed=0):
        self.root = root
        self.size, self.aug = size, aug
        self.rng = np.random.default_rng(seed)
        # each patient contributes both sides
        self.items = [(p, s) for p in patients for s in _SIDES]
        # keep only items whose files actually exist
        self.items = [(p, s) for (p, s) in self.items
                      if os.path.exists(self._img_path(p, s))
                      and os.path.exists(self._msk_path(p, s))]
        if not self.items:
            raise FileNotFoundError(f"no CAMUS 2CH samples under {root}")
        # per-patient id used for grouping (patient-level, not slice)
        self.cases = [_sample_id(p, s) for (p, s) in self.items]
        self.patients = [p for (p, s) in self.items]

    def _img_path(self, p, s):
        return os.path.join(self.root, p, f"{p}_{_VIEW}_{s}.mhd")

    def _msk_path(self, p, s):
        return os.path.join(self.root, p, f"{p}_{_VIEW}_{s}_gt.mhd")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, s = self.items[i]
        img = _read_mhd(self._img_path(p, s)).astype(np.float32)
        msk = _read_mhd(self._msk_path(p, s))
        msk = np.rint(msk).astype(np.int64)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk.astype(np.float32), (self.size, self.size),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)

        if self.aug is not None:
            img, msk = augment(img, msk, self.aug, self.rng)

        m, sd = img.mean(), img.std()
        img = (img - m) / (sd + 1e-8)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(msk.copy()),
                self.cases[i])


def build_camus_datasets(root=None, size=224, aug_regime="full", seed=0,
                         n_classes=4, train_root=None, test_root=None):
    """Same (train, val, test) contract as build_datasets.

    `root`, if given (from --data-root), is treated as the PARENT of
    training/ and testing/; otherwise the module defaults are used.
    Val is a patient-level 10% carve from training (both sides of a
    held-out patient go to val -> no patient leakage).
    """
    if root:
        train_root = os.path.join(root, "training")
        test_root = os.path.join(root, "testing")
    train_root = train_root or CAMUS_TRAIN
    test_root = test_root or CAMUS_TEST
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT

    train_patients = _list_patients(train_root)
    rng = np.random.default_rng(0)                       # fixed, not per-seed
    n_val = max(1, len(train_patients) // 10)
    val_pat = set(rng.choice(train_patients, n_val, replace=False).tolist())
    tr_pat = [p for p in train_patients if p not in val_pat]

    tr = CAMUSDataset(train_root, tr_pat, size, cfg, seed)
    va = CAMUSDataset(train_root, sorted(val_pat), size, None, seed)
    te = CAMUSDataset(test_root, _list_patients(test_root), size, None, seed)
    return tr, va, te
