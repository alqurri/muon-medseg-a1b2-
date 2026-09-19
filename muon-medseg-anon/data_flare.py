"""
FLARE22 (MICCAI FLARE 2022) abdominal multi-organ CT loader.

Raw 3D NIfTI (.nii.gz). 13 foreground organs (+ background) => n_classes=14.
One folder on disk, so we define a deterministic patient-level split.

Layout assumed (standard FLARE22 naming; override in build_flare_datasets):
    <root>/images/FLARE22_Tr_XXXX_0000.nii.gz     (CT, '_0000' = modality 0)
    <root>/labels/FLARE22_Tr_XXXX.nii.gz          (label, no '_0000')
If your layout differs (single dir, different suffixes), adjust IMG_GLOB /
_label_for below -- the check script output tells us which.

Handling that FLARE22 needs and the others did not:
  * CT WINDOWING. Abdominal CT spans ~[-1000, +1000] HU; organs live in a
    narrow soft-tissue window. Without clipping, per-image z-score is
    dominated by air/bone and contrast is unusable. We clip to a soft-tissue
    window [-, +] then normalize. This matters a lot for Dice.
  * 3D volumes are large and anisotropic. Train/val use axial slices that
    contain foreground (empty slices above/below the abdomen are skipped, or
    training wastes most steps on all-background slices). Test keeps whole
    volumes, scored per-volume in 3D like ACDC/Synapse.
  * 13 organs with extreme imbalance (aorta/IVC large; adrenals tiny) => the
    empty-mask NaN handling in metrics.py does heavy lifting; per-organ Dice
    variance is high. n_classes=14.

Slicing every axial plane of every volume is a lot of samples; to keep the
study tractable on one GPU we subsample foreground slices (stride SLICE_STRIDE)
and cap slices per volume. Tune via build args if needed.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset

from data import augment, AUG_FULL, AUG_LIGHT

FLARE_ROOT = "./data/FLARE22"

# soft-tissue CT window (HU). FLARE22 organs are abdominal soft tissue.
HU_MIN, HU_MAX = -125.0, 275.0

SLICE_STRIDE = 2          # keep every 2nd foreground slice for train/val
MAX_SLICES_PER_VOL = 60   # cap, to bound dataset size


def _load_nii(path):
    """Return (H,W,D) float array in canonical axial orientation."""
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    return arr


def _window_ct(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)      # -> [0,1]
    return img


def _resize2d(a, size, order):
    import cv2
    interp = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
    return cv2.resize(a, (size, size), interpolation=interp)


def _norm(x):
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-8)


def _case_id(path):
    b = os.path.basename(path)
    for suf in ("_0000.nii.gz", ".nii.gz"):
        if b.endswith(suf):
            b = b[: -len(suf)]
            break
    return b


class FlareSliceDataset(Dataset):
    """Axial foreground slices from a set of volumes (train/val)."""

    def __init__(self, pairs, size=224, aug=None, seed=0,
                 stride=SLICE_STRIDE, max_slices=MAX_SLICES_PER_VOL):
        # pairs: list of (image_path, label_path)
        self.size, self.aug = size, aug
        self.rng = np.random.default_rng(seed)
        self.index = []   # (img_path, lab_path, z, case)
        for ip, lp in pairs:
            lab = _load_nii(lp)
            fg_z = np.where((lab > 0).any(axis=(0, 1)))[0]   # slices with organ
            if len(fg_z) == 0:
                continue
            fg_z = fg_z[::stride]
            if len(fg_z) > max_slices:
                fg_z = self.rng.choice(fg_z, max_slices, replace=False)
            cid = _case_id(ip)
            for z in fg_z:
                self.index.append((ip, lp, int(z), cid))
        if not self.index:
            raise FileNotFoundError("no foreground slices found for FLARE split")
        # simple cache so we don't reload a whole volume per slice
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _vol(self, path, is_label):
        key = (path, is_label)
        if key not in self._cache:
            self._cache = {}      # keep only one volume in memory at a time
            self._cache[key] = _load_nii(path)
        return self._cache[key]

    def __getitem__(self, i):
        ip, lp, z, cid = self.index[i]
        img = self._vol(ip, False)[:, :, z]
        lab = np.rint(self._vol(lp, True)[:, :, z]).astype(np.int64)
        img = _window_ct(img)
        img = _resize2d(img, self.size, 1)
        lab = _resize2d(lab.astype(np.float32), self.size, 0).astype(np.int64)
        if self.aug is not None:
            img, lab = augment(img, lab, self.aug, self.rng)
        img = _norm(img)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(lab.copy()),
                cid)


class FlareVolumeDataset(Dataset):
    """Whole volumes for test, scored per-volume in 3D."""

    def __init__(self, pairs, size=224):
        self.pairs = pairs
        self.size = size
        self.cases = [_case_id(ip) for ip, lp in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = _load_nii(ip)                     # (H,W,D)
        lab = np.rint(_load_nii(lp)).astype(np.int64)
        # move depth to front -> (D,H,W), window + resize each slice
        img = np.transpose(img, (2, 0, 1))
        lab = np.transpose(lab, (2, 0, 1))
        ims, lbs = [], []
        for z in range(img.shape[0]):
            im = _window_ct(img[z])
            ims.append(_norm(_resize2d(im, self.size, 1)))
            lbs.append(_resize2d(lab[z].astype(np.float32), self.size, 0).astype(np.int64))
        vol = np.stack(ims)[:, None]            # (D,1,H,W)
        msk = np.stack(lbs)                     # (D,H,W)
        return torch.from_numpy(vol), torch.from_numpy(msk), self.cases[i]


# ------------------------------------------------------------------ pairing
IMG_GLOB = ("images/*_0000.nii.gz", "imagesTr/*_0000.nii.gz",
            "*_0000.nii.gz", "images/*.nii.gz")


def _find_pairs(root):
    """Return [(image_path, label_path), ...] resolving FLARE22 naming."""
    imgs = []
    for pat in IMG_GLOB:
        imgs = sorted(glob.glob(os.path.join(root, pat)))
        if imgs:
            break
    if not imgs:
        raise FileNotFoundError(f"no FLARE images under {root} (tried {IMG_GLOB})")

    pairs = []
    for ip in imgs:
        cid = _case_id(ip)
        cand = [
            os.path.join(root, "labels", cid + ".nii.gz"),
            os.path.join(root, "labelsTr", cid + ".nii.gz"),
            ip.replace("_0000.nii.gz", ".nii.gz").replace("images", "labels"),
            ip.replace("_0000.nii.gz", ".nii.gz"),
        ]
        lp = next((c for c in cand if os.path.exists(c)), None)
        if lp:
            pairs.append((ip, lp))
    if not pairs:
        raise FileNotFoundError(
            f"found {len(imgs)} images but could not match labels; "
            f"check labels dir / naming under {root}")
    return pairs


def build_flare_datasets(root=None, size=224, aug_regime="full", seed=0,
                         n_classes=14, data_path=None,
                         val_frac=0.1, test_frac=0.2):
    """(train, val, test) with the same contract as build_datasets.

    One folder -> deterministic patient-level split (fixed seed, not per-run
    seed, so the split is identical across all runs). Test volumes scored in
    3D per-volume; train/val are foreground axial slices.
    """
    root = root or data_path or FLARE_ROOT
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT

    pairs = _find_pairs(root)
    rng = np.random.default_rng(0)                     # FIXED split seed
    idx = rng.permutation(len(pairs))
    n_test = max(1, int(len(pairs) * test_frac))
    n_val = max(1, int(len(pairs) * val_frac))
    test_i = set(idx[:n_test].tolist())
    val_i = set(idx[n_test:n_test + n_val].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in test_i and i not in val_i]
    val_pairs = [pairs[i] for i in val_i]
    test_pairs = [pairs[i] for i in test_i]

    tr = FlareSliceDataset(train_pairs, size, cfg, seed)
    va = FlareSliceDataset(val_pairs, size, None, seed)
    te = FlareVolumeDataset(test_pairs, size)
    return tr, va, te


# ==================================================================
# FAST PATH: pre-extracted .npy slices (see preprocess_flare.py)
# ==================================================================
import json as _json


class FlareNpzSliceDataset(Dataset):
    """Reads pre-extracted uncompressed .npy slices (stacked [image,label]).

    Each .npy is (2, H, W): channel 0 windowed image, channel 1 label. Resize
    happens here (kept out of preprocessing so size stays flexible). This is
    the fast training path -- no NIfTI decompression per step.
    """

    def __init__(self, root, entries, size=224, aug=None, seed=0):
        self.root = root
        self.entries = entries          # list of {"file","case"}
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
                torch.from_numpy(lab.copy()),
                self.cases[i])


def build_flare_npz_datasets(root=None, size=224, aug_regime="full", seed=0,
                             n_classes=14, data_path=None):
    """(train, val, test) from a preprocessed FLARE22_npz dir (manifest.json).

    Same contract/splits as build_flare_datasets, but train/val read fast .npy
    slices. Test reuses FlareVolumeDataset on the manifest's NIfTI test volumes
    (test is loaded once at eval, so no need to preprocess it).
    """
    root = root or data_path or (FLARE_ROOT + "_npz")
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT
    with open(os.path.join(root, "manifest.json")) as f:
        man = _json.load(f)
    tr = FlareNpzSliceDataset(root, man["train"], size, cfg, seed)
    va = FlareNpzSliceDataset(root, man["val"], size, None, seed)
    test_pairs = [(t["image"], t["label"]) for t in man["test_vols"]]
    te = FlareVolumeDataset(test_pairs, size)
    return tr, va, te
