"""
ISIC 2016 (skin-lesion) loaders.

Structurally different from ACDC/Synapse in three ways that matter:

  1. NATURAL 2D IMAGES, no volumes. Test scoring is PER-IMAGE, not
     per-volume. Each image is an independent sample; there is no patient
     grouping in ISIC 2016, so pairing between optimizers is per-image and
     n = number of test images (~379 for Part 1 test).
  2. RGB input (3 channels), vs 1-channel MRI/CT. Models must be built with
     in_ch=3 for ISIC.
  3. BINARY segmentation: background + lesion => n_classes=2. The mask is a
     0/255 PNG; we threshold to {0,1}.

Splits come from the official CSVs, exactly like the user's ISICDataset:
    ISBI2016_ISIC_Part1_Training_GroundTruth.csv
    ISBI2016_ISIC_Part1_Test_GroundTruth.csv
each row: [_, image_relpath, mask_relpath]. Paths resolve under data_path.

There is no official val split, so (as with Synapse) we carve a deterministic
10% of TRAINING images as validation for learning-rate selection, keeping
test untouched. ISIC images are independent, so an image-level split is fine
here (no patient leakage concern).

build_isic_datasets() returns (train, val, test) with the SAME item contract
as the other loaders: (image_tensor[C,H,W], mask_tensor[H,W]long, id_str).
The test set is a plain per-image dataset (NOT a volume dataset), so
train.py scores it with volume=False.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data import augment, AUG_FULL, AUG_LIGHT   # reuse the same augmentations

ISIC_ROOT = "./data/ISIC2018"
_CSV = "ISBI2016_ISIC_Part1_{mode}_GroundTruth.csv"


def _read_csv(data_path, mode):
    # The ISIC CSV HAS a header row: ",img,seg". Let pandas consume it
    # (do NOT pass header=None, or the header is read as a data row and the
    # first "image path" becomes the literal string "img").
    df = pd.read_csv(os.path.join(data_path, _CSV.format(mode=mode)),
                     encoding="gbk")
    imgs = df.iloc[:, 1].astype(str).tolist()   # 'img' column
    msks = df.iloc[:, 2].astype(str).tolist()   # 'seg' column
    return imgs, msks


def _load_pair(data_path, img_rel, msk_rel, size):
    img = Image.open(os.path.join(data_path, img_rel)).convert("RGB")
    msk = Image.open(os.path.join(data_path, msk_rel)).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    msk = msk.resize((size, size), Image.NEAREST)
    img = np.asarray(img, dtype=np.float32) / 255.0            # (H,W,3) in [0,1]
    msk = (np.asarray(msk, dtype=np.float32) > 127).astype(np.int64)  # (H,W) {0,1}
    return img, msk


class ISICSliceDataset(Dataset):
    """Per-image ISIC dataset. `names` optionally overrides the CSV list
    (used to carve train/val)."""

    def __init__(self, data_path, mode, size=224, aug=None, seed=0, names=None):
        imgs, msks = _read_csv(data_path, mode)
        if names is not None:
            keep = set(names)
            pairs = [(i, m) for i, m in zip(imgs, msks) if i in keep]
            imgs, msks = zip(*pairs) if pairs else ([], [])
        self.data_path = data_path
        self.imgs, self.msks = list(imgs), list(msks)
        if not self.imgs:
            raise FileNotFoundError(f"no ISIC images for mode={mode}")
        self.size, self.aug = size, aug
        self.rng = np.random.default_rng(seed)
        # id = image basename without extension; each image is its own "case"
        self.cases = [os.path.splitext(os.path.basename(i))[0] for i in self.imgs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, msk = _load_pair(self.data_path, self.imgs[i], self.msks[i], self.size)

        if self.aug is not None:
            # augment() expects a single-channel (H,W) image; apply the SAME
            # geometric transform to each RGB channel with a shared RNG state
            # so the three channels and the mask stay aligned.
            st = self.rng.bit_generator.state
            chans = []
            for c in range(3):
                self.rng.bit_generator.state = st          # reset per channel
                ic, mc = augment(img[..., c], msk, self.aug, self.rng)
                chans.append(ic)
            img = np.stack(chans, axis=-1)                  # (H,W,3)
            msk = mc                                         # mask from last (identical) pass

        # per-image z-score per channel
        x = img.transpose(2, 0, 1).copy()                   # (3,H,W)
        for c in range(3):
            m, s = x[c].mean(), x[c].std()
            x[c] = (x[c] - m) / (s + 1e-8)
        return (torch.from_numpy(x),
                torch.from_numpy(msk.copy()),
                self.cases[i])


def build_isic_datasets(root=None, size=224, aug_regime="full", seed=0,
                        n_classes=2, data_path=None):
    """Same (train, val, test) contract as build_datasets.

    `root` (from train.py --data-root) takes precedence; falls back to
    data_path, then the module default ISIC_ROOT.
    """
    data_path = root or data_path or ISIC_ROOT
    cfg = AUG_FULL if aug_regime == "full" else AUG_LIGHT

    train_imgs, _ = _read_csv(data_path, "Training")
    rng = np.random.default_rng(0)                          # fixed, not per-seed
    n_val = max(1, len(train_imgs) // 10)
    val_names = set(rng.choice(train_imgs, n_val, replace=False).tolist())
    tr_names = [i for i in train_imgs if i not in val_names]

    tr = ISICSliceDataset(data_path, "Training", size, cfg, seed, names=tr_names)
    va = ISICSliceDataset(data_path, "Training", size, None, seed, names=list(val_names))
    te = ISICSliceDataset(data_path, "Test", size, None, seed)
    return tr, va, te
