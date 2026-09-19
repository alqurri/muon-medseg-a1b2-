"""
One run = one (model, optimizer, lr, aug_regime, seed) configuration.

Writes a JSON with per-case test metrics so analyze.py never needs to retrain.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import build_datasets
from metrics import evaluate_slice, evaluate_volume, aggregate_by_case
from models import build_model
from muon import build_optimizers


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # benchmark=True picks different algorithms run to run; with only 5 seeds
    # you cannot afford extra variance you didn't choose.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dice_ce_loss(logits, target, n_classes, eps=1.0):
    ce = F.cross_entropy(logits, target)
    p = F.softmax(logits, 1)
    t = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = 1 - ((2 * inter + eps) / (denom + eps)).mean()
    return ce + dice


def _forward_pred(model, x, device, amp):
    x = x.to(device, non_blocking=True)
    with torch.autocast("cuda", torch.bfloat16, enabled=amp and device == "cuda"):
        logits = model(x)
    return logits.float().argmax(1).cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, n_classes, device, amp=True, volume=False):
    """
    volume=False : each item is a batch of independent 2D slices (val).
                   aggregate_by_case then groups slices sharing a patient id.
    volume=True  : each item is ONE (1,N,1,H,W) volume + (1,N,H,W) mask (test).
                   metrics are computed once per volume over the whole stack,
                   which is how ACDC is scored. No per-slice pooling.

    The two paths must produce the same dict shape so run() and analyze.py
    don't care which was used.
    """
    model.eval()
    rows, fails = [], 0

    if not volume:
        for x, y, cid in loader:
            pred = _forward_pred(model, x, device, amp)
            gt = y.numpy()
            for b in range(pred.shape[0]):
                d, h, f = evaluate_slice(pred[b], gt[b], n_classes)
                rows.append((cid[b], d, h))
                fails += f
        cases, dice, hd = aggregate_by_case(rows, n_classes)
    else:
        # loader yields one volume at a time (batch_size=1). Shapes:
        #   x: (1, N, 1, H, W)   y: (1, N, H, W)   cid: [str]
        per_case_d, per_case_h = [], []
        cases = []
        for x, y, cid in loader:
            x, y = x[0], y[0]                       # drop the batch dim -> (N,1,H,W),(N,H,W)
            preds = _forward_pred(model, x, device, amp)   # (N,H,W)
            gt = y.numpy()                          # (N,H,W)
            # score the whole volume at once per class, not slice by slice
            d, h, f = evaluate_volume(preds, gt, n_classes)
            fails += f
            cases.append(cid[0])
            per_case_d.append(d)
            per_case_h.append(h)
        dice = np.stack(per_case_d) if per_case_d else np.zeros((0, n_classes - 1))
        hd = np.stack(per_case_h) if per_case_h else np.zeros((0, n_classes - 1))

    with np.errstate(invalid="ignore"):
        return dict(cases=cases,
                    dice_per_case=dice.tolist(), hd95_per_case=hd.tolist(),
                    dice=float(np.nanmean(dice)) if dice.size else float("nan"),
                    hd95=float(np.nanmean(hd)) if hd.size else float("nan"),
                    empty_failures=int(fails))


def run(cfg: dict) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    ds = cfg.get("dataset", "acdc")
    if ds == "synapse":
        from data_synapse import build_synapse_datasets, SynapseVolumeDataset
        tr, va, te = build_synapse_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"],
            train_frac=cfg.get("train_frac", 1.0))
        volume_types = (SynapseVolumeDataset,)
        in_ch = 1
    elif ds == "isic":
        from data_isic import build_isic_datasets
        tr, va, te = build_isic_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = ()            # ISIC test is per-image, not volumetric
        in_ch = 3                    # RGB dermoscopy
    elif ds == "camus":
        from data_camus import build_camus_datasets
        tr, va, te = build_camus_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = ()            # CAMUS 2CH frames scored per-image
        in_ch = 1                    # grayscale ultrasound
    elif ds == "flare":
        from data_flare import build_flare_datasets, FlareVolumeDataset
        tr, va, te = build_flare_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = (FlareVolumeDataset,)   # 3D CT volumes, scored per-volume
        in_ch = 1                    # CT (windowed grayscale)
    elif ds == "flare_npz":
        from data_flare import build_flare_npz_datasets, FlareVolumeDataset
        tr, va, te = build_flare_npz_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = (FlareVolumeDataset,)   # test still per-volume 3D
        in_ch = 1
    elif ds == "brats":
        from data_brats import build_brats_datasets, BratsVolumeDataset
        tr, va, te = build_brats_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = (BratsVolumeDataset,)   # 3D MRI volumes, per-volume 3D scoring
        in_ch = 1                    # single FLAIR modality
    elif ds == "brats_npz":
        from data_brats import build_brats_npz_datasets, BratsVolumeDataset
        tr, va, te = build_brats_npz_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = (BratsVolumeDataset,)
        in_ch = 1
    elif ds == "amos":
        from data_amos import build_amos_datasets, AmosVolumeDataset
        tr, va, te = build_amos_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = (AmosVolumeDataset,)   # 3D MRI volumes, per-volume 3D scoring
        in_ch = 1                    # single-channel MRI
    elif ds == "amos_npz":
        from data_amos import build_amos_npz_datasets, AmosVolumeDataset
        tr, va, te = build_amos_npz_datasets(
            cfg["data_root"], cfg["size"], cfg["aug"], cfg["seed"], cfg["n_classes"])
        volume_types = (AmosVolumeDataset,)
        in_ch = 1
    else:
        tr, va, te = build_datasets(cfg["data_root"], cfg["size"],
                                    cfg["aug"], cfg["seed"], cfg["n_classes"])
        from data import VolumeDataset
        volume_types = (VolumeDataset,)
        in_ch = 1
    g = torch.Generator().manual_seed(cfg["seed"])
    dl = dict(num_workers=cfg["workers"], pin_memory=(dev == "cuda"))
    tl = DataLoader(tr, cfg["batch"], shuffle=True, drop_last=True, generator=g, **dl)
    vl = DataLoader(va, cfg["batch"], shuffle=False, **dl)
    # test items are whole volumes of differing depth N -> batch must be 1
    test_is_volume = bool(volume_types) and isinstance(te, volume_types)
    el = DataLoader(te, 1 if test_is_volume else cfg["batch"], shuffle=False, **dl)

    model = build_model(cfg["model"], in_ch, cfg["n_classes"]).to(dev)
    opts, report = build_optimizers(model, cfg["optimizer"], cfg["lr"],
                                    cfg["weight_decay"],
                                    cfg.get("adamw_lr_for_hybrid"),
                                    verbose=cfg.get("verbose", True))
    total = cfg["epochs"] * max(1, len(tl))
    scheds = [torch.optim.lr_scheduler.OneCycleLR(
        o, max_lr=[pg["lr"] for pg in o.param_groups], total_steps=total,
        pct_start=0.05, anneal_strategy="cos") for o in opts]

    best, best_state, hist = -1.0, None, []
    t0, step_times = time.time(), []
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y, _ in tl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            ts = time.time()
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                loss = dice_ce_loss(model(x), y, cfg["n_classes"])
            loss.backward()
            if cfg["clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            for s in scheds:
                s.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - ts)

        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            m = evaluate(model, vl, cfg["n_classes"], dev)
            hist.append(dict(epoch=ep + 1, val_dice=m["dice"], val_hd95=m["hd95"]))
            if m["dice"] > best:
                best = m["dice"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  ep {ep+1:>4}/{cfg['epochs']}  val_dice {m['dice']:.4f} "
                  f"hd95 {m['hd95']:.2f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, el, cfg["n_classes"], dev, volume=test_is_volume)

    return dict(config=cfg, history=hist, best_val_dice=best, test=test,
                frac_muon=report["frac_muon"],
                n_muon=report["n_muon"], n_adamw=report["n_adamw"],
                # wall-clock matters: Newton-Schulz is ~15 extra matmuls per
                # matrix per step. An epoch-matched win can be a time loss.
                sec_per_step=float(np.median(step_times)),
                total_minutes=(time.time() - t0) / 60)


def default_cfg(**kw):
    c = dict(model="unet", optimizer="adamw", lr=3e-4, weight_decay=1e-2,
             seed=0, epochs=200, batch=16, size=224, n_classes=4,
             aug="full", data_root=None, workers=4, clip=1.0,
             eval_every=5, adamw_lr_for_hybrid=3e-4, dataset="acdc", verbose=True)
    c.update(kw)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in default_cfg().items():
        if k == "data_root":
            ap.add_argument("--data-root", default=None)
        else:
            ap.add_argument(f"--{k.replace('_','-')}",
                            type=type(v) if v is not None else str, default=v)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = default_cfg(**{k: v for k, v in vars(a).items() if k != "out"})
    res = run(cfg)
    print(json.dumps({k: res[k] for k in
                      ("best_val_dice", "frac_muon", "sec_per_step", "total_minutes")},
                     indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
