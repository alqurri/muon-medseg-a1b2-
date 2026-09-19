"""
Study driver. Resumable, because a single A4500 with no SLURM means this runs
for days in a tmux session and WILL be interrupted at some point.

Every run writes results/<hash>.json. Completed runs are skipped on restart.

    python run_study.py --stage 0                      # param split only, ~seconds
    python run_study.py --stage 1 --data-root ./data/ACDC
    python run_study.py --stage 2 --data-root ...
    python run_study.py --stage 3 --data-root ...      # only if stage 2 is non-flat

Stage 1 note: the LR grids are deliberately disjoint. Muon's usable range sits
1-2 orders of magnitude above AdamW's; a shared grid guarantees a rigged
comparison, and that is the single most common flaw in optimizer papers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback

from train import default_cfg, run

ADAMW_LRS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
MUON_LRS = [1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 8e-2, 1.2e-1]

# BraTS-specific grids. On BraTS the AdamW optimum landed at the 3e-3 grid
# EDGE (sweep warned), so AdamW may be under-tuned; we extend it upward to
# recover an interior maximum before any Muon-vs-AdamW comparison. Muon is
# also extended one step up for symmetry (its 8e-2 peak had 1.2e-1 falling
# off, so it is likely already interior, but we confirm). Only BraTS uses
# these; the other five datasets keep the grids above, so their existing
# best_lrs.json / results are unaffected.
ADAMW_LRS_BRATS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
MUON_LRS_BRATS = [1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 8e-2, 1.2e-1, 1.5e-1]


def _grids_for(dataset):
    """Return (adamw_lrs, muon_lrs) for the dataset. BraTS gets extended grids."""
    if dataset.startswith("brats"):
        return ADAMW_LRS_BRATS, MUON_LRS_BRATS
    return ADAMW_LRS, MUON_LRS
MODELS = ["unet", "transunet"]   # default; override with --models
SEEDS = [0, 1, 2, 3, 4]


def key(cfg) -> str:
    ident = {k: cfg[k] for k in
             ("model", "optimizer", "lr", "seed", "aug", "epochs",
              "data_root", "dataset")}
    h = hashlib.md5(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]
    ds = cfg.get("dataset", "acdc")
    return (f"{ds}_{cfg['model']}_{cfg['optimizer']}_lr{cfg['lr']:g}"
            f"_aug-{cfg['aug']}_s{cfg['seed']}_{h}")


def do(cfg, outdir) -> dict | None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, key(cfg) + ".json")
    if os.path.exists(path):
        print(f"[skip] {os.path.basename(path)}")
        return json.load(open(path))
    print(f"\n[run ] {os.path.basename(path)}")
    try:
        res = run(cfg)
    except Exception:
        traceback.print_exc()
        # record the failure so a restart doesn't loop on a config that OOMs
        json.dump({"config": cfg, "failed": True},
                  open(path + ".failed", "w"), indent=2)
        return None
    json.dump(res, open(path, "w"), indent=2)
    return res


def stage0(args):
    """Report the Muon parameter surface area per architecture.

    Run this first. frac_muon is the ceiling on what Muon can do, and the
    asymmetry between the two architectures is the study's hypothesis.
    """
    import torch
    from models import build_model
    from muon import split_params, print_split_report
    for m in MODELS:
        model = build_model(m, 1, args.n_classes)
        _, _, rep = split_params(model)
        print_split_report(rep, title=m, show_layers=args.verbose_layers)
        del model
        torch.cuda.empty_cache()


def stage1(args):
    """LR sweep. One seed, reduced-but-CONVERGED budget.

    Do not truncate to 25% of the schedule to save time: Muon and AdamW have
    differently shaped loss curves, so a truncated sweep systematically
    rewards whichever converges faster early -- which is the very effect the
    study is trying to measure. Cut epochs and let the schedule complete.
    """
    best = {}
    adamw_lrs, muon_lrs = _grids_for(args.dataset)
    for m in MODELS:
        for opt, lrs in (("adamw", adamw_lrs), ("muon", muon_lrs)):
            scores = {}
            for lr in lrs:
                cfg = default_cfg(model=m, optimizer=opt, lr=lr, seed=0,
                                  epochs=args.sweep_epochs, aug=args.aug,
                                  data_root=args.data_root, batch=args.batch,
                                  n_classes=args.n_classes, dataset=args.dataset,
                                  train_frac=args.train_frac, verbose=False)
                r = do(cfg, args.outdir)
                if r:
                    scores[lr] = r["best_val_dice"]
            if scores:
                blr = max(scores, key=scores.get)
                best[f"{m}/{opt}"] = dict(lr=blr, scores=scores)
                print(f"  -> best {m}/{opt}: lr={blr:g} "
                      f"(val dice {scores[blr]:.4f})")
                if blr in (min(scores), max(scores)) and len(scores) > 1:
                    print("     WARNING: best LR is at a grid edge. "
                          "Extend the grid before trusting Stage 2.")
    json.dump(best, open(os.path.join(args.outdir, "best_lrs.json"), "w"), indent=2)
    return best


def _load_best(args):
    p = os.path.join(args.outdir, "best_lrs.json")
    if not os.path.exists(p):
        raise SystemExit("run --stage 1 first (best_lrs.json missing)")
    return json.load(open(p))


def stage2(args):
    """2 optimizers x 2 architectures x 5 seeds, full budget, best LR each."""
    best = _load_best(args)
    for m in MODELS:
        for opt in ("adamw", "muon"):
            k = f"{m}/{opt}"
            if k not in best:
                continue
            for s in SEEDS:
                cfg = default_cfg(model=m, optimizer=opt, lr=best[k]["lr"],
                                  seed=s, epochs=args.epochs, aug=args.aug,
                                  data_root=args.data_root, batch=args.batch,
                                  n_classes=args.n_classes,
                                  dataset=args.dataset,
                                  train_frac=args.train_frac,
                                  adamw_lr_for_hybrid=best[f"{m}/adamw"]["lr"],
                                  verbose=False)
                do(cfg, args.outdir)


def stage3(args):
    """Augmentation-interaction arm. Only worth running if Stage 2 is non-flat."""
    best = _load_best(args)
    m = args.stage3_model
    for aug in ("light", "full"):
        for opt in ("adamw", "muon"):
            k = f"{m}/{opt}"
            for s in SEEDS:
                cfg = default_cfg(model=m, optimizer=opt, lr=best[k]["lr"],
                                  seed=s, epochs=args.epochs, aug=aug,
                                  data_root=args.data_root, batch=args.batch,
                                  n_classes=args.n_classes,
                                  dataset=args.dataset,
                                  train_frac=args.train_frac,
                                  adamw_lr_for_hybrid=best[f"{m}/adamw"]["lr"],
                                  verbose=False)
                do(cfg, args.outdir)



def stage1_light(args):
    """Full-grid LR sweep for the LIGHT aug regime (addresses the reviewer's
    headline-interaction objection). Saves to best_lrs_light.json so the
    light-aug final runs use a light-tuned LR, not the heavy-selected one."""
    best = {}
    adamw_lrs, muon_lrs = _grids_for(args.dataset)
    for m in MODELS:
        for opt, lrs in (("adamw", adamw_lrs), ("muon", muon_lrs)):
            scores = {}
            for lr in lrs:
                cfg = default_cfg(model=m, optimizer=opt, lr=lr, seed=0,
                                  epochs=args.sweep_epochs, aug="light",
                                  data_root=args.data_root, batch=args.batch,
                                  n_classes=args.n_classes, dataset=args.dataset,
                                  train_frac=args.train_frac, verbose=False)
                r = do(cfg, args.outdir)
                if r:
                    scores[lr] = r["best_val_dice"]
            if scores:
                blr = max(scores, key=scores.get)
                best[f"{m}/{opt}"] = dict(lr=blr, scores=scores)
                print(f"  -> LIGHT best {m}/{opt}: lr={blr:g} (val {scores[blr]:.4f})")
                if blr in (min(scores), max(scores)) and len(scores) > 1:
                    print("     WARNING: light best LR at grid edge; extend grid.")
    import json as _j, os as _o
    _j.dump(best, open(_o.path.join(args.outdir, "best_lrs_light.json"), "w"), indent=2)
    return best


def stage3_light(args):
    """Light-aug FINAL runs (5 seeds) at the LIGHT-tuned LR from
    best_lrs_light.json, for the U-Net. Pair with the existing heavy runs to
    recompute the interaction with both regimes fully tuned."""
    import json as _j, os as _o
    p = _o.path.join(args.outdir, "best_lrs_light.json")
    if not _o.path.exists(p):
        raise SystemExit("run --stage 1L first (best_lrs_light.json missing)")
    best = _j.load(open(p))
    m = args.stage3_model
    for opt in ("adamw", "muon"):
        k = f"{m}/{opt}"
        if k not in best:
            continue
        for s in SEEDS:
            cfg = default_cfg(model=m, optimizer=opt, lr=best[k]["lr"],
                              seed=s, epochs=args.epochs, aug="light",
                              data_root=args.data_root, batch=args.batch,
                              n_classes=args.n_classes, dataset=args.dataset,
                              train_frac=args.train_frac,
                              adamw_lr_for_hybrid=best[f"{m}/adamw"]["lr"],
                              verbose=False)
            do(cfg, args.outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, required=True,
                    choices=["0", "1", "2", "3", "1L", "3L"])
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to run, e.g. --models vmunet (default: unet transunet)")
    ap.add_argument("--data-root", default=None,
                    help="omit to run on synthetic data (smoke test)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--sweep-epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--dataset", default="acdc", choices=["acdc", "synapse", "isic", "camus", "flare", "flare_npz", "brats", "brats_npz", "amos", "amos_npz"])
    ap.add_argument("--aug", default="full")
    ap.add_argument("--stage3-model", default="transunet")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="subsample Synapse training cases (difficulty knob); 1.0=full")
    ap.add_argument("--verbose-layers", action="store_true")
    a = ap.parse_args()
    if a.models:
        globals()["MODELS"] = a.models
    {"0": stage0, "1": stage1, "2": stage2, "3": stage3, "1L": stage1_light, "3L": stage3_light}[a.stage](a)
