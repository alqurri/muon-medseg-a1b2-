"""
verify_interaction.py -- clean interaction check with dedup + LR-filter + ED/ES clustering.

Motivated by BraTS: results dirs can hold TWO generations of light-aug runs
(the original reduced-grid runs and the full-grid light-tune runs), sometimes
duplicated at the same seed. Pooling them (as analyze_v2 does without an
LR filter) contaminates the interaction. This script:

  1. filters to the SELECTED learning rate for each optimizer x aug regime,
     reading heavy LRs from best_lrs.json and light LRs from best_lrs_light.json;
  2. deduplicates by (seed) keeping the NEWEST file (the full-grid re-tune);
  3. clusters ED/ES frames to one measurement per patient for ACDC and CAMUS;
  4. computes the per-seed interaction (heavy - light difference-of-differences)
     and a hierarchical bootstrap 95% CI over seeds and patients.

Usage (per dataset):
  python verify_interaction.py --dir results_isic  --dataset isic
  python verify_interaction.py --dir results        --dataset acdc
  python verify_interaction.py --dir results_camus  --dataset camus

If best_lrs_light.json is absent, pass --light-muon-lr / --light-adamw-lr
explicitly, or the script falls back to the heavy-selected LR (and warns).
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np


def patient_of(cid, dataset):
    c = str(cid)
    if dataset in ("acdc", "camus"):
        c = re.sub(r"[_\-](ed|es|ED|ES)$", "", c)
        c = re.sub(r"[_\-]?frame\d+$", "", c, flags=re.I)
        c = re.sub(r"[_\-](0|1)$", "", c)
    return c


def load(dir, opt, aug, lr, dataset):
    """seed -> {patient: mean Dice}, newest file per seed, filtered to lr."""
    best = {}
    for f in glob.glob(os.path.join(dir, "*.json")):
        b = os.path.basename(f)
        if "noortho" in b or b in ("best_lrs.json", "best_lrs_light.json", "summary.json"):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("model") != "unet" or c.get("optimizer") != opt or c.get("aug") != aug:
            continue
        if abs(c.get("lr", -1) - lr) > 1e-9:
            continue
        if c.get("epochs", 0) < 150:
            continue
        s = c["seed"]; mt = os.path.getmtime(f)
        if s not in best or mt > best[s][0]:
            best[s] = (mt, r)
    out = {}
    for s, (mt, r) in best.items():
        pm = defaultdict(list)
        for cid, d in zip(r["test"]["cases"], r["test"]["dice_per_case"]):
            pm[patient_of(cid, dataset)].append(np.nanmean(d))
        out[s] = {p: float(np.nanmean(v)) for p, v in pm.items()}
    return out


def sel_lr(dir, fname, key, fallback=None):
    p = os.path.join(dir, fname)
    if os.path.exists(p):
        d = json.load(open(p))
        if key in d:
            return d[key]["lr"]
    return fallback


def main(dir, dataset, light_muon_lr, light_adamw_lr):
    # heavy LRs
    hM = sel_lr(dir, "best_lrs.json", "unet/muon")
    hA = sel_lr(dir, "best_lrs.json", "unet/adamw")
    # light LRs (prefer best_lrs_light.json, else CLI, else heavy with a warning)
    lM = light_muon_lr or sel_lr(dir, "best_lrs_light.json", "unet/muon", hM)
    lA = light_adamw_lr or sel_lr(dir, "best_lrs_light.json", "unet/adamw", hA)
    if not os.path.exists(os.path.join(dir, "best_lrs_light.json")) \
       and light_muon_lr is None:
        print("  WARNING: no best_lrs_light.json and no --light-*-lr; "
              "falling back to heavy LR for light (reduced-grid, not full-tune).")
    print(f"  LRs: heavy muon {hM}, heavy adamw {hA} | light muon {lM}, light adamw {lA}")

    Mh = load(dir, "muon", "full", hM, dataset)
    Ah = load(dir, "adamw", "full", hA, dataset)
    Ml = load(dir, "muon", "light", lM, dataset)
    Al = load(dir, "adamw", "light", lA, dataset)
    print(f"  run counts: Mh {len(Mh)}, Ah {len(Ah)}, Ml {len(Ml)}, Al {len(Al)}")
    seeds = sorted(set(Mh) & set(Ah) & set(Ml) & set(Al))
    if not seeds:
        print("  no common seeds -- check LRs/dirs.")
        return
    npat = len(set(Mh[seeds[0]]) & set(Ah[seeds[0]]) & set(Ml[seeds[0]]) & set(Al[seeds[0]]))
    print(f"  common seeds {seeds}, n_pat {npat}")

    intx = []
    for s in seeds:
        pats = sorted(set(Mh[s]) & set(Ah[s]) & set(Ml[s]) & set(Al[s]))
        dh = np.mean([Mh[s][p] - Ah[s][p] for p in pats])
        dl = np.mean([Ml[s][p] - Al[s][p] for p in pats])
        intx.append((dh - dl) * 100)
    intx = np.array(intx)

    # hierarchical bootstrap
    p0 = sorted(set(Mh[seeds[0]]) & set(Ah[seeds[0]]) & set(Ml[seeds[0]]) & set(Al[seeds[0]]))
    rng = np.random.default_rng(0); B = 10000; boot = np.empty(B)
    S, P = len(seeds), len(p0)
    for b in range(B):
        bs = rng.integers(0, S, S); bp = rng.integers(0, P, P); v = []
        for si in bs:
            s = seeds[si]
            dh = np.mean([Mh[s][p0[pi]] - Ah[s][p0[pi]] for pi in bp])
            dl = np.mean([Ml[s][p0[pi]] - Al[s][p0[pi]] for pi in bp])
            v.append(dh - dl)
        boot[b] = np.mean(v) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI includes 0)"
    print(f"  per-seed interaction: {[round(x,2) for x in intx]}")
    print(f"  mean {intx.mean():+.2f}  SD {intx.std(ddof=1):.2f}  "
          f"hier-boot 95% CI [{lo:+.2f}, {hi:+.2f}]  -> {sig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--light-muon-lr", type=float, default=None)
    ap.add_argument("--light-adamw-lr", type=float, default=None)
    a = ap.parse_args()
    print(f"=== {a.dataset} ({a.dir}) ===")
    main(a.dir, a.dataset, a.light_muon_lr, a.light_adamw_lr)
