"""
Augmentation x optimizer interaction test + per-seed consistency.

The headline claim is now "Muon's effect flips sign with augmentation regime".
That is a difference-of-differences and must be tested as one, not inferred
from four separately-significant within-regime cells:

    interaction = (Muon - AdamW)_full  -  (Muon - AdamW)_light

Paired at patient-case level. A bootstrap CI on the interaction that excludes
zero is what licenses the word "interaction" in the abstract.
"""
import glob, json
import numpy as np
from collections import defaultdict
from scipy import stats


def load(outdir="results"):
    runs = []
    for f in glob.glob(f"{outdir}/*.json"):
        if any(x in f for x in ("best_lrs", "summary")):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and "config" in r and "test" in r:
            runs.append(r)
    return runs


def case_means(runs, model, opt, aug, metric):
    """patient-case -> mean over seeds of that case's metric (200-epoch runs)."""
    acc = defaultdict(list)
    for r in runs:
        c = r["config"]
        if (c["model"], c["optimizer"], c["aug"]) != (model, opt, aug):
            continue
        if c["epochs"] < 150:
            continue
        t = r["test"]
        col = t[f"{metric}_per_case"]
        for cid, row in zip(t["cases"], col):
            if str(cid).startswith("syn"):
                break
            with np.errstate(invalid="ignore"):
                acc[cid].append(np.nanmean(row))
    return {k: np.nanmean(v) for k, v in acc.items()}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def interaction(runs, model, metric="dice"):
    cells = {(o, a): case_means(runs, model, o, a, metric)
             for o in ("adamw", "muon") for a in ("full", "light")}
    common = set.intersection(*[set(c) for c in cells.values()])
    if len(common) < 3:
        return None
    common = sorted(common)
    g = lambda o, a: np.array([cells[(o, a)][c] for c in common])
    dfull = g("muon", "full") - g("adamw", "full")
    dlight = g("muon", "light") - g("adamw", "light")
    inter = dfull - dlight
    lo, hi = boot_ci(inter)
    try:
        w, p = stats.wilcoxon(dfull, dlight)
    except ValueError:
        p = 1.0
    return dict(model=model, metric=metric, n=len(common),
                mean_full=float(dfull.mean()), mean_light=float(dlight.mean()),
                interaction=float(inter.mean()), ci=(lo, hi), p=float(p),
                crosses_zero=bool(lo <= 0 <= hi))


def seed_table(runs, model, aug, metric="dice"):
    d = defaultdict(dict)
    for r in runs:
        c = r["config"]
        if c["model"] != model or c["aug"] != aug or c["epochs"] < 150:
            continue
        t = r["test"]
        if t["cases"] and str(t["cases"][0]).startswith("syn"):
            continue
        d[c["optimizer"]][c["seed"]] = t[metric]
    a, mu = d.get("adamw", {}), d.get("muon", {})
    s = sorted(set(a) & set(mu))
    if not s:
        return None
    better = sum((mu[i] > a[i]) if metric == "dice" else (mu[i] < a[i]) for i in s)
    return dict(aug=aug, n=len(s), muon_better=better,
                deltas=[round(mu[i] - a[i], 4) for i in s])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results",
                    help="results dir to analyze (e.g. results_synapse)")
    args = ap.parse_args()
    runs = load(args.outdir)
    print(f"loaded {len(runs)} runs from {args.outdir}/\n")
    print("=== AUGMENTATION x OPTIMIZER INTERACTION (diff-of-diffs, Muon-AdamW) ===")
    print("   positive Dice delta = Muon better; interaction = full_effect - light_effect\n")
    for model in ("unet", "transunet"):
        for metric in ("dice", "hd95"):
            r = interaction(runs, model, metric)
            if not r:
                continue
            # significant only if BOTH the bootstrap CI excludes zero AND
            # the Wilcoxon p < 0.05. CI-alone over-fires on borderline p
            # (e.g. p=0.064), which would mislabel results in the paper.
            if (not r["crosses_zero"]) and r["p"] < 0.05:
                sig = "  * SIGNIFICANT"
            elif (not r["crosses_zero"]) or r["p"] < 0.10:
                sig = "  (borderline)"
            else:
                sig = ""
            print(f"  {model:10} {metric:5}  full Δ {r['mean_full']:+.4f}   "
                  f"light Δ {r['mean_light']:+.4f}   "
                  f"interaction {r['interaction']:+.4f}  "
                  f"CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  p={r['p']:.3f}{sig}")
    print("\n=== PER-SEED CONSISTENCY (Dice) ===")
    for model in ("unet", "transunet"):
        for aug in ("full", "light"):
            s = seed_table(runs, model, aug, "dice")
            if s:
                print(f"  {model:10} {aug:6}  Muon better in {s['muon_better']}/{s['n']} "
                      f"seeds  deltas {s['deltas']}")
