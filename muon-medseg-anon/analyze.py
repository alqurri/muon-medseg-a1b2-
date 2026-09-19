"""
Analysis. Reads results/*.json, produces the numbers that go in the paper.

Design decisions baked in here, all of which matter more than the model code:

 * Pairing is at PATIENT level, not slice level. Slices from one patient are
   correlated; treating them as independent inflates n by ~10x and hands you
   significance that isn't real.
 * The unit of the test is the per-patient mean across seeds. Seeds are
   nuisance variation, not extra samples.
 * Wilcoxon signed-rank (paired, non-parametric) because per-case Dice is
   bounded and skewed. A t-test on Dice is a bad default.
 * Effect size and a bootstrap CI are reported alongside p. With 5 seeds you
   have limited power and the CI is the honest summary; p alone is not.
 * The kill criterion is evaluated automatically so you cannot quietly move
   the goalposts after seeing the result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# Pre-register this. Change it BEFORE you look at results, or not at all.
KILL_DELTA_DICE = 0.005     # 0.5 Dice points


def load(outdir):
    runs = []
    for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        if os.path.basename(f) == "best_lrs.json":
            continue
        r = json.load(open(f))
        if "test" not in r:
            continue
        cases = r["test"].get("cases", [])
        # Drop synthetic smoke-test runs: their case ids start with 'syn'.
        # Mixing them with real data silently averages two metric regimes
        # (trivial ellipses ~0.99 vs real ACDC ~0.92) and manufactures an
        # effect. This is exactly the contamination that nearly produced a
        # wrong published number.
        if cases and str(cases[0]).startswith("syn"):
            continue
        runs.append(r)

    if not runs:
        return runs

    # Keep ONLY final-budget (Stage 2/3) runs, not the reduced-epoch Stage 1
    # LR-sweep runs. The sweep runs many LRs -- including deliberately bad ones
    # (grid edges that diverge) -- at a shorter budget; pooling them with the
    # tuned full-length runs drags the mean and inflates HD95. This was benign
    # on datasets whose sweep LRs sat near the tuned optimum, but on vmunet the
    # sweep's diverged high-LR runs corrupted the aggregate (apparent -0.065
    # "collapse" that per-run inspection showed was false). Aggregate the
    # majority epoch budget only -- the full-length runs.
    epoch_counts = {}
    for r in runs:
        ep = r.get("config", {}).get("epochs", 0)
        epoch_counts[ep] = epoch_counts.get(ep, 0) + 1
    if epoch_counts:
        full_budget = max(epoch_counts)          # Stage 2/3 use the largest budget
        runs = [r for r in runs if r.get("config", {}).get("epochs", 0) == full_budget]

    # Every surviving run must score the same test set the same way. A mix of
    # per-slice and per-volume scoring (different n_cases) is not comparable
    # and must not be averaged -- fail loudly rather than silently.
    sizes = {}
    for r in runs:
        sizes.setdefault(len(r["test"]["cases"]), []).append(
            f"{r['config']['model']}/{r['config']['optimizer']}"
            f"/lr{r['config']['lr']:g}/s{r['config']['seed']}")
    if len(sizes) > 1:
        msg = ["MIXED TEST-SET SIZES -- results are not comparable:"]
        for n, who in sorted(sizes.items()):
            msg.append(f"  n_cases={n}: {len(who)} runs, e.g. {who[:3]}")
        msg.append("Archive the odd ones out and re-run; do not trust any "
                   "averaged number until this is a single size.")
        raise SystemExit("\n".join(msg))
    return runs


def bootstrap_ci(d, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def per_case_matrix(runs, model, opt, aug):
    """case_id -> mean metric over seeds, for one (model, optimizer, aug) cell."""
    dice, hd = defaultdict(list), defaultdict(list)
    seeds = set()
    for r in runs:
        c = r["config"]
        if c["model"] != model or c["optimizer"] != opt or c["aug"] != aug:
            continue
        seeds.add(c["seed"])
        t = r["test"]
        for cid, d, h in zip(t["cases"], t["dice_per_case"], t["hd95_per_case"]):
            with np.errstate(invalid="ignore"):
                dice[cid].append(np.nanmean(d))
                hd[cid].append(np.nanmean(h))
    if not dice:
        return None
    cases = sorted(dice)
    with np.errstate(invalid="ignore"):
        return dict(cases=cases,
                    dice=np.array([np.nanmean(dice[c]) for c in cases]),
                    hd95=np.array([np.nanmean(hd[c]) for c in cases]),
                    n_seeds=len(seeds))


def compare(runs, model, aug, metric="dice", higher_better=True):
    A = per_case_matrix(runs, model, "adamw", aug)
    M = per_case_matrix(runs, model, "muon", aug)
    if A is None or M is None:
        return None
    common = sorted(set(A["cases"]) & set(M["cases"]))
    ia = [A["cases"].index(c) for c in common]
    im = [M["cases"].index(c) for c in common]
    a, m = A[metric][ia], M[metric][im]
    ok = ~(np.isnan(a) | np.isnan(m))
    a, m = a[ok], m[ok]
    if len(a) < 3:
        return None

    d = m - a                                  # Muon minus AdamW
    lo, hi = bootstrap_ci(d)
    try:
        w, p = stats.wilcoxon(m, a)
    except ValueError:                         # all differences zero
        w, p = float("nan"), 1.0
    pooled = np.sqrt((a.var(ddof=1) + m.var(ddof=1)) / 2)
    return dict(model=model, aug=aug, metric=metric, n_cases=int(len(a)),
                n_seeds_adamw=A["n_seeds"], n_seeds_muon=M["n_seeds"],
                adamw=float(a.mean()), muon=float(m.mean()),
                delta=float(d.mean()), ci=(lo, hi), p=float(p),
                cohens_d=float(d.mean() / pooled) if pooled > 0 else 0.0,
                crosses_zero=bool(lo <= 0 <= hi),
                higher_better=higher_better)


def timing(runs):
    out = defaultdict(list)
    for r in runs:
        c = r["config"]
        out[(c["model"], c["optimizer"])].append(r.get("sec_per_step", np.nan))
    return {k: float(np.nanmedian(v)) for k, v in out.items()}


def main(outdir):
    runs = load(outdir)
    if not runs:
        raise SystemExit(f"no completed runs in {outdir}/")
    print(f"loaded {len(runs)} runs\n")

    fm = {}
    for r in runs:
        fm.setdefault(r["config"]["model"], r.get("frac_muon"))
    print("Muon parameter surface area (Stage 0):")
    for m, f in fm.items():
        print(f"  {m:<12} frac_muon = {f:.1%}" if f is not None else f"  {m}: n/a")

    print("\nwall-clock, median sec/step:")
    for (m, o), t in sorted(timing(runs).items()):
        print(f"  {m:<12} {o:<6} {t*1000:8.1f} ms")

    augs = sorted({r["config"]["aug"] for r in runs})
    models = sorted({r["config"]["model"] for r in runs})
    results = []
    for aug in augs:
        print(f"\n=== aug regime: {aug} ===")
        for m in models:
            for metric, hb in (("dice", True), ("hd95", False)):
                c = compare(runs, m, aug, metric, hb)
                if not c:
                    continue
                results.append(c)
                arrow = "higher better" if hb else "lower better"
                sig = "" if c["crosses_zero"] else "  *"
                print(f"  {m:<11} {metric:<5} ({arrow:<13})  "
                      f"AdamW {c['adamw']:.4f}  Muon {c['muon']:.4f}  "
                      f"delta {c['delta']:+.4f}  "
                      f"95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  "
                      f"p={c['p']:.3f}  d={c['cohens_d']:+.2f}  "
                      f"n={c['n_cases']}{sig}")

    # ---- pre-registered kill criterion -----------------------------------
    dice_res = [r for r in results if r["metric"] == "dice" and r["aug"] == "full"]
    if dice_res:
        flat = all(r["crosses_zero"] and abs(r["delta"]) < KILL_DELTA_DICE
                   for r in dice_res)
        print("\n" + "=" * 66)
        if flat:
            print("KILL CRITERION MET.")
            print(f"  Every Dice CI spans zero and |delta| < "
                  f"{KILL_DELTA_DICE:.3f} on all architectures.")
            print("  Stop here. Do not add seeds hunting for significance --")
            print("  a well-powered null on a hyped optimizer, in a domain")
            print("  where nobody has checked, is the publishable result.")
        else:
            print("Kill criterion NOT met -- at least one cell shows a real effect.")
            print("  Check whether the effect tracks frac_muon across")
            print("  architectures. If it does, that supports the spectral-")
            print("  geometry account. If it doesn't, the mechanism is wrong,")
            print("  which is the more interesting finding. Stage 3 next.")
        print("=" * 66)

    json.dump(results, open(os.path.join(outdir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    main(ap.parse_args().outdir)
