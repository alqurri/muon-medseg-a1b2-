"""
analyze_v2.py -- statistically corrected optimizer comparison.

Fixes the two inference flaws a reviewer flagged in the original analyze.py:

  (1) SEED VARIANCE WAS COLLAPSED. The old pipeline averaged each case over
      5 seeds, THEN tested over cases. That measures case-to-case variance and
      hides training-run (seed) variance -- the very component an optimizer
      claim depends on. A patient-wise p<0.001 over seed-averaged cases is NOT
      evidence the effect replicates across training runs.

  (2) NON-INDEPENDENT SUB-VOLUMES. ACDC (ED/ES) and CAMUS (ED/ES) contribute
      two correlated measurements per patient. Treating them as independent
      cases inflates n and understates correlation.

This script reports THREE views, from most to least conservative, so the
paper can lean on the honest one:

  A. SEED-LEVEL paired test (primary replication evidence).
     For each seed s, compute the patient-clustered mean Dice for Muon and
     AdamW; take the paired difference d_s = mean_patients(Muon_s) -
     mean_patients(AdamW_s). Report the 5 values, their mean +/- SD, the
     paired count (k/5 with d_s > 0), and a paired test over the 5 seed
     differences (Wilcoxon if n>=6 pairs, else sign / t as a descriptive
     companion -- with 5 seeds the honest summary is the SD and k/5, not a
     p-value). This is the "does it replicate across runs" answer.

  B. PATIENT-CLUSTERED, SEED-AVERAGED (the old test, but clustered).
     Average each PATIENT (pooling ED/ES) over seeds, then paired Wilcoxon
     over patients. Reported for continuity, but explicitly labelled as
     measuring case variance at fixed averaged-seed, NOT run-to-run
     replication.

  C. HIERARCHICAL BOOTSTRAP (propagates BOTH variance components).
     Resample patients with replacement AND seeds with replacement, recompute
     the Muon-AdamW patient-clustered mean difference each draw, and take the
     2.5/97.5 percentiles. This CI is the fullest honest summary; it is wider
     than the old case-level CI because it carries seed variance.

Patient clustering: cases are grouped by patient id. For ACDC/CAMUS a case id
encodes patient + frame (ED/ES); we strip the frame suffix so both frames map
to one patient, and average within patient before any test. For datasets that
are already one-image-per-subject (ISIC) or one-volume-per-patient
(Synapse/FLARE/AMOS/BraTS) this is a no-op.

Usage:
    python analyze_v2.py --outdir results_camus --dataset camus
    python analyze_v2.py --outdir results        --dataset acdc

The --dataset flag only controls the patient-id extraction rule; everything
else is generic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Patient-id extraction: map a case id to the patient it belongs to.
# The ONLY datasets with >1 measurement per patient are ACDC and CAMUS
# (ED/ES frames). Everything else is already one case per patient.
# ---------------------------------------------------------------------------
def patient_of(case_id: str, dataset: str) -> str:
    c = str(case_id)
    if dataset in ("acdc", "camus"):
        # strip a trailing frame marker: _ED / _ES / _frame01 / *_0 / *_1 etc.
        # Adjust these patterns to your actual case-id scheme.
        c = re.sub(r"[_\-](ed|es|ED|ES)$", "", c)
        c = re.sub(r"[_\-]?frame\d+$", "", c, flags=re.I)
        c = re.sub(r"[_\-](0|1)$", "", c)   # last resort: trailing frame index
    return c


def load_runs(outdir):
    runs = []
    for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        b = os.path.basename(f)
        if b in ("best_lrs.json", "summary.json"):
            continue
        r = json.load(open(f))
        if "test" not in r or "config" not in r:
            continue
        cases = r["test"].get("cases", [])
        if cases and str(cases[0]).startswith("syn"):     # drop smoke tests
            continue
        runs.append(r)
    # keep only the max epoch budget (Stage 2/3), as before
    if runs:
        budgets = defaultdict(int)
        for r in runs:
            budgets[r["config"].get("epochs", 0)] += 1
        full = max(budgets)
        runs = [r for r in runs if r["config"].get("epochs", 0) == full]
    return runs


def patient_means_for_run(run, dataset):
    """One run -> {patient_id: mean Dice over that patient's frames}."""
    t = run["test"]
    per_pat = defaultdict(list)
    for cid, drow in zip(t["cases"], t["dice_per_case"]):
        with np.errstate(invalid="ignore"):
            per_pat[patient_of(cid, dataset)].append(np.nanmean(drow))
    return {p: float(np.nanmean(v)) for p, v in per_pat.items()}


def cell_runs(runs, model, opt, aug):
    out = {}
    for r in runs:
        c = r["config"]
        if c["model"] == model and c["optimizer"] == opt and c["aug"] == aug:
            out[c["seed"]] = r
    return out                    # seed -> run


def seed_level_analysis(runs, model, aug, dataset):
    """View A: paired seed-level differences (the replication test)."""
    A = cell_runs(runs, model, "adamw", aug)
    M = cell_runs(runs, model, "muon", aug)
    seeds = sorted(set(A) & set(M))
    if len(seeds) < 2:
        return None
    d = []
    for s in seeds:
        pa = patient_means_for_run(A[s], dataset)
        pm = patient_means_for_run(M[s], dataset)
        common = sorted(set(pa) & set(pm))
        # paired at patient level WITHIN a seed, then reduced to one scalar
        diff = np.mean([pm[p] - pa[p] for p in common]) * 100.0
        d.append(diff)
    d = np.array(d)
    k = int((d > 0).sum())
    res = dict(seeds=seeds, per_seed_delta=d.tolist(),
               mean=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
               k_of_n=f"{k}/{len(d)}")
    # paired test over seed differences vs 0. With 5 seeds this is low power;
    # report it but the SD and k/n are the honest summary.
    try:
        w, p = stats.wilcoxon(d)          # needs >=6 for a real p; scipy warns
        res["wilcoxon_p"] = float(p)
    except Exception:
        res["wilcoxon_p"] = float("nan")
    # sign test (exact) as a robust low-n companion
    res["sign_p"] = float(stats.binomtest(k, len(d), 0.5).pvalue)
    return res


def patient_clustered_pooled(runs, model, aug, dataset):
    """View B: seed-averaged, patient-clustered paired Wilcoxon over patients."""
    A = cell_runs(runs, model, "adamw", aug)
    M = cell_runs(runs, model, "muon", aug)
    seeds = sorted(set(A) & set(M))
    if not seeds:
        return None
    # patient -> mean over seeds
    def avg(cell):
        acc = defaultdict(list)
        for s in seeds:
            for p, v in patient_means_for_run(cell[s], dataset).items():
                acc[p].append(v)
        return {p: np.mean(v) for p, v in acc.items()}
    pa, pm = avg(A), avg(M)
    common = sorted(set(pa) & set(pm))
    a = np.array([pa[p] for p in common]); m = np.array([pm[p] for p in common])
    d = (m - a) * 100.0
    try:
        w, p = stats.wilcoxon(m, a)
    except ValueError:
        p = 1.0
    return dict(n_patients=len(common), delta=float(d.mean()), p=float(p))


def hierarchical_bootstrap(runs, model, aug, dataset, n_boot=10000, seed=0):
    """View C: resample patients AND seeds; propagate both variance sources."""
    A = cell_runs(runs, model, "adamw", aug)
    M = cell_runs(runs, model, "muon", aug)
    seeds = sorted(set(A) & set(M))
    if len(seeds) < 2:
        return None
    # precompute patient-mean dicts per seed
    Amaps = {s: patient_means_for_run(A[s], dataset) for s in seeds}
    Mmaps = {s: patient_means_for_run(M[s], dataset) for s in seeds}
    patients = sorted(set.intersection(*[set(Amaps[s]) for s in seeds],
                                       *[set(Mmaps[s]) for s in seeds]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    S, P = len(seeds), len(patients)
    for b in range(n_boot):
        bs = rng.integers(0, S, S)            # resample seeds
        bp = rng.integers(0, P, P)            # resample patients
        diffs = []
        for si in bs:
            s = seeds[si]
            am, mm = Amaps[s], Mmaps[s]
            diffs.append(np.mean([mm[patients[pi]] - am[patients[pi]] for pi in bp]))
        boot[b] = np.mean(diffs) * 100.0
    return dict(mean=float(boot.mean()),
                lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)))


def _seed_patient_delta(cellA, cellM, s, dataset):
    """Patient-clustered mean (Muon-AdamW) Dice for one seed, in points."""
    pa = patient_means_for_run(cellA[s], dataset)
    pm = patient_means_for_run(cellM[s], dataset)
    common = sorted(set(pa) & set(pm))
    if not common:
        return None, None
    return (np.mean([pm[p] - pa[p] for p in common]) * 100.0, common)


def interaction_analysis(runs, model, dataset, n_boot=10000, seed=0):
    """
    Seed-level optimizer x augmentation INTERACTION (difference-of-differences),
    with a hierarchical bootstrap over seeds and patients.

    Per seed s:  intx_s = delta_heavy(s) - delta_light(s),
    where delta_aug(s) is the patient-clustered mean (Muon-AdamW) Dice at that
    seed. We report the per-seed interactions, mean +/- SD, k/n favouring the
    heavy>light direction, a sign test, and a hierarchical-bootstrap 95% CI that
    resamples seeds and patients jointly. A CI excluding zero is the honest
    "the interaction replicates" criterion; the seed-level p is under-powered at
    n=5 and reported only as a companion.

    Requires both a 'full' (heavy) and a 'light' arm for this model.
    """
    Ah = cell_runs(runs, model, "adamw", "full")
    Mh = cell_runs(runs, model, "muon", "full")
    Al = cell_runs(runs, model, "adamw", "light")
    Ml = cell_runs(runs, model, "muon", "light")
    seeds = sorted(set(Ah) & set(Mh) & set(Al) & set(Ml))
    if len(seeds) < 2:
        return None                      # no light arm (e.g. FLARE U-Net) -> skip

    # per-seed interaction (scalar) for view A
    intx = []
    for s in seeds:
        dh, _ = _seed_patient_delta(Ah, Mh, s, dataset)
        dl, _ = _seed_patient_delta(Al, Ml, s, dataset)
        if dh is None or dl is None:
            continue
        intx.append(dh - dl)
    intx = np.array(intx)
    if len(intx) < 2:
        return None
    k = int((intx > 0).sum())

    # hierarchical bootstrap: resample seeds and patients jointly, recompute
    # the interaction each draw. Patients common to all four cells across seeds.
    Ahm = {s: patient_means_for_run(Ah[s], dataset) for s in seeds}
    Mhm = {s: patient_means_for_run(Mh[s], dataset) for s in seeds}
    Alm = {s: patient_means_for_run(Al[s], dataset) for s in seeds}
    Mlm = {s: patient_means_for_run(Ml[s], dataset) for s in seeds}
    patients = sorted(set.intersection(
        *[set(Ahm[s]) for s in seeds], *[set(Mhm[s]) for s in seeds],
        *[set(Alm[s]) for s in seeds], *[set(Mlm[s]) for s in seeds]))
    rng = np.random.default_rng(seed)
    S, P = len(seeds), len(patients)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        bs = rng.integers(0, S, S)
        bp = rng.integers(0, P, P)
        vals = []
        for si in bs:
            s = seeds[si]
            dh = np.mean([Mhm[s][patients[pi]] - Ahm[s][patients[pi]] for pi in bp])
            dl = np.mean([Mlm[s][patients[pi]] - Alm[s][patients[pi]] for pi in bp])
            vals.append((dh - dl))
        boot[b] = np.mean(vals) * 100.0

    res = dict(
        per_seed=intx.tolist(),
        mean=float(intx.mean()),
        sd=float(intx.std(ddof=1)) if len(intx) > 1 else float("nan"),
        k_of_n=f"{k}/{len(intx)}",
        sign_p=float(stats.binomtest(k, len(intx), 0.5).pvalue),
        boot_lo=float(np.percentile(boot, 2.5)),
        boot_hi=float(np.percentile(boot, 97.5)),
        boot_mean=float(boot.mean()),
        crosses_zero=bool(np.percentile(boot, 2.5) <= 0 <= np.percentile(boot, 97.5)),
    )
    try:
        res["wilcoxon_p"] = float(stats.wilcoxon(intx)[1])
    except Exception:
        res["wilcoxon_p"] = float("nan")
    return res


def main(outdir, dataset):
    runs = load_runs(outdir)
    if not runs:
        raise SystemExit(f"no runs in {outdir}")
    models = sorted({r["config"]["model"] for r in runs})
    augs = sorted({r["config"]["aug"] for r in runs})
    print(f"# {outdir}  dataset={dataset}  ({len(runs)} runs)\n")
    for aug in augs:
        for m in models:
            A = seed_level_analysis(runs, m, aug, dataset)
            if A is None:
                continue
            B = patient_clustered_pooled(runs, m, aug, dataset)
            C = hierarchical_bootstrap(runs, m, aug, dataset)
            print(f"[{m} / {aug}]")
            print(f"  A seed-level : delta {A['mean']:+.3f} +/- {A['sd']:.3f} SD "
                  f"| {A['k_of_n']} seeds | sign p={A['sign_p']:.3f}"
                  + (f" wilcoxon p={A['wilcoxon_p']:.3f}" if not np.isnan(A['wilcoxon_p']) else ""))
            print(f"    per-seed deltas: {[round(x,3) for x in A['per_seed_delta']]}")
            if B:
                print(f"  B pooled(clust): delta {B['delta']:+.3f}  "
                      f"patient-Wilcoxon p={B['p']:.3f}  n_pat={B['n_patients']}")
            if C:
                print(f"  C hier-boot   : delta {C['mean']:+.3f}  "
                      f"95% CI [{C['lo']:+.3f}, {C['hi']:+.3f}]  (seeds+patients)")
            print()

    # ---- interaction (difference-of-differences), seed-level + hier bootstrap ----
    print("=== INTERACTION (heavy - light), seed-level + hierarchical bootstrap ===")
    for m in models:
        I = interaction_analysis(runs, m, dataset)
        if I is None:
            print(f"[{m}] interaction: not available (needs both heavy and light arms)")
            continue
        star = "" if I["crosses_zero"] else "  * CI excludes zero"
        print(f"[{m}] interaction {I['mean']:+.3f} +/- {I['sd']:.3f} SD | "
              f"{I['k_of_n']} seeds | hier-boot 95% CI "
              f"[{I['boot_lo']:+.3f}, {I['boot_hi']:+.3f}]{star}")
        print(f"    per-seed interactions: {[round(x,3) for x in I['per_seed']]}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", required=True,
                    help="acdc|camus trigger ED/ES patient clustering; "
                         "others are one-case-per-patient (no-op).")
    a = ap.parse_args()
    main(a.outdir, a.dataset)
