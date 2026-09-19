"""
paired_noortho.py -- paired Muon-NS vs Muon difference (the reviewer's ask).

The orthogonalization table currently reports Muon-AdamW and (Muon-NS)-AdamW
separately. To support any statement about whether removing Newton-Schulz
changes the result, we need the DIRECT paired difference

    delta_seed = mean_patients(Dice_MuonNS[seed]) - mean_patients(Dice_Muon[seed])

per seed, then a hierarchical bootstrap CI over seeds and patients. A CI tight
around zero => the two are close; a CI far from zero => removing orthogonalization
changes the result. This does NOT run an equivalence test (no pre-specified
margin), so we report the paired difference and its CI, not "equivalence".

Requires, in each dataset's dirs:
  - the Muon runs (from the main study, e.g. results_synapse/*muon*aug-full*)
  - the Muon-NS runs (from results_*_noortho/*muon_noortho*aug-full*)
They must share seeds. Patients are clustered (ED/ES merged) as in analyze_v2.

Usage:
  python paired_noortho.py \
      --muon-dir results_synapse --noortho-dir results_syn_noortho --dataset synapse
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


def patient_means(run, dataset):
    t = run["test"]
    per = defaultdict(list)
    for cid, drow in zip(t["cases"], t["dice_per_case"]):
        with np.errstate(invalid="ignore"):
            per[patient_of(cid, dataset)].append(np.nanmean(drow))
    return {p: float(np.nanmean(v)) for p, v in per.items()}


def load_cell(outdir, opt_substr, model="unet", aug="full"):
    """seed -> patient_means dict, for the max-epoch runs of one optimizer."""
    runs = {}
    for f in glob.glob(os.path.join(outdir, "*.json")):
        b = os.path.basename(f)
        if b in ("best_lrs.json", "summary.json"):
            continue
        r = json.load(open(f))
        c = r.get("config", {})
        if c.get("model") != model or c.get("aug") != aug:
            continue
        if opt_substr not in c.get("optimizer", ""):
            continue
        runs.setdefault(c.get("epochs", 0), []).append(r)
    if not runs:
        return {}
    full = max(runs)
    out = {}
    for r in runs[full]:
        out[r["config"]["seed"]] = r
    return out


def main(muon_dir, noortho_dir, dataset):
    muon = load_cell(muon_dir, "muon")            # matches "muon" (not muon_noortho if dir separate)
    # guard: exclude muon_noortho if present in the muon dir
    muon = {s: r for s, r in muon.items() if r["config"]["optimizer"] == "muon"}
    nsr = load_cell(noortho_dir, "muon_noortho")
    seeds = sorted(set(muon) & set(nsr))
    if len(seeds) < 2:
        print(f"{dataset}: need >=2 shared seeds, have {len(seeds)} "
              f"(muon {sorted(muon)}, ns {sorted(nsr)})")
        return

    # per-seed paired difference (Muon-NS minus Muon), patient-clustered
    per_seed = []
    for s in seeds:
        pm = patient_means(muon[s], dataset)
        pn = patient_means(nsr[s], dataset)
        common = sorted(set(pm) & set(pn))
        per_seed.append(np.mean([pn[p] - pm[p] for p in common]) * 100.0)
    per_seed = np.array(per_seed)

    # hierarchical bootstrap over seeds and patients
    Pm = {s: patient_means(muon[s], dataset) for s in seeds}
    Pn = {s: patient_means(nsr[s], dataset) for s in seeds}
    patients = sorted(set.intersection(*[set(Pm[s]) for s in seeds],
                                       *[set(Pn[s]) for s in seeds]))
    rng = np.random.default_rng(0)
    S, P = len(seeds), len(patients)
    boot = np.empty(10000)
    for b in range(10000):
        bs = rng.integers(0, S, S); bp = rng.integers(0, P, P)
        vals = []
        for si in bs:
            s = seeds[si]
            vals.append(np.mean([Pn[s][patients[pi]] - Pm[s][patients[pi]] for pi in bp]))
        boot[b] = np.mean(vals) * 100.0
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"=== {dataset}: paired (Muon-NS) - Muon ===")
    print(f"  per-seed diffs: {[round(x,3) for x in per_seed]}")
    print(f"  mean {per_seed.mean():+.3f}  SD {per_seed.std(ddof=1):.3f}  "
          f"n_seeds {len(seeds)}  n_pat {len(patients)}")
    print(f"  hierarchical bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  -> {'CI includes 0 (no detectable Muon-vs-MuonNS difference)' if lo <= 0 <= hi else 'CI excludes 0 (they differ)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--muon-dir", required=True, help="dir with the full-Muon runs")
    ap.add_argument("--noortho-dir", required=True, help="dir with the muon_noortho runs")
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    main(a.muon_dir, a.noortho_dir, a.dataset)
