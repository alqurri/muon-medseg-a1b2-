# =====================================================================
# HISTORICAL / DEMONSTRATION SCRIPT -- NOT the source of the paper's tables.
# This script does not apply the full validated run-selection (per-cell LR
# filtering, newest-per-seed deduplication, and ED/ES patient clustering)
# used for the reported numbers. Use regenerate_all.py as the single pipeline
# for every value and table in the paper. Retained only to document earlier
# analysis steps.
# =====================================================================
import json, glob, numpy as np

for nc in [9, 6, 4, 2]:
    d = f"results_syn_merge{nc}"
    try:
        bl = json.load(open(f"{d}/best_lrs.json"))
    except FileNotFoundError:
        print(f"merge{nc}: no best_lrs.json yet (not finished)")
        continue
    mlr = bl["unet/muon"]["lr"]
    alr = bl["unet/adamw"]["lr"]

    def dice(opt, lr):
        out = []
        for f in glob.glob(f"{d}/*unet_{opt}*aug-full*.json"):
            r = json.load(open(f))
            if r["config"]["epochs"] >= 150 and abs(r["config"]["lr"] - lr) < 1e-9:
                out.append(r["test"]["dice"])
        return out

    m = dice("muon", mlr)
    a = dice("adamw", alr)
    if m and a:
        delta = np.mean(m) - np.mean(a)
        print(f"merge{nc}: muon(lr={mlr}) {len(m)} runs={np.mean(m):.4f} | "
              f"adamw(lr={alr}) {len(a)} runs={np.mean(a):.4f} | DELTA {delta:+.4f}")
    else:
        print(f"merge{nc}: incomplete (muon {len(m)}, adamw {len(a)} runs)")
