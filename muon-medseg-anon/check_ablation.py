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
