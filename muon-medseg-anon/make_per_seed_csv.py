import csv, regenerate_all as ra
R = ra.compute()
rows = []
for ds in ra.ORDER:
    name = R[ds]["name"]
    for model, label in (("unet", "U-Net"), ("transunet", "TransUNet")):
        for aug in ("heavy", "light"):
            c = R[ds].get(f"{model}_{aug}")
            if not c:
                continue
            A, M, seeds, pats = c["A"], c["M"], c["seeds"], c["pats"]
            for s in seeds:
                aw = sum(A[s][p] for p in pats) / len(pats) * 100
                mu = sum(M[s][p] for p in pats) / len(pats) * 100
                rows.append([name, aug, label, s, round(aw, 4), round(mu, 4), round(mu - aw, 4)])
with open("per_seed_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "augmentation", "architecture", "seed", "adamw_dice", "muon_dice", "delta"])
    w.writerows(rows)
print("wrote", len(rows), "rows")
