"""
Difficulty-gradient figure for the paper.

Grouped bar chart: for each dataset, two bars (heavy-aug and light-aug) showing
the Muon-minus-AdamW U-Net Dice delta (in points), with bootstrap 95% CI
whiskers. Datasets ordered left->right by increasing baseline Dice (i.e.
decreasing task difficulty): Synapse (hardest) ... ACDC (easiest).

The figure is designed to show BOTH claims honestly at once:
  * the INTERACTION: within every dataset, the heavy bar is positive and the
    light bar sits near zero (or below, for ACDC).
  * the STEP (not ramp): Synapse's heavy bar towers over the other three,
    which cluster together -- the benefit magnitude tracks difficulty, and
    Synapse stands alone rather than sitting at the top of a smooth ramp.

Reads the same per-run JSON your analyze.py/interaction.py consume, from the
four results dirs. Run on the cluster:

    python make_figure.py \
        --acdc results --synapse results_synapse \
        --isic results_isic --camus results_camus \
        --out difficulty_gradient.pdf

Only U-Net, 200-epoch (Stage 2/3) runs are used. Deltas are computed the same
way as analyze.py: paired at the case level (patient for ACDC/Synapse, image
for ISIC/CAMUS), averaged over seeds, then Muon-minus-AdamW.
"""
from __future__ import annotations

import argparse
import glob
import json
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(outdir):
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


def _case_means(runs, opt, aug, model="unet"):
    """case_id -> Dice averaged over seeds, for U-Net (opt, aug), 200-epoch."""
    acc = defaultdict(list)
    for r in runs:
        c = r["config"]
        if c["model"] != model or c["optimizer"] != opt or c["aug"] != aug:
            continue
        if c["epochs"] < 150:
            continue
        t = r["test"]
        for cid, row in zip(t["cases"], t["dice_per_case"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc[cid].append(np.nanmean(row))
    return {k: np.nanmean(v) for k, v in acc.items()}


def _delta_ci(runs, aug, n_boot=10000, seed=0):
    """Muon-AdamW paired Dice delta (in points) + bootstrap 95% CI."""
    a = _case_means(runs, "adamw", aug)
    m = _case_means(runs, "muon", aug)
    common = sorted(set(a) & set(m))
    if len(common) < 3:
        return None
    d = np.array([m[c] - a[c] for c in common])
    d = d[~np.isnan(d)] * 100.0                       # -> Dice points
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def _baseline(runs, aug="full"):
    """AdamW mean Dice (for ordering datasets by baseline). aug='full'=heavy."""
    a = _case_means(runs, "adamw", aug)
    return np.nanmean(list(a.values())) * 100.0 if a else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc", default="results")
    ap.add_argument("--synapse", default="results_synapse")
    ap.add_argument("--isic", default="results_isic")
    ap.add_argument("--camus", default="results_camus")
    ap.add_argument("--flare", default="results_flare")
    ap.add_argument("--amos", default="results_amos")
    ap.add_argument("--brats", default="results_brats")
    ap.add_argument("--out", default="difficulty_gradient.pdf")
    a = ap.parse_args()

    dsets = {"AMOS22\n(MRI, 15-cls)": _load(a.amos),
             "FLARE22\n(CT, 13-cls)": _load(a.flare),
             "Synapse\n(CT, 9-cls)": _load(a.synapse),
             "BraTS\n(MRI, 4-cls)": _load(a.brats),
             "CAMUS\n(US, 4-cls)": _load(a.camus),
             "ISIC\n(derm, bin)": _load(a.isic),
             "ACDC\n(MRI, 4-cls)": _load(a.acdc)}

    # order by baseline difficulty (ascending baseline Dice = hard->easy);
    # NaN baselines sort last, so guard with a large fallback
    def _ord_key(k):
        b = _baseline(dsets[k])
        return b if np.isfinite(b) else 1e9
    order = sorted(dsets, key=_ord_key)
    rows = []
    for name in order:
        runs = dsets[name]
        base = _baseline(runs)
        heavy = _delta_ci(runs, "full")
        light = _delta_ci(runs, "light")
        rows.append((name, base, heavy, light))
        base_s = f"{base:5.1f}" if np.isfinite(base) else "  nan"
        heavy_s = (f"heavy {heavy[0]:+.2f} [{heavy[1]:+.2f},{heavy[2]:+.2f}]"
                   if heavy else "heavy --")
        light_s = (f"light {light[0]:+.2f} [{light[1]:+.2f},{light[2]:+.2f}]"
                   if light else "light --")
        print(f"{name.splitlines()[0]:10} baseline {base_s}  {heavy_s}  {light_s}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(rows))
    w = 0.38
    c_heavy, c_light = "#2c6fbb", "#c44e52"

    for i, (name, base, heavy, light) in enumerate(rows):
        if heavy:
            ax.bar(i - w/2, heavy[0], w, color=c_heavy,
                   label="heavy aug" if i == 0 else None)
            ax.errorbar(i - w/2, heavy[0],
                        yerr=[[heavy[0]-heavy[1]], [heavy[2]-heavy[0]]],
                        fmt="none", ecolor="black", capsize=2.5, lw=0.9)
        if light:
            ax.bar(i + w/2, light[0], w, color=c_light,
                   label="light aug" if i == 0 else None)
            ax.errorbar(i + w/2, light[0],
                        yerr=[[light[0]-light[1]], [light[2]-light[0]]],
                        fmt="none", ecolor="black", capsize=2.5, lw=0.9)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("Muon $-$ AdamW\nU-Net Dice (points)", fontsize=10)
    ax.set_title("Muon's advantage: large on low-baseline tasks (CT and MRI), "
                 "small on high-baseline tasks; light aug removes it",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    # annotate the one negative bar
    for i, (name, base, heavy, light) in enumerate(rows):
        if light and light[0] < -0.1:
            ax.annotate("harm", (i + w/2, light[0]),
                        textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=7.5, color=c_light)
    # difficulty direction, placed under the tick labels (below axis)
    ax.annotate("", xy=(len(rows)-1, -0.16), xytext=(0, -0.16),
                xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.text(-0.15, -0.16, "lower baseline", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=7.5, style="italic", color="gray")
    ax.text(len(rows)-1+0.15, -0.16, "higher baseline",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=7.5, style="italic", color="gray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"\nwrote {a.out} (+ .png)")


if __name__ == "__main__":
    main()
