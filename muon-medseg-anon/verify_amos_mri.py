"""
Verify the AMOS MRI-only filter kept NO CT cases (and lost no MRI).

AMOS22 mixes CT and MRI in the same imagesTr/labelsTr folders by case id. The
loader keeps case_id >= MRI_MIN_ID as "MRI". That threshold is metadata; this
script checks the *pixels* instead and cross-checks the two.

Modality fingerprint (independent of the id threshold):
  CT  -> Hounsfield units: an air peak near -1000, substantial sub-water negatives.
  MRI -> arbitrary positive intensities, no air peak, min not far below 0.

    python verify_amos_mri.py --images-dir ./data/AMOS22/imagesTr --mri-min-id 500
    # or check exactly the test volumes you scored:
    python verify_amos_mri.py --images-dir ... --ids amos_0507 amos_0510 ...

Exit code is nonzero if any kept "MRI" case is actually CT (contamination) so it
can gate a pipeline. Prints every disagreement.
"""
from __future__ import annotations
import os, re, glob, argparse
import numpy as np


def classify_modality(arr, air_thresh=-500.0, neg_frac_thresh=0.02):
    """Return ('CT'|'MRI', stats). Pure function so it can be unit-tested."""
    a = np.asarray(arr, dtype=np.float32).ravel()
    vmin = float(a.min())
    p01 = float(np.percentile(a, 1))
    frac_air = float(np.mean(a < air_thresh))       # CT air lives well below -500 HU
    frac_neg = float(np.mean(a < -100.0))
    is_ct = (vmin < air_thresh) and (frac_air > neg_frac_thresh)
    return ("CT" if is_ct else "MRI",
            {"min": vmin, "p01": p01, "frac_air": frac_air, "frac_neg": frac_neg})


def case_id_num(fname):
    m = re.search(r"(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="AMOS imagesTr folder")
    ap.add_argument("--mri-min-id", type=int, default=500)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="restrict to these case stems (e.g. amos_0507); default = all")
    args = ap.parse_args()

    import nibabel as nib
    files = sorted(glob.glob(os.path.join(args.images_dir, "*.nii*")))
    if args.ids:
        want = set(args.ids)
        files = [f for f in files if any(os.path.basename(f).startswith(s) for s in want)]
    if not files:
        raise SystemExit(f"no NIfTI files matched under {args.images_dir}")

    contamination, lost_mri, rows = [], [], []
    for f in files:
        cid = case_id_num(f)
        id_says = "MRI" if cid >= args.mri_min_id else "CT"
        arr = nib.load(f).get_fdata(dtype=np.float32)
        pix_says, st = classify_modality(arr)
        agree = (id_says == pix_says)
        rows.append((os.path.basename(f), cid, id_says, pix_says, st, agree))
        if id_says == "MRI" and pix_says == "CT":
            contamination.append(os.path.basename(f))
        if id_says == "CT" and pix_says == "MRI":
            lost_mri.append(os.path.basename(f))

    print(f"{'file':28s} {'id':>6s} {'id_says':>8s} {'pixels':>7s} "
          f"{'min':>9s} {'frac_air':>9s}  ok")
    for name, cid, ids, px, st, ok in rows:
        print(f"{name:28s} {cid:6d} {ids:>8s} {px:>7s} "
              f"{st['min']:9.1f} {st['frac_air']:9.4f}  {'y' if ok else 'NO'}")

    kept = [r for r in rows if r[2] == "MRI"]
    print(f"\nfiles checked: {len(rows)}   kept as MRI by id-filter: {len(kept)}")
    if contamination:
        print(f"CONTAMINATION: {len(contamination)} CT case(s) kept as MRI -> {contamination}")
    if lost_mri:
        print(f"LOST MRI: {len(lost_mri)} MRI case(s) dropped as CT -> {lost_mri}")
    if not contamination and not lost_mri:
        print("CLEAN: pixel modality agrees with id-threshold for every case.")

    raise SystemExit(1 if contamination else 0)


if __name__ == "__main__":
    main()
