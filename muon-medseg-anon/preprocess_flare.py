"""
One-time FLARE22 preprocessing: extract windowed foreground axial slices to
UNCOMPRESSED .npy so training reads are near-instant instead of re-decompressing
gzipped NIfTI every step. This fixes the FLARE I/O bottleneck (GPU was starving
on nibabel gzip inflation).

Run ONCE (a few minutes):
    python preprocess_flare.py --root ./data/FLARE22Train \
        --out ./data/FLARE22_npz

Then run the study with --dataset flare_npz (identical splits & numbers to
--dataset flare, just faster). Test volumes stay as NIfTI in the manifest and
are loaded once at eval, not per epoch.

CORRECTNESS: reproduces EXACTLY the split and slice selection of
data_flare.build_flare_datasets (same fixed rng seed=0, same val/test fracs,
same stride/max-per-vol, same foreground-slice rule), so flare_npz results are
directly comparable to flare (NIfTI) runs. Split written to manifest.json.
"""
import argparse, json, os
import numpy as np
import nibabel as nib
from data_flare import _find_pairs, _window_ct, _case_id, SLICE_STRIDE, MAX_SLICES_PER_VOL


def _split(pairs, val_frac, test_frac):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    n_test = max(1, int(len(pairs) * test_frac))
    n_val = max(1, int(len(pairs) * val_frac))
    test_i = set(idx[:n_test].tolist())
    val_i = set(idx[n_test:n_test + n_val].tolist())
    train = [i for i in range(len(pairs)) if i not in test_i and i not in val_i]
    return train, sorted(val_i), sorted(test_i)


def _extract(pairs, indices, out_dir, split_name, stride, max_per_vol, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(out_dir, split_name)
    os.makedirs(d, exist_ok=True)
    manifest = []
    for vi in indices:
        ip, lp = pairs[vi]
        cid = _case_id(ip)
        lab = np.asanyarray(nib.load(lp).dataobj)
        fg = np.where((lab > 0).any(axis=(0, 1)))[0]
        if len(fg) == 0:
            continue
        fg = fg[::stride]
        if len(fg) > max_per_vol:
            fg = rng.choice(fg, max_per_vol, replace=False)
        img = np.asanyarray(nib.load(ip).dataobj).astype(np.float32)
        for z in fg:
            arr_img = _window_ct(img[:, :, int(z)]).astype(np.float32)
            arr_lab = np.rint(lab[:, :, int(z)]).astype(np.float32)
            fname = f"{cid}_z{int(z):03d}.npy"
            np.save(os.path.join(d, fname), np.stack([arr_img, arr_lab]))
            manifest.append({"file": os.path.join(split_name, fname), "case": cid})
        print(f"  {split_name}: {cid} -> {len(fg)} slices", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=SLICE_STRIDE)
    ap.add_argument("--max-per-vol", type=int, default=MAX_SLICES_PER_VOL)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pairs = _find_pairs(a.root)
    train_i, val_i, test_i = _split(pairs, a.val_frac, a.test_frac)
    print(f"{len(pairs)} volumes -> train {len(train_i)} / val {len(val_i)} / test {len(test_i)}")

    tr = _extract(pairs, train_i, a.out, "train", a.stride, a.max_per_vol)
    va = _extract(pairs, val_i, a.out, "val", a.stride, a.max_per_vol)
    test_vols = [{"image": pairs[i][0], "label": pairs[i][1], "case": _case_id(pairs[i][0])}
                 for i in test_i]

    manifest = {"train": tr, "val": va, "test_vols": test_vols,
                "stride": a.stride, "max_per_vol": a.max_per_vol,
                "val_frac": a.val_frac, "test_frac": a.test_frac}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"wrote {len(tr)} train + {len(va)} val slices, "
          f"{len(test_vols)} test volumes; manifest.json in {a.out}")


if __name__ == "__main__":
    main()
