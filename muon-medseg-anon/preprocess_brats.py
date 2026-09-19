"""
One-time BraTS 2020 preprocessing: extract z-scored foreground FLAIR slices to
uncompressed .npy so training skips NIfTI decompression per step (same fix as
FLARE). Preserves EXACTLY the split of data_brats.build_brats_datasets.

Run ONCE:
    python preprocess_brats.py --root ./data/BraTS2020 \
        --out ./data/BraTS2020_npz
Then run with --dataset brats_npz.
"""
import argparse, json, os
import numpy as np
from data_brats import (_find_pairs, _load_nii, _remap_label, _znorm_brain,
                        _case_id, _split, SLICE_STRIDE, MAX_SLICES_PER_VOL)


def _extract(pairs, indices, out_dir, split, stride, max_per_vol, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(out_dir, split); os.makedirs(d, exist_ok=True)
    man = []
    for vi in indices:
        ip, lp = pairs[vi]; cid = _case_id(ip)
        lab_full = _load_nii(lp)
        fg = np.where((lab_full > 0).any(axis=(0, 1)))[0][::stride]
        if len(fg) == 0:
            continue
        if len(fg) > max_per_vol:
            fg = rng.choice(fg, max_per_vol, replace=False)
        img = _load_nii(ip)
        for z in fg:
            arr_img = _znorm_brain(img[:, :, int(z)]).astype(np.float32)
            arr_lab = _remap_label(np.rint(lab_full[:, :, int(z)]).astype(np.int64)).astype(np.float32)
            fn = f"{cid}_z{int(z):03d}.npy"
            np.save(os.path.join(d, fn), np.stack([arr_img, arr_lab]))
            man.append({"file": os.path.join(split, fn), "case": cid})
        print(f"  {split}: {cid} -> {len(fg)} slices", flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=SLICE_STRIDE)
    ap.add_argument("--max-per-vol", type=int, default=MAX_SLICES_PER_VOL)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs = _find_pairs(a.root)
    tr_i, va_i, te_i = _split(pairs, a.val_frac, a.test_frac)
    print(f"{len(pairs)} vols -> train {len(tr_i)} / val {len(va_i)} / test {len(te_i)}")
    tr = _extract(pairs, tr_i, a.out, "train", a.stride, a.max_per_vol)
    va = _extract(pairs, va_i, a.out, "val", a.stride, a.max_per_vol)
    test_vols = [{"flair": pairs[i][0], "seg": pairs[i][1], "case": _case_id(pairs[i][0])}
                 for i in te_i]
    man = {"train": tr, "val": va, "test_vols": test_vols,
           "stride": a.stride, "max_per_vol": a.max_per_vol}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"wrote {len(tr)} train + {len(va)} val slices, {len(test_vols)} test vols to {a.out}")


if __name__ == "__main__":
    main()
