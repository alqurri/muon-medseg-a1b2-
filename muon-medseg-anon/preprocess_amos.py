"""One-time AMOS-MRI preprocessing to uncompressed .npy slices (fast path).
Preserves the exact split of data_amos.build_amos_datasets.

    python preprocess_amos.py --root ./data/AMOS22 --out ./data/AMOS22_npz --mri-min-id 500
Then run with --dataset amos_npz.
"""
import argparse, json, os, numpy as np
from data_amos import (_find_pairs, _load_nii, _znorm, _case_id, _split,
                       SLICE_STRIDE, MAX_SLICES_PER_VOL)

def _extract(pairs, idxs, out, split, stride, mx, seed=0):
    rng=np.random.default_rng(seed); d=os.path.join(out,split); os.makedirs(d,exist_ok=True); man=[]
    for vi in idxs:
        ip,lp=pairs[vi]; cid=_case_id(ip); lab=_load_nii(lp)
        fg=np.where((lab>0).any(axis=(0,1)))[0][::stride]
        if len(fg)==0: continue
        if len(fg)>mx: fg=rng.choice(fg,mx,replace=False)
        img=_load_nii(ip)
        for z in fg:
            ai=_znorm(img[:,:,int(z)]).astype(np.float32)
            al=np.rint(lab[:,:,int(z)]).astype(np.float32)
            fn=f"{cid}_z{int(z):03d}.npy"; np.save(os.path.join(d,fn),np.stack([ai,al]))
            man.append({"file":os.path.join(split,fn),"case":cid})
        print(f"  {split}: {cid} -> {len(fg)}",flush=True)
    return man

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--stride",type=int,default=SLICE_STRIDE)
    ap.add_argument("--max-per-vol",type=int,default=MAX_SLICES_PER_VOL)
    ap.add_argument("--val-frac",type=float,default=0.1); ap.add_argument("--test-frac",type=float,default=0.2)
    ap.add_argument("--mri-min-id",type=int,default=500)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    pairs=_find_pairs(a.root,a.mri_min_id)
    tr_i,va_i,te_i=_split(pairs,a.val_frac,a.test_frac)
    print(f"{len(pairs)} MRI vols -> train {len(tr_i)} / val {len(va_i)} / test {len(te_i)}")
    tr=_extract(pairs,tr_i,a.out,"train",a.stride,a.max_per_vol)
    va=_extract(pairs,va_i,a.out,"val",a.stride,a.max_per_vol)
    tv=[{"image":pairs[i][0],"label":pairs[i][1],"case":_case_id(pairs[i][0])} for i in te_i]
    json.dump({"train":tr,"val":va,"test_vols":tv},open(os.path.join(a.out,"manifest.json"),"w"))
    print(f"wrote {len(tr)} train + {len(va)} val slices, {len(tv)} test vols to {a.out}")

if __name__=="__main__": main()
