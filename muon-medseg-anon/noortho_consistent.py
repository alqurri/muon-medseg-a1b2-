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

def load(dir, opt, lr, dataset, aug="full", expect_nc=None, expect_frac=1.0,
         expect_aux_lr=None, require_aux=False):
    best = {}
    for f in glob.glob(os.path.join(dir, "*.json")):
        b = os.path.basename(f)
        if b in ("best_lrs.json","best_lrs_light.json","summary.json"): continue
        try: r = json.load(open(f))
        except: continue
        c = r.get("config", {})
        if c.get("model")!="unet" or c.get("optimizer")!=opt or c.get("aug")!=aug: continue
        if abs(c.get("lr",-1)-lr)>1e-9: continue
        # skip tuning runs; require final runs at the corrected protocol
        role = c.get("run_role")
        if role == "tuning": continue
        if role is None and c.get("epochs",0) < 200: continue
        if c.get("_protocol_version") != 2:
            raise ValueError(f"{f}: protocol {c.get('_protocol_version')} != 2 "
                             f"(old/uncorrected run; rerun before analysis)")
        if c.get("epochs") != 200:
            raise ValueError(f"{f}: epochs {c.get('epochs')} != 200")
        # dataset identity + required metadata
        _dsroot = str(dataset).split("_")[0]
        _cds = str(c.get("dataset","")).split("_")[0]
        if _cds and _cds != _dsroot:
            raise ValueError(f"{f}: dataset '{c.get('dataset')}' != requested '{dataset}'")
        for _req in ("dataset", "n_classes", "seed"):
            if _req not in c:
                raise ValueError(f"{f}: missing required metadata '{_req}'")
        # class-count VALUE, training fraction, auxiliary LR
        if expect_nc is not None and c.get("n_classes") != expect_nc:
            raise ValueError(f"{f}: n_classes {c.get('n_classes')} != expected {expect_nc}")
        if abs(float(c.get("train_frac", 1.0)) - float(expect_frac)) > 1e-9:
            raise ValueError(f"{f}: train_frac {c.get('train_frac')} != expected {expect_frac}")
        if require_aux:
            if c.get("adamw_lr_for_hybrid") is None:
                raise ValueError(f"{f}: missing required adamw_lr_for_hybrid metadata")
            if expect_aux_lr is not None and \
               abs(float(c["adamw_lr_for_hybrid"]) - float(expect_aux_lr)) > 1e-9:
                raise ValueError(f"{f}: adamw_lr_for_hybrid {c['adamw_lr_for_hybrid']} "
                                 f"!= expected {expect_aux_lr}")
        s=c["seed"]; mt=os.path.getmtime(f)
        if s not in best or mt>best[s][0]: best[s]=(mt,r)
    out={}
    for s,(mt,r) in best.items():
        pm=defaultdict(list)
        for cid,d in zip(r["test"]["cases"], r["test"]["dice_per_case"]):
            pm[patient_of(cid,dataset)].append(np.nanmean(d))
        out[s]={p:float(np.nanmean(v)) for p,v in pm.items()}
    return out

def boot_ci(A,B,seeds,patients,n=10000,seed=0):
    rng=np.random.default_rng(seed); S,P=len(seeds),len(patients); out=np.empty(n)
    for b in range(n):
        bs=rng.integers(0,S,S); bp=rng.integers(0,P,P); v=[]
        for si in bs:
            s=seeds[si]
            v.append(np.mean([B[s][patients[pi]]-A[s][patients[pi]] for pi in bp]))
        out[b]=np.mean(v)*100
    return np.percentile(out,2.5), np.percentile(out,97.5)

def main(a):
    # Expected class count per dataset (Synapse full = 9; others below). If
    # --n-classes is given it overrides; otherwise it is derived so the value
    # check is NEVER silently skipped.
    _NC = {"synapse": 9, "flare": 14, "amos": 16, "brats": 4,
           "acdc": 4, "camus": 4, "isic": 2}
    if a.n_classes is None:
        a.n_classes = _NC.get(str(a.dataset).split("_")[0])
        if a.n_classes is None:
            raise SystemExit(f"unknown dataset '{a.dataset}': pass --n-classes explicitly")
    A_m=load(a.muon_dir,"adamw",a.adamw_lr,a.dataset,
             expect_nc=a.n_classes,expect_frac=a.train_frac)
    M=load(a.muon_dir,"muon",a.muon_lr,a.dataset,
           expect_nc=a.n_classes,expect_frac=a.train_frac,
           expect_aux_lr=a.aux_lr,require_aux=True)
    NS=load(a.noortho_dir,"muon_noortho",a.nsopt_lr,a.dataset,
            expect_nc=a.n_classes,expect_frac=a.train_frac,
            expect_aux_lr=a.aux_lr,require_aux=True)
    seeds=sorted(set(A_m)&set(M)&set(NS))
    if not seeds:
        print(f"{a.dataset}: no common seeds (A {sorted(A_m)}, M {sorted(M)}, NS {sorted(NS)})"); return
    pats=sorted(set.intersection(*[set(A_m[s]) for s in seeds],*[set(M[s]) for s in seeds],*[set(NS[s]) for s in seeds]))
    def md(D): return np.mean([np.mean([D[s][p] for p in pats]) for s in seeds])*100
    adamw,muon,ns=md(A_m),md(M),md(NS)
    d_ma,d_na,d_nm=muon-adamw,ns-adamw,ns-muon
    ci_ma=boot_ci(A_m,M,seeds,pats); ci_na=boot_ci(A_m,NS,seeds,pats); ci_nm=boot_ci(M,NS,seeds,pats)
    print(f"=== {a.dataset} (matched: {len(seeds)} seeds, {len(pats)} patients) ===")
    print(f"  AdamW {adamw:.2f}  Muon {muon:.2f}  Muon-NS {ns:.2f}")
    print(f"  Muon-AdamW {d_ma:+.2f} CI [{ci_ma[0]:+.2f},{ci_ma[1]:+.2f}]")
    print(f"  NS-AdamW   {d_na:+.2f} CI [{ci_na[0]:+.2f},{ci_na[1]:+.2f}]")
    print(f"  NS-Muon    {d_nm:+.2f} CI [{ci_nm[0]:+.2f},{ci_nm[1]:+.2f}]")
    print(f"  identity: (NS-AdamW)-(Muon-AdamW)={d_na-d_ma:+.2f} == NS-Muon {d_nm:+.2f} {'OK' if abs((d_na-d_ma)-d_nm)<1e-6 else 'MISMATCH'}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--muon-dir",required=True); ap.add_argument("--noortho-dir",required=True)
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--muon-lr",type=float,required=True); ap.add_argument("--adamw-lr",type=float,required=True)
    ap.add_argument("--nsopt-lr",type=float,required=True)
    ap.add_argument("--n-classes",type=int,default=None,dest="n_classes",
                    help="expected class count; validated against each run")
    ap.add_argument("--train-frac",type=float,default=1.0,dest="train_frac")
    ap.add_argument("--aux-lr",type=float,default=3e-4,dest="aux_lr",
                    help="expected auxiliary AdamW LR for Muon/Muon-NS runs")
    main(ap.parse_args())
