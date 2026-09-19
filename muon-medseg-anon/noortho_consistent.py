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

def load(dir, opt, lr, dataset, aug="full"):
    best = {}
    for f in glob.glob(os.path.join(dir, "*.json")):
        b = os.path.basename(f)
        if b in ("best_lrs.json","best_lrs_light.json","summary.json"): continue
        try: r = json.load(open(f))
        except: continue
        c = r.get("config", {})
        if c.get("model")!="unet" or c.get("optimizer")!=opt or c.get("aug")!=aug: continue
        if abs(c.get("lr",-1)-lr)>1e-9 or c.get("epochs",0)<150: continue
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
    A_m=load(a.muon_dir,"adamw",a.adamw_lr,a.dataset)
    M=load(a.muon_dir,"muon",a.muon_lr,a.dataset)
    NS=load(a.noortho_dir,"muon_noortho",a.nsopt_lr,a.dataset)
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
    main(ap.parse_args())
