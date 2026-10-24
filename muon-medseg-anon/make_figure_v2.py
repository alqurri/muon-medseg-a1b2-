import argparse, glob, json, re, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = [
    ("brats",   "results_brats_v2",     "BraTS"),
    ("amos",    "results_amos_npz_v2",  "AMOS22"),
    ("synapse", "results_syn_v3",       "Synapse"),
    ("flare",   "results_flare_npz_v2", "FLARE22"),
    ("isic",    "results_isic_v2",      "ISIC"),
    ("camus",   "results_camus_v2",     "CAMUS"),
    ("acdc",    "results_acdc_v2",      "ACDC"),
]

def pat(cid, ds):
    c = str(cid)
    if ds in ("acdc", "camus"):
        c = re.sub(r"[_\-](ed|es|ED|ES)$", "", c)
        c = re.sub(r"[_\-]?frame\d+$", "", c, flags=re.I)
        c = re.sub(r"[_\-](0|1)$", "", c)
    return c

def sel_lr(d, fname, key):
    p = os.path.join(d, fname)
    if os.path.exists(p):
        try: return json.load(open(p)).get(key, {}).get("lr")
        except: return None
    return None

def spm(d, opt, lr, ds, aug):
    if lr is None: return {}
    best = {}
    for f in glob.glob(f"{d}/*.json"):
        b = os.path.basename(f)
        if "noortho" in b or b in ("best_lrs.json","best_lrs_light.json","summary.json"): continue
        try: r = json.load(open(f))
        except: continue
        c = r.get("config", {})
        if c.get("model")!="unet" or c.get("optimizer")!=opt or c.get("aug")!=aug: continue
        if abs(c.get("lr",-1)-lr)>1e-9 or c.get("epochs",0)<150: continue
        if c.get("run_role")=="tuning": continue
        s=c["seed"]; mt=os.path.getmtime(f)
        if s not in best or mt>best[s][0]: best[s]=(mt,r)
    out={}
    for s,(mt,r) in best.items():
        pm=defaultdict(list)
        for cid,dc in zip(r["test"]["cases"], r["test"]["dice_per_case"]):
            pm[pat(cid,ds)].append(np.nanmean(dc))
        out[s]={p:float(np.nanmean(v)) for p,v in pm.items()}
    return out

def dci(d, ds, aug):
    lf = "best_lrs_light.json" if (aug=="light" and os.path.exists(f"{d}/best_lrs_light.json")
         and json.load(open(f"{d}/best_lrs_light.json")).get("unet/muon")) else "best_lrs.json"
    mlr=sel_lr(d,lf,"unet/muon"); alr=sel_lr(d,lf,"unet/adamw")
    M=spm(d,"muon",mlr,ds,aug); A=spm(d,"adamw",alr,ds,aug)
    seeds=sorted(set(M)&set(A))
    if not seeds: return (0,0,0,0)
    pats=sorted(set.intersection(*[set(M[s]) for s in seeds],*[set(A[s]) for s in seeds]))
    per=[np.mean([M[s][p]-A[s][p] for p in pats])*100 for s in seeds]
    base=np.mean([np.mean([A[s][p] for p in pats]) for s in seeds])*100
    rng=np.random.default_rng(0); S,P=len(seeds),len(pats); bt=np.empty(10000)
    for b in range(10000):
        bs=rng.integers(0,S,S); bp=rng.integers(0,P,P); v=[]
        for si in bs:
            s=seeds[si]; v.append(np.mean([M[s][pats[pi]]-A[s][pats[pi]] for pi in bp]))
        bt[b]=np.mean(v)*100
    lo,hi=np.percentile(bt,[2.5,97.5])
    return (float(np.mean(per)),float(lo),float(hi),float(base))

def main(out):
    names=[]; H=[]; L=[]
    for ds,d,name in DS:
        h=dci(d,ds,"full"); l=dci(d,ds,"light")
        names.append(name); H.append(h); L.append(l)
        print(f"{name}: heavy {h[0]:+.2f} [{h[1]:+.2f},{h[2]:+.2f}] base {h[3]:.1f} | light {l[0]:+.2f} [{l[1]:+.2f},{l[2]:+.2f}]")
    x=np.arange(len(names)); w=0.38
    fig,ax=plt.subplots(figsize=(9,4.2))
    ax.bar(x-w/2,[h[0] for h in H],w,yerr=[[h[0]-h[1] for h in H],[h[2]-h[0] for h in H]],capsize=3,label="Heavy aug",color="#2c7fb8")
    ax.bar(x+w/2,[l[0] for l in L],w,yerr=[[l[0]-l[1] for l in L],[l[2]-l[0] for l in L]],capsize=3,label="Light aug",color="#f03b20",alpha=0.85)
    ax.axhline(0,color="k",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Muon $-$ AdamW Dice (points)")
    ax.set_title("Muon's U-Net benefit over AdamW by dataset (ordered by increasing baseline)")
    ax.legend(frameon=False); plt.tight_layout(); plt.savefig(out,bbox_inches="tight"); print("wrote",out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="benefit_figure.pdf"); a=ap.parse_args()
    main(a.out)
