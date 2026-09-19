import glob, json, os, re
from collections import defaultdict
import numpy as np

DS = {
 "brats":  ("results_brats",   4,  "BraTS"),
 "synapse":("results_synapse", 9,  "Synapse"),
 "acdc":   ("results",         4,  "ACDC"),
 "flare":  ("results_flare",   14, "FLARE22"),
 "isic":   ("results_isic",    2,  "ISIC"),
 "camus":  ("results_camus",   4,  "CAMUS"),
 "amos":   ("results_amos",    16, "AMOS22"),
}
VMDIR = {"acdc":"results_vmunet_acdc", "camus":"results_vmunet_camus", "synapse":"results_vmunet_synapse"}
ORDER = ["amos","flare","synapse","brats","isic","camus","acdc"]

def pat(cid, ds):
    c=str(cid)
    if ds in ("acdc","camus"):
        c=re.sub(r"[_\-](ed|es|ED|ES)$","",c); c=re.sub(r"[_\-]?frame\d+$","",c,flags=re.I); c=re.sub(r"[_\-](0|1)$","",c)
    return c

def sel_lr(d, fname, key):
    p=os.path.join(d,fname)
    if os.path.exists(p):
        try: return json.load(open(p)).get(key,{}).get("lr")
        except: return None
    return None

def load(d, model, opt, lr, ds, aug):
    if lr is None: return {}
    best={}
    for f in glob.glob(os.path.join(d,"*.json")):
        b=os.path.basename(f)
        if "noortho" in b or b in ("best_lrs.json","best_lrs_light.json","summary.json"): continue
        try: r=json.load(open(f))
        except: continue
        c=r.get("config",{})
        if c.get("model")!=model or c.get("optimizer")!=opt or c.get("aug")!=aug: continue
        if abs(c.get("lr",-1)-lr)>1e-9 or c.get("epochs",0)<150: continue
        s=c["seed"]; mt=os.path.getmtime(f)
        if s not in best or mt>best[s][0]: best[s]=(mt,r)
    out={}
    for s,(mt,r) in best.items():
        pm=defaultdict(list)
        for cid,dd in zip(r["test"]["cases"], r["test"]["dice_per_case"]):
            pm[pat(cid,ds)].append(np.nanmean(dd))
        out[s]={p:float(np.nanmean(v)) for p,v in pm.items()}
    return out

def bootci(A,B,seeds,pats,n=10000):
    rng=np.random.default_rng(0); S,P=len(seeds),len(pats); o=np.empty(n)
    for b in range(n):
        bs=rng.integers(0,S,S); bp=rng.integers(0,P,P); v=[]
        for si in bs:
            s=seeds[si]; v.append(np.mean([B[s][pats[pi]]-A[s][pats[pi]] for pi in bp]))
        o[b]=np.mean(v)*100
    return np.percentile(o,[2.5,97.5])

def cell(d, model, ds, aug, lrfile):
    mlr=sel_lr(d, lrfile, f"{model}/muon"); alr=sel_lr(d, lrfile, f"{model}/adamw")
    A=load(d,model,"adamw",alr,ds,aug); M=load(d,model,"muon",mlr,ds,aug)
    seeds=sorted(set(A)&set(M))
    if not seeds: return None
    pats=sorted(set.intersection(*[set(A[s]) for s in seeds],*[set(M[s]) for s in seeds]))
    def md(D): return np.mean([np.mean([D[s][p] for p in pats]) for s in seeds])*100
    a,m=md(A),md(M); per=[np.mean([M[s][p]-A[s][p] for p in pats])*100 for s in seeds]
    lo,hi=bootci(A,M,seeds,pats); k=sum(1 for x in per if x>0)
    return dict(adamw=a,muon=m,delta=m-a,per=per,k=k,n=len(seeds),lo=lo,hi=hi,A=A,M=M,seeds=seeds,pats=pats)

def interaction(h,l):
    seeds=sorted(set(h['seeds'])&set(l['seeds'])); pats=sorted(set(h['pats'])&set(l['pats']))
    per=[]
    for s in seeds:
        dh=np.mean([h['M'][s][p]-h['A'][s][p] for p in pats]); dl=np.mean([l['M'][s][p]-l['A'][s][p] for p in pats])
        per.append((dh-dl)*100)
    rng=np.random.default_rng(0); S,P=len(seeds),len(pats); bt=np.empty(10000)
    for b in range(10000):
        bs=rng.integers(0,S,S); bp=rng.integers(0,P,P); v=[]
        for si in bs:
            s=seeds[si]
            dh=np.mean([h['M'][s][pats[pi]]-h['A'][s][pats[pi]] for pi in bp]); dl=np.mean([l['M'][s][pats[pi]]-l['A'][s][pats[pi]] for pi in bp])
            v.append(dh-dl)
        bt[b]=np.mean(v)*100
    lo,hi=np.percentile(bt,[2.5,97.5])
    return dict(mean=np.mean(per),per=per,lo=lo,hi=hi,sig=(lo>0 or hi<0))

def compute():
    R={}
    for ds,(d,nc,name) in DS.items():
        R[ds]={"name":name,"nc":nc}
        for model in ("unet","transunet"):
            h=cell(d,model,ds,"full","best_lrs.json")
            lf="best_lrs_light.json" if (os.path.exists(os.path.join(d,"best_lrs_light.json")) and json.load(open(os.path.join(d,"best_lrs_light.json")))) else "best_lrs.json"
            l=cell(d,model,ds,"light",lf)
            R[ds][f"{model}_heavy"]=h; R[ds][f"{model}_light"]=l
            R[ds][f"{model}_intx"]=interaction(h,l) if (h and l) else None
        if ds in VMDIR and os.path.isdir(VMDIR[ds]):
            R[ds]["mamba_heavy"]=cell(VMDIR[ds],"vmunet",ds,"full","best_lrs.json")
    return R

def fmt(x, bold=False):
    s=f"{x:+.2f}".replace("-","$-$")
    return f"\\textbf{{{s}}}" if bold else s

def latex(R):
    print("% ===== MAIN TABLE rows =====")
    for ds in ORDER:
        r=R[ds]; name=r["name"]
        for model,label in (("unet","U-Net"),("transunet","TransUNet")):
            for aug in ("heavy","light"):
                c=r.get(f"{model}_{aug}")
                if not c:
                    if aug=="heavy": continue
                    print(f"{name:7} & {aug:5} & {label:9} & \\multicolumn{{4}}{{c}}{{\\emph{{not run}}}} \\\\")
                    continue
                bold=(c['lo']>0 or c['hi']<0)
                print(f"{name:7} & {aug:5} & {label:9} & {c['adamw']:.2f} & {c['muon']:.2f} & {fmt(c['delta'],bold)} & {c['k']}/{c['n']} \\\\")
        if r.get("mamba_heavy"):
            c=r["mamba_heavy"]; bold=(c['lo']>0 or c['hi']<0)
            print(f"{name:7} & heavy & Mamba     & {c['adamw']:.2f} & {c['muon']:.2f} & {fmt(c['delta'],bold)} & {c['k']}/{c['n']} \\\\")
    print("\n% ===== INTERACTION TABLE rows =====")
    for ds in ORDER:
        r=R[ds]; ix=r.get("unet_intx"); h=r.get("unet_heavy"); l=r.get("unet_light")
        if ix is None:
            print(f"{r['name']:7} & \\multicolumn{{4}}{{c}}{{\\emph{{light not run}}}} \\\\"); continue
        star="$^{*}$" if ix['sig'] else ""
        print(f"{r['name']:7} & {fmt(h['delta'])} & {fmt(l['delta'])} & {fmt(ix['mean'],ix['sig'])}{star} & [{fmt(ix['lo'])}, {fmt(ix['hi'])}] \\\\")
    print("\n% ===== PER-SEED INTERACTION rows =====")
    for ds in ORDER:
        r=R[ds]; ix=r.get("unet_intx")
        if ix is None: continue
        ps=" & ".join(fmt(x) for x in ix['per']); sd=np.std(ix['per'],ddof=1)
        print(f"{r['name']:7} & {ps} & ${ix['mean']:+.2f}\\pm{sd:.2f}$ & [{fmt(ix['lo'])},{fmt(ix['hi'])}] \\\\")

def human(R):
    for ds in ORDER:
        r=R[ds]; print(f"\n### {r['name']} (nc {r['nc']}) ###")
        for model in ("unet","transunet","mamba"):
            h=r.get(f"{model}_heavy")
            if h: print(f"  [{model}] HEAVY d {h['delta']:+.2f} {h['k']}/{h['n']} CI[{h['lo']:+.2f},{h['hi']:+.2f}]")
            l=r.get(f"{model}_light")
            if l: print(f"  [{model}] LIGHT d {l['delta']:+.2f} {l['k']}/{l['n']}")
            ix=r.get(f"{model}_intx")
            if ix: print(f"  [{model}] INTX {ix['mean']:+.2f} CI[{ix['lo']:+.2f},{ix['hi']:+.2f}] {'SIG' if ix['sig'] else 'n.s.'}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--latex",action="store_true"); a=ap.parse_args()
    R=compute()
    (latex if a.latex else human)(R)
