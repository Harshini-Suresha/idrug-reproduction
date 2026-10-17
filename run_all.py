import json, time, copy, warnings; warnings.filterwarnings("ignore")
import numpy as np
from idrug import *
HP = dict(r=15, alpha=0.03, beta=0.1, gamma=0.001, w=0.05, iters=100)
M = ["iDrug","GRMF-style","Similarity-kNN","Popularity"]
SEEDS = [0,1,2]
TASK = {0:"Drug-disease", 1:"Drug-target"}

def agg(list_of_dicts):
    """list over seeds of {method:{metric:val}} -> {method:{metric:[mean,std]}}"""
    out={}
    for m in list_of_dicts[0]:
        out[m]={}
        for k in list_of_dicts[0][m]:
            v=[d[m][k] for d in list_of_dicts]; out[m][k]=[float(np.mean(v)),float(np.std(v))]
    return out

def degrade(D, task, f_links=1.0, f_anchor=1.0, seed=0):
    """Withhold part of the OTHER domain's links and/or part of the anchor (shared-drug) links."""
    rng=np.random.default_rng(seed); D=copy.deepcopy(D)
    ok = "X2" if task==0 else "X1"
    X=D[ok]; pos=np.argwhere(X>0); drop=pos[rng.random(len(pos))>f_links]
    X[drop[:,0],drop[:,1]]=0
    keep=rng.random(len(D["anc1"]))<f_anchor
    D["anc1"]=D["anc1"][keep]; D["anc2"]=D["anc2"][keep]
    return D

res={"hyperparameters":HP,"seeds":SEEDS}
t0=time.time()
# 1+2: main benchmark
for task in (0,1):
    pr=[];nd=[];nt=[]
    for s in SEEDS:
        D=make_data(s)
        pr.append(pair_cv(D,task,M,reps=2,seed=s,**HP))
        nd.append(cold_cv(D,task,0,M,reps=1,seed=s,**HP))
        nt.append(cold_cv(D,task,1,M,reps=1,seed=s,**HP))
    res[f"pair_{task}"]=agg(pr); res[f"newdrug_{task}"]=agg(nd); res[f"newcol_{task}"]=agg(nt)
    print("done main",task,time.time()-t0,flush=True)
# 3a: withhold other-domain links;  3b: withhold anchor links
levels=[0.0,0.25,0.5,0.75,1.0]
for task in (0,1):
    for kind in ("links","anchors"):
        rows={"levels":levels,"iDrug_pair_aupr":[],"iDrug_newdrug_p10":[]}
        for lv in levels:
            pa=[];nd=[]
            for s in SEEDS:
                D=degrade(make_data(s),task,f_links=lv if kind=="links" else 1.0,f_anchor=lv if kind=="anchors" else 1.0,seed=s)
                pa.append(pair_cv(D,task,["iDrug"],reps=1,seed=s,**HP)["iDrug"]["aupr"])
                nd.append(cold_cv(D,task,0,["iDrug"],ks=(10,),reps=1,seed=s,**HP)["iDrug"]["10"])
            rows["iDrug_pair_aupr"].append([float(np.mean(pa)),float(np.std(pa))])
            rows["iDrug_newdrug_p10"].append([float(np.mean(nd)),float(np.std(nd))])
        ref_p=[];ref_n=[]
        for s in SEEDS:
            D=make_data(s)
            ref_p.append(pair_cv(D,task,["GRMF-style"],reps=1,seed=s,**HP)["GRMF-style"]["aupr"])
            ref_n.append(cold_cv(D,task,0,["GRMF-style"],ks=(10,),reps=1,seed=s,**HP)["GRMF-style"]["10"])
        rows["GRMF_pair_aupr"]=float(np.mean(ref_p)); rows["GRMF_newdrug_p10"]=float(np.mean(ref_n))
        res[f"withhold_{kind}_{task}"]=rows
        print("done withhold",task,kind,time.time()-t0,flush=True)
# 4: beta sweep
betas=[0.0,0.01,0.03,0.1,0.3,1.0]
sw=[]
for b in betas:
    v=[pair_cv(make_data(s),0,["iDrug"],reps=1,seed=s,**dict(HP,beta=b))["iDrug"]["aupr"] for s in SEEDS]
    sw.append([float(np.mean(v)),float(np.std(v))])
res["beta_sweep"]={"betas":betas,"aupr":sw}
# data stats
D=make_data(0)
res["data"]={"drugs_d1":D["X1"].shape[0],"diseases":D["X1"].shape[1],"links_dd":int(D["X1"].sum()),
 "drugs_d2":D["X2"].shape[0],"targets":D["X2"].shape[1],"links_dt":int(D["X2"].sum()),"anchors":int(len(D["anc1"]))}
json.dump(res,open("results.json","w"),indent=1)
print("ALL DONE",time.time()-t0,flush=True)
