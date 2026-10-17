import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from idrug import *
HP = dict(r=15, alpha=0.03, beta=0.1, gamma=0.001, w=0.05, iters=100)
M = ["iDrug","GRMF-style","GRMF","RLS-Kron","BiRW-simplified","Similarity-kNN","Popularity"]
out={}
for task in (0,1):
    pr=[];nd=[]
    for s in (0,1,2):
        D=make_data(s); pr.append(pair_cv(D,task,M,reps=1,seed=s,**HP)); nd.append(cold_cv(D,task,0,M,ks=(10,),reps=1,seed=s,**HP))
    out[f"pair_{task}"]={m:{k:[float(np.mean([d[m][k] for d in pr])),float(np.std([d[m][k] for d in pr]))] for k in ("auroc","aupr")} for m in M}
    out[f"newdrug_{task}"]={m:[float(np.mean([d[m]["10"] for d in nd])),float(np.std([d[m]["10"] for d in nd]))] for m in M}
    print(task,{m:(round(v["aupr"][0],3),round(out[f"newdrug_{task}"][m][0],3)) for m,v in out[f"pair_{task}"].items()},flush=True)
json.dump(out,open("results_baselines.json","w"),indent=1)
