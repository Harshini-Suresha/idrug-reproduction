import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from idrug import *
HP = dict(r=15, alpha=0.03, beta=0.1, gamma=0.001, w=0.05, iters=100)
out={}
for name,kw in {"rich":dict(),"sparse_task":dict(p_dd=0.07,sim_noise=0.6)}.items():
    rows={}
    for label,beta in [("GRMF-style",0.0),("iDrug b=0.03",0.03),("iDrug b=0.1",0.1)]:
        v=[pair_cv(make_data(s,**kw),0,["iDrug"],reps=1,seed=s,**dict(HP,beta=beta))["iDrug"] for s in (0,1,2)]
        rows[label]={"auroc":[float(np.mean([x["auroc"] for x in v])),float(np.std([x["auroc"] for x in v]))],
                     "aupr":[float(np.mean([x["aupr"] for x in v])),float(np.std([x["aupr"] for x in v]))]}
    D=make_data(0,**kw); rows["links"]=int(D["X1"].sum())
    knn=[pair_cv(make_data(s,**kw),0,["Similarity-kNN"],reps=1,seed=s,**HP)["Similarity-kNN"]["aupr"] for s in (0,1,2)]
    rows["Similarity-kNN"]={"aupr":[float(np.mean(knn)),float(np.std(knn))]}
    out[name]=rows; print(name,json.dumps(rows))
json.dump(out,open("sparse.json","w"),indent=1)
