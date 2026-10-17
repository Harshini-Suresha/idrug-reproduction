"""Fast subset of run_real_all.py's sweeps (same protocol, iters=80, trimmed grids, pair-only withhold).
Writes sweeps_real.json. Used to parallelise the long full run; results are labeled "trimmed" in the JSON.
"""
import json, time, copy, warnings; warnings.filterwarnings("ignore")
import numpy as np
from idrug import pair_cv
from loaders import load_mat

MAT = "/tmp/iDrug_official"
FOLDS = 5
SHP = dict(r=90, alpha=0.001, beta=0.001, gamma=0.001, w=0.3, iters=80)


def save(obj):
    json.dump(obj, open("sweeps_real.json", "w"), indent=1)


def degrade(D, f_links=1.0, f_anchor=1.0, seed=0):
    rng = np.random.default_rng(seed); D = copy.deepcopy(D)
    X = D["X2"]; pos = np.argwhere(X > 0); drop = pos[rng.random(len(pos)) > f_links]
    X[drop[:, 0], drop[:, 1]] = 0
    keep = rng.random(len(D["anc1"])) < f_anchor
    D["anc1"] = D["anc1"][keep]; D["anc2"] = D["anc2"][keep]
    return D


t0 = time.time()
D = load_mat(MAT)
sw = {"settings": {**SHP, "folds": FOLDS, "task": "drug_disease", "method": "iDrug (my implementation)",
                    "trimmed": True,
                    "note": "trimmed grids, iters=80; withhold is pair AUPR only (run_real_all.py has the full version)"}}
try:
    res = json.load(open("results_real.json"))
    g = res["ctd_pair_drug_disease"]["GRMF-style"]["aupr"]
    sw["ref_grmf_style"] = {"pair_aupr": g, "newdrug_p10": None}
except Exception:
    sw["ref_grmf_style"] = None

run = lambda **kw: pair_cv(D, 0, ["iDrug"], folds=FOLDS, reps=1, seed=0, stats=True, **{**SHP, **kw})["iDrug"]["aupr"]

betas = [0.0, 0.001, 0.01, 0.1, 0.3]
sw["beta_sweep"] = {"betas": betas, "aupr": []}
for b in betas:
    sw["beta_sweep"]["aupr"].append(run(beta=b)); print("beta", b, "->", sw["beta_sweep"]["aupr"][-1], f"({time.time()-t0:.0f}s)", flush=True); save(sw)

alphas = [0.0001, 0.001, 0.01]
sw["alpha_beta_grid"] = {"alphas": alphas, "betas": [0.0, 0.001, 0.01], "aupr": []}
for al in alphas:
    row = []
    for b in sw["alpha_beta_grid"]["betas"]:
        row.append(run(alpha=al, beta=b))
    sw["alpha_beta_grid"]["aupr"].append(row)
    print("alpha", al, "->", [round(x[0], 3) for x in row], f"({time.time()-t0:.0f}s)", flush=True); save(sw)

ws = [0.05, 0.3, 0.5]
sw["w_sweep"] = {"w": ws, "aupr": []}
for wv in ws:
    sw["w_sweep"]["aupr"].append(run(w=wv)); print("w", wv, "->", sw["w_sweep"]["aupr"][-1], f"({time.time()-t0:.0f}s)", flush=True); save(sw)

levels = [0.0, 0.5, 1.0]
for kind in ("links", "anchors"):
    rows = {"levels": levels, "iDrug_pair_aupr": []}
    for lv in levels:
        Dd = degrade(D, f_links=lv if kind == "links" else 1.0, f_anchor=lv if kind == "anchors" else 1.0, seed=0)
        rows["iDrug_pair_aupr"].append(pair_cv(Dd, 0, ["iDrug"], folds=FOLDS, reps=1, seed=0, stats=True, **SHP)["iDrug"]["aupr"])
    sw[f"withhold_{kind}"] = rows
    print(f"withhold {kind} done ({time.time()-t0:.0f}s)", flush=True); save(sw)

print(f"SWEEPS DONE in {(time.time()-t0)/60:.1f} min", flush=True)
