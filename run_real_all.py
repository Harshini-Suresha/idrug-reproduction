"""Real-data version of run_all.py: benchmark + sweeps on the authors' CTD/DrugBank .mat data
and on their goldDataset (Gottlieb PREDICT sizes).

Clone the data first (not shipped here -- it is ~290 MB):
    git clone --depth 1 https://github.com/Case-esaC/iDrug iDrug
Run:
    python run_real_all.py --mat iDrug          # full run, 5-fold, hours on one CPU core
    python run_real_all.py --mat iDrug --quick  # 2-fold smoke test
Outputs: results_real.json (main benchmark), sweeps_real.json (robustness sweeps).
Protocol notes: pair_cv hides 1/5 of known links per fold and samples an equal number of
unknown pairs as negatives; cold_cv hides every link of a fold of drugs (new drug) or of
diseases/targets (new disease/target) and reports precision@k. Seeds are fixed at 0.
"""
import argparse, json, time, copy, warnings; warnings.filterwarnings("ignore")
import numpy as np
from idrug import pair_cv, cold_cv
from loaders import load_mat

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True, help="clone of github.com/Case-esaC/iDrug (or a folder with the same files)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--out", default="results_real.json")
ap.add_argument("--sweeps-out", default="sweeps_real.json")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()

FOLDS = 2 if a.quick else a.folds
ITERS = 3 if (a.quick and a.iters == 200) else a.iters
M = ["iDrug", "iDrug-official", "GRMF-style", "GRMF", "RLS-Kron", "BiRW-simplified", "Similarity-kNN", "Popularity"]
HP = dict(r=90, r2=70, alpha=0.001, beta=0.001, gamma=0.001, w=0.3, iters=ITERS)  # the paper's pair-prediction settings
TASK = {0: "drug_disease", 1: "drug_target"}


def save(obj, path):
    json.dump(obj, open(path, "w"), indent=1)


def degrade(D, task, f_links=1.0, f_anchor=1.0, seed=0):
    """Keep a fraction f_links of the OTHER domain's links and f_anchor of the anchor links (same as run_all.py)."""
    rng = np.random.default_rng(seed); D = copy.deepcopy(D)
    ok = "X2" if task == 0 else "X1"
    X = D[ok]; pos = np.argwhere(X > 0); drop = pos[rng.random(len(pos)) > f_links]
    X[drop[:, 0], drop[:, 1]] = 0
    keep = rng.random(len(D["anc1"])) < f_anchor
    D["anc1"] = D["anc1"][keep]; D["anc2"] = D["anc2"][keep]
    return D


t0 = time.time()
out = {"hyperparameters": HP, "folds": FOLDS, "seed": 0,
       "protocol": "pair_cv: hide 1/5 of known links, negatives 1:1 from unknown; cold_cv: entire rows/cols held out"}
datasets = {"ctd": load_mat(a.mat), "gold": load_mat(a.mat.rstrip("/") + "/goldDataset")}

# ------------------------------------------------------------------ main benchmark, both datasets
for name, D in datasets.items():
    for task in (0, 1):
        t = TASK[task]
        out[f"{name}_pair_{t}"] = pair_cv(D, task, M, folds=FOLDS, reps=1, seed=0, stats=True, **HP)
        print(f"{name} pair {t} done ({time.time()-t0:.0f}s)", {m: round(v["aupr"][0], 3) for m, v in out[f"{name}_pair_{t}"].items()}, flush=True)
        save(out, a.out)
        out[f"{name}_newdrug_{t}"] = cold_cv(D, task, 0, M, folds=FOLDS, reps=1, seed=0, stats=True, **HP)
        out[f"{name}_newcol_{t}"] = cold_cv(D, task, 1, M, folds=FOLDS, reps=1, seed=0, stats=True, **HP)
        print(f"{name} cold {t} done ({time.time()-t0:.0f}s)", {m: round(v["10"][0], 3) for m, v in out[f"{name}_newdrug_{t}"].items()}, flush=True)
        save(out, a.out)

# ------------------------------------------------------------------ paper check: their documented new-drug config
pc = cold_cv(datasets["ctd"], 0, 0, ["iDrug-official"], ks=(20,), folds=FOLDS, reps=1, seed=0, stats=True,
             r=70, r2=70, alpha=0.01, beta=0.01, gamma=0.01, w=0.3, iters=ITERS)
out["paper_check_newdrug_official_p20"] = pc
print("paper check (official, rank 70/70, 0.01, k=20):", pc, flush=True)
save(out, a.out)

# ------------------------------------------------------------------ sweeps on the full CTD/DrugBank data, drug-disease, my iDrug
SHP = dict(HP); SHP["iters"] = 100
sw = {"settings": {**SHP, "folds": FOLDS, "task": "drug_disease", "method": "iDrug (my implementation)"}}
sw["ref_grmf_style"] = {"pair_aupr": out["ctd_pair_drug_disease"]["GRMF-style"]["aupr"],
                        "newdrug_p10": out["ctd_newdrug_drug_disease"]["GRMF-style"]["10"]}
betas = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.3]
sw["beta_sweep"] = {"betas": betas, "aupr": []}
for b in betas:
    sw["beta_sweep"]["aupr"].append(pair_cv(datasets["ctd"], 0, ["iDrug"], folds=FOLDS, reps=1, seed=0, stats=True, **dict(SHP, beta=b))["iDrug"]["aupr"])
    print("beta", b, "->", sw["beta_sweep"]["aupr"][-1], flush=True); save(sw, a.sweeps_out)

alphas = [0.0001, 0.001, 0.01, 0.1]
grid = []
for al in alphas:
    row = []
    for b in [0.0, 0.001, 0.01, 0.1]:
        row.append(pair_cv(datasets["ctd"], 0, ["iDrug"], folds=FOLDS, reps=1, seed=0, stats=True, **dict(SHP, alpha=al, beta=b))["iDrug"]["aupr"])
    grid.append(row); print("alpha", al, "->", [round(x[0], 3) for x in row], flush=True); save(sw, a.sweeps_out)
sw["alpha_beta_grid"] = {"alphas": alphas, "betas": [0.0, 0.001, 0.01, 0.1], "aupr": grid}

ws = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
sw["w_sweep"] = {"w": ws, "aupr": []}
for wv in ws:
    sw["w_sweep"]["aupr"].append(pair_cv(datasets["ctd"], 0, ["iDrug"], folds=FOLDS, reps=1, seed=0, stats=True, **dict(SHP, w=wv))["iDrug"]["aupr"])
    print("w", wv, "->", sw["w_sweep"]["aupr"][-1], flush=True); save(sw, a.sweeps_out)

levels = [0.0, 0.25, 0.5, 0.75, 1.0]
for kind in ("links", "anchors"):
    rows = {"levels": levels, "iDrug_pair_aupr": [], "iDrug_newdrug_p10": []}
    for lv in levels:
        D = degrade(datasets["ctd"], 0, f_links=lv if kind == "links" else 1.0, f_anchor=lv if kind == "anchors" else 1.0, seed=0)
        rows["iDrug_pair_aupr"].append(pair_cv(D, 0, ["iDrug"], folds=FOLDS, reps=1, seed=0, stats=True, **SHP)["iDrug"]["aupr"])
        rows["iDrug_newdrug_p10"].append(cold_cv(D, 0, 0, ["iDrug"], ks=(10,), folds=FOLDS, reps=1, seed=0, stats=True, **SHP)["iDrug"]["10"])
    sw[f"withhold_{kind}"] = rows
    print(f"withhold {kind} done ({time.time()-t0:.0f}s)", flush=True); save(sw, a.sweeps_out)

print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
