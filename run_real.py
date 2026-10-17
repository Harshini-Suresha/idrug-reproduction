"""Run the same experiments as run_all.py on the real data in data/.
Usage:  python run_real.py            (paper-like ranks)   |   python run_real.py --rank 90
NOTE: the real matrices are ~1000 x 4000, so use CPU patience; lower --iters for a quick pass."""
import argparse, json, warnings; warnings.filterwarnings("ignore")
from idrug import pair_cv, cold_cv
from loaders import load_real, load_mat, is_placeholder

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--out", default="results_real.json"); ap.add_argument("--mat", default=None, help="folder with the authors' .mat files (a clone of the official repo, or its goldDataset/)")
ap.add_argument("--r2", type=int, default=None); ap.add_argument("--official", action="store_true", help="also run the port of the authors' MATLAB solver")
ap.add_argument("--baselines", action="store_true", help="also run GRMF, RLS-Kron and simplified bi-random walk (my implementations)")
ap.add_argument("--rank", type=int, default=70); ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--alpha", type=float, default=0.01); ap.add_argument("--beta", type=float, default=0.01)
ap.add_argument("--gamma", type=float, default=0.01); ap.add_argument("--w", type=float, default=0.3)
a = ap.parse_args()
if not a.mat and is_placeholder(a.data):
    raise SystemExit("data/ still holds the placeholder files. Read data/README.md and drop the real CSVs in.")
HP = dict(r=a.rank, alpha=a.alpha, beta=a.beta, gamma=a.gamma, w=a.w, iters=a.iters)
M = ["iDrug", "GRMF-style", "Similarity-kNN", "Popularity"] + (["iDrug-official"] if a.official else []) + (["GRMF", "RLS-Kron", "BiRW-simplified"] if a.baselines else [])
if a.r2: HP["r2"] = a.r2
D = load_mat(a.mat) if a.mat else load_real(a.data); out = {"hyperparameters": HP}
for task, name in ((0, "drug_disease"), (1, "drug_target")):
    out[f"pair_{name}"] = pair_cv(D, task, M, folds=a.folds, reps=a.reps, **HP); print(name, "pair", out[f"pair_{name}"])
    out[f"newdrug_{name}"] = cold_cv(D, task, 0, M, folds=a.folds, reps=1, **HP)
    out[f"newcol_{name}"] = cold_cv(D, task, 1, M, folds=a.folds, reps=1, **HP)
json.dump(out, open(a.out, "w"), indent=1)
print("saved", a.out)
