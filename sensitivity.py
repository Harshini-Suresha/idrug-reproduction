"""alpha x beta grid, and w sweep (paper S2-S5 Figs analogue). AUPR, drug-disease pair prediction, synthetic_small-like data."""
import json, warnings; warnings.filterwarnings("ignore")
from idrug import make_data, pair_cv
D = make_data(0); base = dict(r=15, alpha=0.03, beta=0.1, gamma=0.001, w=0.05, iters=100)
vals = [0.001, 0.01, 0.03, 0.1, 0.3]
grid = [[pair_cv(D, 0, ["iDrug"], folds=3, reps=1, **dict(base, alpha=a, beta=b))["iDrug"]["aupr"] for b in vals] for a in vals]
ws = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
wsw = [pair_cv(D, 0, ["iDrug"], folds=3, reps=1, **dict(base, w=w))["iDrug"]["aupr"] for w in ws]
out = {"values": vals, "alpha_x_beta_aupr": grid, "w": ws, "w_aupr": wsw}
json.dump(out, open("sensitivity.json", "w"), indent=1)
print("alpha(rows) x beta(cols)"); [print(v, [round(x, 3) for x in row]) for v, row in zip(vals, grid)]
print("w", [round(x, 3) for x in wsw])
