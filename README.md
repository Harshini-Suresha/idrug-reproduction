# iDrug reproduction (CPU-only) — real CTD/DrugBank data

**Results website: https://harshini-suresha.github.io/idrug-reproduction/**
(renders everything below from the committed JSONs; source of the page is `docs/index.html`)

Reproduction of *iDrug: Integration of drug repositioning and drug-target prediction via
cross-network embedding* (Chen, Cheng & Li, PLoS Comput Biol 2020), benchmarked on the
**authors' real data**:

- **CTD + DrugBank** (their `.mat` release): drug–disease 1,321 × 3,966 (111,481 links),
  drug–target 946 × 3,610 (10,234 links), 469 shared anchor drugs — paper Table 1 scale.
- **Gottlieb gold standard** (their `goldDataset/`): 593 × 313 and 593 × 1,011, 593 anchors.

The results website (GitHub Pages, from `docs/`) renders everything below from the committed
JSONs. The data itself (~290 MB) is **not shipped** — clone it:

```
git clone --depth 1 https://github.com/Case-esaC/iDrug iDrug
pip install -r requirements.txt

python run_real_all.py --mat iDrug            # full benchmark + sweeps → results_real.json, sweeps_real.json
python run_real_all.py --mat iDrug --quick    # 2-fold, 3-iteration smoke test
python make_site_data.py                      # rebuild docs/idrug-data.js from the JSONs
```

Protocol: 5-fold CV (`pair_cv`: hide 1/5 of known links, negatives 1:1 uniform from unknown)
and cold start (`cold_cv`: hold out every link of a fold of drugs, or of diseases/targets;
report precision@5/10/15). Settings follow the paper's pair-prediction values: ranks 90/70,
α = β = γ = 0.001, w = 0.3, up to 200 iterations, seed 0. One CPU core, hours for the full run.

## What is in the box
| Path | What it is |
|---|---|
| `idrug.py` | iDrug model (multiplicative updates), baselines' entry points, CV scenarios |
| `loaders.py` | reads the authors' `.mat` files (`load_mat`, incl. the gold dataset's headerless CSVs) or the 5-CSV layout (`load_real`) |
| `run_real_all.py` | the real-data benchmark + sweeps behind the website |
| `run_real.py` | single-run CLI (`pair`/`newdrug`/`newcol`, `--official` for the ported solver, `--baselines`) |
| `baselines.py` | GRMF, RLS-Kron, simplified bi-random walk (my implementations from the papers) |
| `make_site_data.py` | builds `docs/idrug-data.js` for the website |
| `sensitivity.py`, `enrichment.py` | α×β grid + w sweep; the paper's hypergeometric enrichment test (function + demo) |
| `gen_datasets.py`, `run_all.py`, `run_baselines.py`, `sparse.py` | synthetic generators + fast pipeline tests (sizes match the paper; never used for the website numbers) |
| `results.json`, `results_baselines.json`, `sparse.json`, `sensitivity.json` | synthetic-test outputs (kept for regression testing) |
| `data/` | placeholder CSVs + format docs for swapping in your own data (`data/README.md`) |

## Honest limits
- `idrug()` is my own implementation of Eq. 1–3 (drug–drug Gram consistency over anchors, one shared rank,
  plain multiplicative updates, weight `w` not `w²`). `idrug_official()` is a line-by-line port of the
  authors' `iDrug.m`; it reproduces the qualitative behaviour of their documented new-drug example
  (README-printed precision 0.23 vs. my run in `results_real.json`), not their full paper tables.
- My protocol is not their code: fold splitting, negative sampling and P@k averaging are independent.
  With uniform negatives, popularity/degree is a strong baseline on CTD — the website says so loudly.
- GRMF and RLS-Kron are my implementations from their papers with default hyperparameters; the bi-random
  walk is a simplified MBiRW stand-in; TL-HGBI is missing.
- Real-data results are seed 0 only (s.d. = across folds). Tuning any baseline could reorder the tables.
- Nothing here is biomedical advice; predictions are unvalidated computational hypotheses.

## Contact
harshinisuresha7@gmail.com
