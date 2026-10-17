"""Build docs/idrug-data.js from the result JSONs so the website renders from committed data.
Run after run_real_all.py (real data) -- results.json / sweep JSONs are optional extras.
"""
import json, os

here = os.path.dirname(os.path.abspath(__file__))
out = {"real": None, "sweeps": None}

def load(name):
    path = os.path.join(here, name)
    if os.path.exists(path):
        return json.load(open(path))
    print(f"note: {name} missing, skipped")
    return None

res = load("results_real.json")
if res is None:
    raise SystemExit("results_real.json is required -- run: python run_real_all.py --mat iDrug")

out["real"] = {k: v for k, v in res.items() if k != "protocol"}
out["sweeps"] = load("sweeps_real.json")

js = "window.IDRUG_DATA = " + json.dumps(out, indent=1) + ";\n"
dst = os.path.join(here, "docs", "idrug-data.js")
open(dst, "w").write(js)
print("wrote", dst, f"({len(js)/1024:.1f} KB)")
