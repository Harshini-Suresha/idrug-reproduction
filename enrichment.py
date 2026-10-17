"""Paper's 'new predictions are enriched in CTD' test, as a reusable function.
train on a small curated set (`X_known`), take the top-`top` predicted drugs per disease that are NOT already known,
and ask how many appear in a larger reference set (`X_ref`, e.g. CTD) via the hypergeometric distribution.
For real data: restrict CTD to the gold-standard drugs/diseases (same row/column order) and pass it as X_ref."""
import numpy as np
from scipy.stats import hypergeom


def enrichment(scores, X_known, X_ref, top=20):
    n_drug, n_dis = X_known.shape
    unknown = X_known == 0
    cand = np.where(unknown, scores, -np.inf)
    picked = np.zeros_like(X_known, dtype=bool)
    for j in range(n_dis):
        idx = np.argsort(-cand[:, j])[:top]; picked[idx, j] = True
    picked &= unknown
    M = int(unknown.sum()); K = int((X_ref[unknown] > 0).sum()); n = int(picked.sum()); k = int((X_ref[picked] > 0).sum())
    return dict(pairs=M, reference_positives=K, considered=n, hits=k, expected_by_chance=n * K / M,
                fold_enrichment=(k / n) / (K / M), p_value=float(hypergeom.sf(k - 1, M, K, n)), log10_p_value=float(hypergeom.logsf(k - 1, M, K, n) / np.log(10)))


if __name__ == "__main__":  # demo on synthetic data: pretend 60% of links are the "curated" set, all links are the "reference"
    import warnings; warnings.filterwarnings("ignore")
    from idrug import make_data, idrug
    D = make_data(0); X = D["X1"]; rng = np.random.default_rng(0)
    known = X * (rng.random(X.shape) < 0.6); D2 = dict(D, X1=known)
    S = idrug(known, D["A1"], D["B1"], D["X2"], D["A2"], D["B2"], D["anc1"], D["anc2"], r=15, alpha=0.03, beta=0.1, gamma=0.001, w=0.05)[0]
    print(enrichment(S, known, X))
