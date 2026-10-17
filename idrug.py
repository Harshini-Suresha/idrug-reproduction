"""
iDrug (Chen, Cheng & Li, PLoS Comput Biol 2020) -- small CPU reproduction.

Model (paper Eq. 1-3, re-derived; the equations are images in the HTML version):
  per domain i in {1: drug-disease, 2: drug-target}
    ||W_i . (X_i - U_i V_i^T)||_F^2
    + alpha Tr(U_i^T L_drug_i U_i) + alpha Tr(V_i^T L_other_i V_i)   (graph smoothness)
    + gamma (|U_i|_1 + |V_i|_1)                                       (sparsity)
  + beta * D(U1, U2)    cross-network consistency on anchor (shared) drugs
  with U, V >= 0, optimised by multiplicative updates.

D(U1,U2) here is ||U1a U1a^T - U2a U2a^T||_F^2 over the anchor drugs. The paper uses different
ranks (r1=90, r2=70), so U1 and U2 cannot be compared row-by-row; comparing drug-drug Gram
matrices is rank-agnostic. This is my reading of the method, not a copy of the official code.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-9


# ----------------------------------------------------------------------------- model
def idrug(X1, A1, B1, X2, A2, B2, anc1, anc2, r=20, alpha=0.01, beta=0.01,
          gamma=0.01, w=0.3, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    doms = []
    for X, A, B in ((X1, A1, B1), (X2, A2, B2)):
        n, m = X.shape
        s = np.sqrt(max(X.mean(), 1e-3) / r)
        doms.append(dict(
            X=X, A=A, B=B, W=np.where(X > 0, 1.0, w),
            dA=A.sum(1), dB=B.sum(1),
            U=rng.random((n, r)) * s * 2 + 1e-3, V=rng.random((m, r)) * s * 2 + 1e-3))
    anc = (np.asarray(anc1), np.asarray(anc2))
    use_cross = beta > 0 and len(anc[0]) > 0
    for _ in range(iters):
        for k in (0, 1):
            d, o = doms[k], doms[1 - k]
            U, V, W, X = d["U"], d["V"], d["W"], d["X"]
            num = (W * X) @ V + alpha * d["A"] @ U
            den = (W * (U @ V.T)) @ V + alpha * d["dA"][:, None] * U + gamma
            if use_cross:
                a, b = anc[k], anc[1 - k]
                Ua, Ub = U[a], o["U"][b]
                num[a] += beta * (Ub @ Ub.T) @ Ua
                den[a] += beta * (Ua @ Ua.T) @ Ua
            d["U"] = U = U * num / (den + EPS)
            num = (W * X).T @ U + alpha * d["B"] @ V
            den = (W * (U @ V.T)).T @ U + alpha * d["dB"][:, None] * V + gamma
            d["V"] = V * num / (den + EPS)
    return [d["U"] @ d["V"].T for d in doms]


# ----------------------------------------------------------------------------- baselines
def knn_scores(X, A, B, k=10):
    """Guilt-by-association: average of drug-side and disease/target-side neighbour votes."""
    def topk(S):
        S = S.copy(); np.fill_diagonal(S, 0)
        idx = np.argsort(-S, 1)[:, k:]
        np.put_along_axis(S, idx, 0, 1)
        return S / (S.sum(1, keepdims=True) + EPS)
    return 0.5 * topk(A) @ X + 0.5 * X @ topk(B).T


def popularity_scores(X):
    return np.outer(X.sum(1) + 0.1, X.sum(0) + 0.1)


# ----------------------------------------------------------------------------- synthetic data
def make_data(seed=0, n_drug=300, n_dis=200, n_tgt=150, K=15, overlap=(0, 250, 50, 300),
              p_dd=0.30, p_dt=0.40, sim_noise=0.35):
    """Drugs/diseases/targets share latent 'mechanism' memberships; links are sampled sparsely.
    Drug set of domain 1 = [0,250), domain 2 = [50,300) -> 200 shared drugs."""
    rng = np.random.default_rng(seed)

    def memb(n, per):
        M = np.zeros((n, K))
        for i in range(n):
            M[i, rng.choice(K, rng.integers(1, per + 1), replace=False)] = 1
        return M
    Md, Mi, Mt = memb(n_drug, 2), memb(n_dis, 2), memb(n_tgt, 2)

    def links(Ma, Mb, p):
        shared = (Ma @ Mb.T > 0)
        return (rng.random(shared.shape) < np.where(shared, p, 0.004)).astype(float)
    XD, XT = links(Md, Mi, p_dd), links(Md, Mt, p_dt)

    def sim(M, noise):
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        S = Mn @ Mn.T + rng.normal(0, noise, (len(M),) * 2)
        S = np.clip((S + S.T) / 2, 0, 1); np.fill_diagonal(S, 0)
        return S
    d1, d2 = slice(overlap[0], overlap[1]), slice(overlap[2], overlap[3])
    anc1 = np.arange(overlap[2], overlap[1]) - overlap[0]
    anc2 = np.arange(overlap[2], overlap[1]) - overlap[2]
    SD, SI, ST = sim(Md, sim_noise), sim(Mi, sim_noise), sim(Mt, sim_noise)
    return dict(
        X1=XD[d1], A1=SD[d1, d1], B1=SI, X2=XT[d2], A2=SD[d2, d2], B2=ST, anc1=anc1, anc2=anc2)


# ----------------------------------------------------------------------------- evaluation
def run_methods(D, task, methods, r=20, **kw):
    """Return {method: score matrix for task domain (0=drug-disease, 1=drug-target)}."""
    X1, X2 = D["X1"], D["X2"]
    out = {}
    if "iDrug" in methods:
        kwi = {k: v for k, v in kw.items() if k != "r2"}
        out["iDrug"] = idrug(X1, D["A1"], D["B1"], X2, D["A2"], D["B2"], D["anc1"], D["anc2"],
                             r=r, **kwi)[task]
    if "iDrug-official" in methods:  # port of the authors' MATLAB solver (ranks r, r2; their w semantics)
        S = build_S(D["anc1"], D["anc2"], D["X1"].shape[0], D["X2"].shape[0])
        okw = {k: kw[k] for k in ("alpha", "beta", "gamma", "w", "seed") if k in kw}
        out["iDrug-official"] = idrug_official(D["X1"], D["A1"], D["B1"], D["X2"], D["A2"], D["B2"], S,
                                               r1=r, r2=kw.get("r2", r), max_iter=kw.get("iters", 200), **okw)[0][task]
    if "GRMF-style" in methods:  # single domain: same model with beta=0
        kw0 = {k: v for k, v in kw.items() if k != "r2"}; kw0["beta"] = 0.0
        out["GRMF-style"] = idrug(X1, D["A1"], D["B1"], X2, D["A2"], D["B2"], D["anc1"], D["anc2"],
                                  r=r, **kw0)[task]
    X, A, B = (X1, D["A1"], D["B1"]) if task == 0 else (X2, D["A2"], D["B2"])
    if {"GRMF", "RLS-Kron", "BiRW-simplified"} & set(methods):
        import baselines as bl
        if "GRMF" in methods: out["GRMF"] = bl.grmf(X, A, B, k=min(r, 50), seed=kw.get("seed", 0))
        if "RLS-Kron" in methods: out["RLS-Kron"] = bl.rls_kron(X, A, B)
        if "BiRW-simplified" in methods: out["BiRW-simplified"] = bl.birw(X, A, B)
    if "Similarity-kNN" in methods:
        out["Similarity-kNN"] = knn_scores(X, A, B)
    if "Popularity" in methods:
        out["Popularity"] = popularity_scores(X)
    return out


def swap_task(D):
    """View the dataset with the drug-target network as the task domain."""
    return dict(X1=D["X2"], A1=D["A2"], B1=D["B2"], X2=D["X1"], A2=D["A1"], B2=D["B1"],
                anc1=D["anc2"], anc2=D["anc1"])


def pair_cv(D, task, methods, folds=5, reps=2, seed=0, stats=False, **kw):
    """Scenario 1: hide 1/5 of known links; test = those + equal number of random unknown pairs.
    stats=True returns [mean, s.d.] per metric (across folds/reps) instead of the mean alone."""
    Dt = D if task == 0 else swap_task(D)
    rng = np.random.default_rng(seed)
    res = {m: {"auroc": [], "aupr": []} for m in methods}
    X = Dt["X1"]
    for rep in range(reps):
        pos = np.argwhere(X > 0); rng.shuffle(pos)
        for f in range(folds):
            te = pos[f::folds]
            neg_pool = np.argwhere(X == 0)
            neg = neg_pool[rng.choice(len(neg_pool), len(te), replace=False)]
            Dtr = dict(Dt); Xtr = X.copy(); Xtr[te[:, 0], te[:, 1]] = 0; Dtr["X1"] = Xtr
            sc = run_methods(Dtr, 0, methods, seed=rep * 10 + f, **kw)
            y = np.r_[np.ones(len(te)), np.zeros(len(neg))]
            for m, S in sc.items():
                s = np.r_[S[te[:, 0], te[:, 1]], S[neg[:, 0], neg[:, 1]]]
                res[m]["auroc"].append(roc_auc_score(y, s)); res[m]["aupr"].append(average_precision_score(y, s))
    if stats:
        return {m: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for m, d in res.items()}
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}


def cold_cv(D, task, axis, methods, ks=(5, 10, 15), folds=5, reps=2, seed=0, stats=False, **kw):
    """Scenarios 2/3: hide every link of a fold of drugs (axis=0) or diseases/targets (axis=1).
    Anchor links are kept: the held-out drug still exists in the other domain.
    stats=True returns [mean, s.d.] per k (across rows/folds/reps) instead of the mean alone."""
    Dt = D if task == 0 else swap_task(D)
    rng = np.random.default_rng(seed)
    X = Dt["X1"]; n = X.shape[axis]
    res = {m: {k: [] for k in ks} for m in methods}
    for rep in range(reps):
        perm = rng.permutation(n)
        for f in range(folds):
            hold = perm[f::folds]
            Dtr = dict(Dt); Xtr = X.copy()
            if axis == 0: Xtr[hold] = 0
            else: Xtr[:, hold] = 0
            Dtr["X1"] = Xtr
            sc = run_methods(Dtr, 0, methods, seed=rep * 10 + f, **kw)
            for m, S in sc.items():
                rows = hold if axis == 0 else hold
                for h in hold:
                    s = S[h] if axis == 0 else S[:, h]
                    y = X[h] if axis == 0 else X[:, h]
                    if y.sum() == 0: continue
                    order = np.argsort(-s)
                    for k in ks: res[m][k].append(y[order[:k]].mean())
    if stats:
        return {m: {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in d.items()} for m, d in res.items()}
    return {m: {str(k): float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}


# ----------------------------------------------------------------------------- official-code port
def build_S(anc1, anc2, n1, n2):
    """Mapping matrix of the official repo: S (n2 x n1), S[i,j]=1 if row i of domain 2 and row j of domain 1 are the same drug."""
    S = np.zeros((n2, n1)); S[np.asarray(anc2, int), np.asarray(anc1, int)] = 1.0
    return S


def idrug_official(X1, A1, B1, X2, A2, B2, S, r1=90, r2=70, alpha=0.01, beta=0.01, gamma=0.01,
                   w=0.3, max_iter=200, eps=1e-4, seed=0):
    """Line-by-line numpy port of iDrug.m from github.com/Case-esaC/iDrug (MATLAB).
    Differences from my own `idrug()`: separate ranks, square-root (damped) multiplicative updates,
    unobserved cells weighted w**2, consistency ||(S U1)(S U1)^T - U2 U2^T||_F^2 over ALL drugs of domain 2,
    trace-normalised random init, stop when the objective drops by < eps (max 200 iterations)."""
    rng = np.random.default_rng(seed)
    X, Au, Av, rk = [X1, X2], [A1, A2], [B1, B2], [r1, r2]
    U, V = [], []
    for i in (0, 1):
        n, m = X[i].shape
        u = rng.random((n, rk[i])); v = rng.random((m, rk[i]))
        U.append(u / np.sqrt(np.trace(u.T @ u))); V.append(v / np.sqrt(np.trace(v.T @ v)))
    dAu = [a.sum(1) for a in Au]; dAv = [a.sum(1) for a in Av]
    SS = S.T @ S

    def objective():
        J = 0.0
        for i in (0, 1):
            Wm = np.where(X[i] > 0, 1.0, w)
            J += np.linalg.norm(Wm * (X[i] - U[i] @ V[i].T)) ** 2
            Lu = np.diag(dAu[i]) - Au[i]; Lv = np.diag(dAv[i]) - Av[i]
            J += alpha * (np.trace(U[i].T @ Lu @ U[i]) + np.trace(V[i].T @ Lv @ V[i]))
            J += gamma * (U[i].sum() + V[i].sum())
        SU = S @ U[0]
        return J + beta * np.linalg.norm(SU @ SU.T - U[1] @ U[1].T) ** 2

    def R(i):
        r = U[i] @ V[i].T; r[X[i] == 0] = 0; return r

    objs = [objective()]
    for it in range(max_iter):
        # U1
        num = X[0] @ V[0] + alpha * Au[0] @ U[0] + 2 * beta * S.T @ (U[1] @ (U[1].T @ (S @ U[0])))
        den = ((1 - w**2) * R(0) + w**2 * (U[0] @ V[0].T)) @ V[0] + alpha * dAu[0][:, None] * U[0]
        den += 2 * beta * (S.T @ (S @ U[0])) @ (U[0].T @ (SS @ U[0])) + 0.5 * gamma
        U[0] = U[0] * np.sqrt(num / (den + EPS))
        # U2
        num = X[1] @ V[1] + alpha * Au[1] @ U[1] + 2 * beta * S @ (U[0] @ (U[0].T @ (S.T @ U[1])))
        den = ((1 - w**2) * R(1) + w**2 * (U[1] @ V[1].T)) @ V[1] + alpha * dAu[1][:, None] * U[1]
        den += 2 * beta * U[1] @ (U[1].T @ U[1]) + 0.5 * gamma
        U[1] = U[1] * np.sqrt(num / (den + EPS))
        # V1, V2
        for i in (0, 1):
            num = X[i].T @ U[i] + alpha * Av[i] @ V[i]
            den = ((1 - w**2) * R(i).T + w**2 * (V[i] @ U[i].T)) @ U[i] + alpha * dAv[i][:, None] * V[i] + 0.5 * gamma
            V[i] = V[i] * np.sqrt(num / (den + EPS))
        objs.append(objective())
        if objs[-2] - objs[-1] <= eps: break
    return [U[0] @ V[0].T, U[1] @ V[1].T], objs
