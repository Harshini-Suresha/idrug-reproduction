"""Turn the 5 CSV files in data/ into the dict that idrug.py expects."""
import os
import numpy as np
import pandas as pd


def _resolve(path):
    return path if os.path.exists(path) else path + ".gz"


def _square(path):
    S = pd.read_csv(_resolve(path), index_col=0)
    S.index = S.index.astype(str); S.columns = S.columns.astype(str)
    return S


def _pairs(path):
    df = pd.read_csv(path, dtype=str)
    return df.iloc[:, 0], df.iloc[:, 1]


def is_placeholder(data_dir="data"):
    dd = pd.read_csv(_resolve(os.path.join(data_dir, "drug_disease.csv")))
    return len(dd) < 20  # the shipped placeholder has a single example row


def load_real(data_dir="data"):
    note = os.path.join(data_dir, "SYNTHETIC.txt")
    if os.path.exists(note): print("*** SYNTHETIC DATA ***\n" + open(note).read())
    d_dis, dis = _pairs(os.path.join(data_dir, "drug_disease.csv"))
    d_tgt, tgt = _pairs(os.path.join(data_dir, "drug_target.csv"))
    Sd = _square(os.path.join(data_dir, "drug_similarity.csv"))
    Si = _square(os.path.join(data_dir, "disease_similarity.csv"))
    St = _square(os.path.join(data_dir, "target_similarity.csv"))

    drugs1 = sorted(set(d_dis)); drugs2 = sorted(set(d_tgt))
    diseases = sorted(set(dis)); targets = sorted(set(tgt))

    def matrix(rows, cols, a, b):
        ri = {x: i for i, x in enumerate(rows)}; ci = {x: i for i, x in enumerate(cols)}
        X = np.zeros((len(rows), len(cols)))
        for u, v in zip(a, b): X[ri[u], ci[v]] = 1
        return X

    def sim(S, ids, normalise=False):
        miss = [i for i in ids if i not in S.index]
        if miss: print(f"warning: {len(miss)} ids missing from a similarity file (treated as 0), e.g. {miss[:3]}")
        M = S.reindex(index=ids, columns=ids).fillna(0).to_numpy(float)
        M = (M + M.T) / 2; np.fill_diagonal(M, 0)
        if normalise and M.max() > 0: M = M / M.max()   # disease similarity is unbounded in the paper
        return np.clip(M, 0, None)

    shared = sorted(set(drugs1) & set(drugs2))
    i1 = {x: i for i, x in enumerate(drugs1)}; i2 = {x: i for i, x in enumerate(drugs2)}
    D = dict(
        X1=matrix(drugs1, diseases, d_dis, dis), A1=sim(Sd, drugs1), B1=sim(Si, diseases, True),
        X2=matrix(drugs2, targets, d_tgt, tgt), A2=sim(Sd, drugs2), B2=sim(St, targets, True),
        anc1=np.array([i1[x] for x in shared]), anc2=np.array([i2[x] for x in shared]))
    print(f"drug-disease: {len(drugs1)} drugs, {len(diseases)} diseases, {int(D['X1'].sum())} links")
    print(f"drug-target : {len(drugs2)} drugs, {len(targets)} targets, {int(D['X2'].sum())} links")
    print(f"anchor (shared) drugs: {len(shared)}")
    return D


def load_mat(repo_dir="."):
    """Load the authors' MATLAB files (DrugDisease.mat, DrugTarget.mat, DrugSimMat1.mat, DrugSimMat2.mat,
    DiseaseSimMat.mat, TargetSimMat.mat, SMat.mat) from a clone of github.com/Case-esaC/iDrug
    (or from its goldDataset/ folder). Returns the same dict as load_real, plus S."""
    from scipy.io import loadmat
    def get(name):
        path = os.path.join(repo_dir, name + ".mat")
        if not os.path.exists(path): path = os.path.join(repo_dir, name)   # goldDataset files have no extension
        try:
            m = loadmat(path)
        except (ValueError, NotImplementedError):  # some goldDataset files are plain headerless CSVs
            try:
                return pd.read_csv(path, header=None).to_numpy(float)
            except ValueError:
                return pd.read_csv(path, header=None, sep=r"\s+").to_numpy(float)
        keys = [k for k in m if not k.startswith("__")]
        k = name if name in m else keys[0]
        a = m[k]; return np.asarray(a.todense() if hasattr(a, "todense") else a, dtype=float)
    X1, X2 = get("DrugDisease"), get("DrugTarget")
    A1, A2, B1, B2 = get("DrugSimMat1"), get("DrugSimMat2"), get("DiseaseSimMat"), get("TargetSimMat")
    S = get("SMat")
    assert S.shape == (X2.shape[0], X1.shape[0]), f"SMat should be (drugs_dt x drugs_dd), got {S.shape}"
    for M in (A1, A2, B1, B2): np.fill_diagonal(M, 0)
    r, c = np.nonzero(S)
    D = dict(X1=(X1 > 0).astype(float), A1=A1, B1=B1, X2=(X2 > 0).astype(float), A2=A2, B2=B2, anc1=c, anc2=r, S=S)
    print(f"drug-disease {X1.shape}, {int(D['X1'].sum())} links | drug-target {X2.shape}, {int(D['X2'].sum())} links | anchors {len(r)}")
    return D
