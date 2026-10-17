"""Baselines from the iDrug paper, re-implemented from their published descriptions (NOT the authors' code).
  GRMF        Ezzat et al. 2017, graph-regularised MF, alternating least squares, p-nearest-neighbour graphs
  RLS-Kron    van Laarhoven et al. 2011/2013, Gaussian-interaction-profile kernels + Kronecker regularised least squares
  BiRW        simplified bi-random walk in the spirit of MBiRW (Luo et al. 2016): no comprehensive similarity, no
              cluster/threshold step -- so it is NOT a faithful MBiRW
TL_HGBI is not implemented."""
import numpy as np


def _pnn(S, p=5):
    S = S.copy(); np.fill_diagonal(S, 0)
    out = np.zeros_like(S)
    idx = np.argsort(-S, 1)[:, :p]
    rows = np.arange(len(S))[:, None]
    out[rows, idx] = S[rows, idx]
    return np.maximum(out, out.T)


def grmf(Y, Sd, St, k=50, lam_l=0.5, lam_d=1e-3, lam_t=1e-3, p=5, iters=100, seed=0):
    rng = np.random.default_rng(seed); n, m = Y.shape
    Sd, St = _pnn(Sd, p), _pnn(St, p); Dd, Dt = np.diag(Sd.sum(1)), np.diag(St.sum(1))
    A = rng.random((n, k)) * 0.1; B = rng.random((m, k)) * 0.1; I = np.eye(k)
    for _ in range(iters):
        A = np.linalg.solve((B.T @ B + lam_l * I + lam_d * (Dd.sum() / n) * I).T, (Y @ B + lam_d * Sd @ A).T).T
        B = np.linalg.solve((A.T @ A + lam_l * I + lam_t * (Dt.sum() / m) * I).T, (Y.T @ A + lam_t * St @ B).T).T
    return A @ B.T


def _psd(K):
    K = (K + K.T) / 2; w, V = np.linalg.eigh(K); return V, np.clip(w, 0, None)


def _gip(Y, gamma=1.0):
    sq = (Y ** 2).sum(1); g = gamma / max(sq.mean(), 1e-9)
    D = sq[:, None] + sq[None, :] - 2 * Y @ Y.T
    return np.exp(-g * np.clip(D, 0, None))


def rls_kron(Y, Sd, St, sigma=1.0, gamma=1.0, mix=0.5):
    Sd1 = Sd.copy(); np.fill_diagonal(Sd1, 1.0); St1 = St.copy(); np.fill_diagonal(St1, 1.0)
    Kd = mix * Sd1 + (1 - mix) * _gip(Y, gamma); Kt = mix * St1 + (1 - mix) * _gip(Y.T, gamma)
    Vd, ld = _psd(Kd); Vt, lt = _psd(Kt)
    L = np.outer(ld, lt); Z = Vd.T @ Y @ Vt
    return Vd @ (Z * L / (L + sigma)) @ Vt.T


def birw(Y, Sd, St, alpha=0.4, iters=20):
    def norm(S):
        S = S.copy(); np.fill_diagonal(S, 0); d = S.sum(1) + 1e-9
        return S / np.sqrt(np.outer(d, d))
    Ld, Lt = norm(Sd), norm(St); R = Y.copy()
    for _ in range(iters):
        R = alpha / 2 * (Ld @ R + R @ Lt) + (1 - alpha) * Y
    return R
