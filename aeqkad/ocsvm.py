"""One-class SVM on a precomputed Gram matrix, and the update margins."""

import numpy as np
from sklearn.svm import OneClassSVM


def clip_psd(G):
    """Symmetrise and drop negative eigenvalues (shot noise breaks PSD-ness)."""
    G = (G + G.T) / 2
    w, V = np.linalg.eigh(G)
    return (V * np.clip(w, 0, None)) @ V.T


def fit(G, nu=0.05):
    """Return (alpha, rho) of the normalised dual, sum(alpha) = 1.

    scikit-learn scales the dual so that sum(alpha) = nu * n. Dividing alpha
    and rho by the same constant leaves the relative score g/rho unchanged.
    """
    svm = OneClassSVM(kernel="precomputed", nu=nu).fit(clip_psd(G))
    alpha = np.zeros(len(G))
    alpha[svm.support_] = svm.dual_coef_[0]
    total = alpha.sum()
    return alpha / total, -svm.intercept_[0] / total


def farthest_point(G, m, start):
    """Greedy max-min selection of m indices under the distance 1 - K."""
    chosen = [start]
    dist = 1 - G[start].copy()
    while len(chosen) < min(m, len(G)):
        dist[chosen] = -np.inf
        j = int(np.argmax(dist))
        chosen.append(j)
        dist = np.minimum(dist, 1 - G[j])
    return chosen


def bernstein_margin(alpha, K_hat, shots=1024, delta=0.05):
    """Per-sample bound on |g_hat - g| with a plug-in variance (Eq. 6)."""
    K_tilde = (shots * K_hat + 0.5) / (shots + 1)
    L = np.log(2 / delta)
    b = alpha.max() / shots
    var = np.sum(alpha ** 2 * K_tilde * (1 - K_tilde)) / shots
    return b * L / 3 + np.sqrt((b * L / 3) ** 2 + 2 * var * L)


def hoeffding_margin(m, shots=1024, delta=0.05):
    """Worst-case bound of Proposition 1 (union over m kernel estimates)."""
    return np.sqrt(np.log(2 * m / delta) / (2 * shots))
