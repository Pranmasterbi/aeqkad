"""Static classical detectors under the same streams and probes.

The scaled-Manhattan detector is the strongest simple detector in Killourhy &
Maxion (2009). Both detectors are run on all 31 raw features and on the 8 PCA
features the quantum kernel sees, to separate the cost of the reduction from
the cost of the kernel.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .kernels import N_QUBITS
from .metrics import eer
from .protocol import FUSE_K, build_stream


def _detectors(enroll, project, nu):
    E = project(enroll)
    mu = E.mean(0)
    mad = np.abs(E - mu).mean(0) + 1e-9
    z = StandardScaler().fit(E)
    Ez = z.transform(E)
    rbf = OneClassSVM(kernel="rbf", nu=nu, gamma=1 / (Ez.shape[1] * Ez.var(0).mean())).fit(Ez)
    return {
        "manhattan": lambda X: -np.abs(project(X) - mu) @ (1 / mad),
        "ocsvm_rbf": lambda X: rbf.decision_function(z.transform(project(X))),
    }


def run_classical(data, seeds=(0, 1, 2), users=None, nu=0.05):
    rows = []
    for user in users or data.test:
        E = data.raw[(user, 1)]
        scaler = StandardScaler().fit(E)
        pca = PCA(N_QUBITS).fit(scaler.transform(E))
        spaces = {"raw31": lambda X: X,
                  "pca8": lambda X: pca.transform(scaler.transform(X))}
        for space_name, project in spaces.items():
            for det_name, score in _detectors(E, project, nu).items():
                for seed in seeds:
                    rng = np.random.default_rng([seed, data.test.index(user), 7])
                    for k in data.sessions[1:]:
                        stream = build_stream(data, user, k, rng)
                        gen = score(data.raw[(user, k)])
                        groups = [score(data.raw[(o, k)][idx]) for kind, o, idx in stream if kind == "probe"]
                        fused = np.convolve(gen, np.ones(FUSE_K) / FUSE_K, "valid")
                        rows.append(dict(label=f"{det_name}_{space_name}", seed=seed, user=user, session=k,
                                         eer=eer(gen, np.concatenate(groups)),
                                         eer_k5=eer(fused, [g.mean() for g in groups])))
    return pd.DataFrame(rows)
