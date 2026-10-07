"""Per-user detector: OC-SVM over a prototype memory, with an optional gate.

Memories
    static  all enrollment samples, never updated
    naive   every accepted sample is added (unbounded)
    window  FIFO of at most m_max prototypes (AE-QKAD when gated)
    mm      anchored buffer with max-min prototype replacement (the variant
            the paper reports as a negative result)

Gates (which update criteria are active)
    none        learn from every accepted sample
    full        score margin/quantile + mean fidelity + drift check
    no_drift, no_margin, no_meanfid, score_only   ablations
    union       full gate with the worst-case margin of Proposition 1
"""

import math
from dataclasses import dataclass

import numpy as np

from . import ocsvm


@dataclass
class Settings:
    nu: float = 0.05
    m_max: int = 16
    n_max: int = 100          # buffer size of the mm variant
    anchor_frac: float = 0.3  # anchors in the mm variant
    drift_window: int = 20
    shots: int = 1024
    delta: float = 0.05
    q_up: float = 25          # percentile for the update quantile tau_q
    tau_pct: float = 5        # calibrated FRR of the access threshold
    eta_pct: float = 10
    drift_pct: float = 95


# (margin, mean fidelity, drift, quantile)
GATES = {
    "none": None,
    "full": (True, True, True, True),
    "no_meanfid": (True, False, True, True),
    "no_drift": (True, True, False, True),
    "no_margin": (False, True, True, True),
    "score_only": (True, False, False, True),
    "union": "union",
}


class Detector:

    def __init__(self, memory, gate, feats, reps, kernel, rng,
                 anchors=True, relative=True, cfg=Settings()):
        self.memory, self.gate, self.kernel, self.rng = memory, gate, kernel, rng
        self.relative, self.cfg = relative, cfg
        use_anchors = anchors and memory == "mm"
        n = len(feats)

        # enrollment Gram matrix: measure the upper triangle once, mirror it
        G = kernel.measure(kernel.exact(reps, reps), rng)
        G = np.triu(G, 1)
        G = G + G.T
        self.self_k = kernel.self_value if kernel.kind == "fidelity" and kernel.noisy else 1.0
        np.fill_diagonal(G, self.self_k)
        self.enroll_gram = G

        centrality = (G.sum(1) - self.self_k) / (n - 1)
        n_anchor = math.ceil(cfg.anchor_frac * n) if use_anchors else 0
        anchors_idx = set(np.argsort(-centrality)[:n_anchor])
        self.items = [dict(f=feats[i], rep=reps[i], anchor=i in anchors_idx,
                           born=i - n, enrolled=True) for i in range(n)]

        if memory in ("static", "naive"):
            protos = list(range(n))
        else:
            protos = ocsvm.farthest_point(G, cfg.m_max, int(np.argmax(centrality)))
        self.protos = list(protos)
        self.gram = G[np.ix_(protos, protos)].copy()
        self.buffer = list(range(n))
        self.recent = [feats[i] for i in range(max(0, n - cfg.drift_window), n)]
        self.reset_counters()
        self._retrain()

    # ------------------------------------------------------------------ scoring
    def reset_counters(self):
        self.counters = dict(accepted=0, rej_score=0, margin_binds=0,
                             rej_meanfid=0, rej_drift=0, sv_total=0, decisions=0)

    def _retrain(self):
        self.alpha, self.rho = ocsvm.fit(self.gram, self.cfg.nu)
        self.proto_reps = np.stack([self.items[i]["rep"] for i in self.protos])

    def _measure(self, rep):
        return self.kernel.measure(self.kernel.exact(rep[None], self.proto_reps)[0], self.rng)

    def score(self, rep):
        """Relative score g/rho (or g with fixed thresholds) and the measured row."""
        k = self._measure(rep)
        g = k @ self.alpha - self.rho
        self.counters["sv_total"] += int((self.alpha > 1e-9).sum())
        self.counters["decisions"] += 1
        return (g / self.rho if self.relative else g), k

    def drift_stat(self, f):
        R = np.array(self.recent)
        mu = R.mean(0)
        spread = np.linalg.norm(R - mu, axis=1).mean() + 1e-12
        return np.linalg.norm(f - mu) / spread

    # ------------------------------------------------------------------ gating
    def admits(self, r, k, f, th):
        """Should an accepted sample update the model?"""
        if self.memory == "static" or r < th["tau"]:
            return False
        rule = GATES[self.gate]
        if rule is None:
            return True
        self.counters["accepted"] += 1
        scale = self.rho if self.relative else 1.0
        if rule == "union":
            use_margin = use_mf = use_drift = use_q = True
            eps = ocsvm.hoeffding_margin(len(self.protos), self.cfg.shots, self.cfg.delta)
        else:
            use_margin, use_mf, use_drift, use_q = rule
            eps = ocsvm.bernstein_margin(self.alpha, k, self.cfg.shots, self.cfg.delta) if use_margin else 0.0

        margin_thr = th["tau"] + 2 * eps / scale
        if use_margin and use_q and margin_thr > th["tau_q"]:
            self.counters["margin_binds"] += 1
        tau_up = max(margin_thr, th["tau_q"] if use_q else -np.inf)

        if r < tau_up:
            self.counters["rej_score"] += 1
            return False
        if use_mf and k.mean() < th["eta"]:
            self.counters["rej_meanfid"] += 1
            return False
        if use_drift and self.drift_stat(f) > th["dmax"]:
            self.counters["rej_drift"] += 1
            return False
        return True

    # ------------------------------------------------------------------ updates
    def update(self, f, rep, k, t, impostor):
        """Add an admitted sample. ``k`` is the row already measured for scoring,
        so no extra circuits are needed."""
        new = len(self.items)
        self.items.append(dict(f=f, rep=rep, anchor=False, born=t, enrolled=False, impostor=impostor))
        self.recent = (self.recent + [f])[-self.cfg.drift_window:]

        if self.memory in ("naive", "window"):
            self._append(new, k)
            if self.memory == "window" and len(self.protos) > self.cfg.m_max:
                oldest = int(np.argmin([self.items[i]["born"] for i in self.protos]))
                self._drop(oldest)
        elif self.memory == "mm":
            self._update_mm(new, k)
        self._retrain()

    def _update_mm(self, new, k):
        self.buffer.append(new)
        if len(self.buffer) > self.cfg.n_max:
            evictable = [b for b in self.buffer if not self.items[b]["anchor"]]
            victim = min(evictable, key=lambda b: self.items[b]["born"])
            self.buffer.remove(victim)
            if victim in self.protos:
                j = self.protos.index(victim)
                self._drop(j)
                k = np.delete(k, j)

        if len(self.protos) < self.cfg.m_max:
            self._append(new, k)
            return
        D = 1 - self.gram.copy()
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if (1 - k).min() > D[i, j]:
            candidates = [c for c in (i, j) if not self.items[self.protos[c]]["anchor"]]
            if candidates:
                c = min(candidates, key=lambda c: self.items[self.protos[c]]["born"])
                self._drop(c)
                self._append(new, np.delete(k, c))

    def _append(self, idx, k):
        m = len(self.protos)
        G = np.full((m + 1, m + 1), self.self_k)
        G[:m, :m] = self.gram
        G[m, :m] = G[:m, m] = k[:m]
        self.gram = G
        self.protos.append(idx)

    def _drop(self, j):
        self.protos.pop(j)
        self.gram = np.delete(np.delete(self.gram, j, 0), j, 1)

    def memory_stats(self, t):
        P = [self.items[i] for i in self.protos]
        return dict(M=len(P),
                    anchors_in_P=sum(p["anchor"] for p in P),
                    enroll_in_P=sum(p["enrolled"] for p in P),
                    imp_in_P=sum(p.get("impostor", False) for p in P),
                    mean_age=float(np.mean([t - p["born"] for p in P])))
