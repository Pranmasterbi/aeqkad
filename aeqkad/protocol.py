"""Prequential evaluation protocol.

Session 1 enrolls the user. Sessions 2-8 are streamed in order: every sample is
scored by the current model and only then considered for an update. Held-out
probes from the other test users are interleaved through each session and
scored but never learned, so genuine and impostor scores come from the same
model states.

Random streams are seeded per (seed, user), so every configuration sees the
same stream and per-user comparisons are paired.
"""

import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import ocsvm
from .kernels import N_QUBITS, Kernel, zz_states
from .metrics import eer
from .model import Detector, Settings, initial_prototypes

N_PROBE = 5
FUSE_K = 5


def calibrate(det, feats, reps, rng, cfg, robust=False):
    """Thresholds from 5-fold out-of-fold scores on genuine enrollment data.

    With ``robust=True`` the access threshold is median - 1.645 * 1.4826 * MAD,
    the 5% point of a normal fit that ignores the few enrollment samples with
    near-zero fidelity to everything else.
    """
    n = len(feats)
    folds = np.array_split(rng.permutation(n), 5)
    scores, mean_fid, drift = [], [], []
    for held in folds:
        train = np.setdiff1d(np.arange(n), held)
        G = det.enroll_gram[np.ix_(train, train)]
        P = np.array(initial_prototypes(det.memory, G, det.self_k, cfg.m_max, det.reserve))
        alpha, rho = ocsvm.fit(G[np.ix_(P, P)], cfg.nu)
        recent = feats[np.sort(train)[-cfg.drift_window:]]
        mu = recent.mean(0)
        spread = np.linalg.norm(recent - mu, axis=1).mean() + 1e-12
        for j in held:
            k = det.kernel.measure(det.kernel.exact(reps[j:j + 1], reps[train[P]])[0], rng)
            g = k @ alpha - rho
            scores.append(g / rho if det.relative else g)
            mean_fid.append(k.mean())
            drift.append(np.linalg.norm(feats[j] - mu) / spread)
    if robust:
        med = np.median(scores)
        tau = med - 1.645 * 1.4826 * np.median(np.abs(np.asarray(scores) - med))
    else:
        tau = np.percentile(scores, cfg.tau_pct)
    return dict(tau=tau,
                tau_q=np.percentile(scores, cfg.q_up),
                eta=np.percentile(mean_fid, cfg.eta_pct),
                dmax=np.percentile(drift, cfg.drift_pct))


class UserSpace:
    """Standardisation and PCA fitted on one user's enrollment session, applied
    to every test subject's data. ``shift`` simulates an abrupt change in the
    user's own behaviour (timings scaled by ``factor`` from ``from_session``)."""

    def __init__(self, data, user, shift=None, bandwidth=0.08):
        E = data.raw[(user, 1)]
        self.scaler = StandardScaler().fit(E)
        self.pca = PCA(N_QUBITS).fit(self.scaler.transform(E))
        self.bandwidth = bandwidth
        self.feats, self.states = {}, {}
        for s in data.test:
            for k in data.sessions:
                X = data.raw[(s, k)]
                if shift and s == user and k >= shift["from_session"]:
                    X = X * shift["factor"]
                self.feats[(s, k)] = self.project(X)
                self.states[(s, k)] = zz_states(self.feats[(s, k)], bandwidth)

    def project(self, X):
        return self.pca.transform(self.scaler.transform(X))


def build_stream(data, user, session, rng, mode="iid", rate=0.1, block=20):
    """One session's event list: genuine samples, injected impostors, probes."""
    others = [o for o in data.test if o != user]
    probes, pool = {}, []
    for o in others:
        perm = rng.permutation(50)
        probes[o] = perm[:N_PROBE]
        pool += [(o, i) for i in perm[N_PROBE:]]

    events = [("gen", user, i) for i in range(50)]
    if mode in ("iid", "poison") and rate > 0:
        n_imp = int(round(rate * 50 / (1 - rate)))
        for j in rng.choice(len(pool), n_imp, replace=False):
            o, i = pool[j]
            events.insert(int(rng.integers(0, len(events) + 1)), ("imp", o, i))
    elif mode == "takeover":
        o = others[int(rng.integers(len(others)))]
        idx = [i for (oo, i) in pool if oo == o][:block]
        pos = int(rng.integers(10, 40))
        events[pos:pos] = [("imp", o, i) for i in idx]

    # spread the probe groups evenly through the session, one impostor at a time
    rng.shuffle(others)
    n, out, g = len(events), [], 0
    for t, e in enumerate(events):
        out.append(e)
        while g < len(others) and (g + 1) * n / (len(others) + 1) <= t + 1:
            out.append(("probe", others[g], probes[others[g]]))
            g += 1
    while g < len(others):
        out.append(("probe", others[g], probes[others[g]]))
        g += 1
    return out


def run_user(data, space, user, memory, gate="none", anchors=True, seed=0,
             mode="iid", rate=0.1, kind="fidelity", noisy=False, relative=True,
             m_max=None, nu=None, q_up=None, reserve=0, robust=False,
             label=None, keep_scores=False):
    """Stream sessions 2-8 for one user and one configuration."""
    overrides = {k: v for k, v in dict(m_max=m_max, nu=nu, q_up=q_up).items() if v is not None}
    cfg = replace(Settings(), **overrides)
    u = data.test.index(user)
    stream_rng = np.random.default_rng([seed, u, 7])
    shot_rng = np.random.default_rng([seed, u, 11])
    attack_rng = np.random.default_rng([seed, u, 99])
    others = [o for o in data.test if o != user]
    attacker = others[int(attack_rng.integers(len(others)))]

    F = space.feats
    gamma = 1.0 / (N_QUBITS * F[(user, 1)].var(0).mean())
    kernel = Kernel(kind, noisy, gamma, cfg.shots)
    R = F if kind == "rbf" else space.states

    det = Detector(memory, gate, F[(user, 1)], R[(user, 1)], kernel, shot_rng,
                   anchors=anchors, relative=relative, reserve=reserve, cfg=cfg)
    th = calibrate(det, F[(user, 1)], R[(user, 1)], shot_rng, cfg, robust=robust)
    label = label or f"{memory}/{gate}"

    rows, kept, t = [], [], 0
    for k in data.sessions[1:]:
        stream = build_stream(data, user, k, stream_rng, mode=mode, rate=rate)
        attack = None
        if mode == "poison":
            # interpolate from the victim's recorded previous session toward the attacker
            w = 0.1 + 0.9 * (k - 2) / 6
            v = data.raw[(user, k - 1)][attack_rng.choice(50, 10, replace=False)]
            a = data.raw[(attacker, k)][attack_rng.choice(50, 10, replace=False)]
            fa = space.project((1 - w) * v + w * a)
            attack = (fa, fa if kind == "rbf" else zz_states(fa, space.bandwidth))
            for i in range(10):
                stream.insert(int(attack_rng.integers(0, len(stream) + 1)), ("atk", None, i))

        det.reset_counters()
        gen_scores, probe_groups, decisions, is_imp = [], [], [], []
        adm = dict(gen=0, imp=0, atk=0)
        n_imp = n_updates = 0
        for kind_, s, i in stream:
            if kind_ == "probe":
                probe_groups.append([det.score(R[(s, k)][j])[0] for j in i])
                continue
            if kind_ == "atk":
                f, rep = attack[0][i], attack[1][i]
            else:
                f, rep = F[(s, k)][i], R[(s, k)][i]
            r, krow = det.score(rep)
            t += 1
            decisions.append(r)
            is_imp.append(kind_ == "imp")
            if kind_ == "gen":
                gen_scores.append(r)
            elif kind_ == "imp":
                n_imp += 1
            if det.admits(r, krow, f, th):
                det.update(f, rep, krow, t, impostor=kind_ != "gen")
                n_updates += 1
                adm[kind_] += 1

        gen_scores = np.array(gen_scores)
        probes = np.array(probe_groups)
        fused_gen = np.convolve(gen_scores, np.ones(FUSE_K) / FUSE_K, "valid")
        row = dict(label=label, memory=memory, gate=gate, kind=kind, noisy=noisy,
                   relative=relative, m_max=cfg.m_max, nu=cfg.nu, q_up=cfg.q_up,
                   reserve=reserve, robust=robust, mode=mode, rate=rate,
                   seed=seed, user=user, session=k,
                   eer=eer(gen_scores, probes.ravel()),
                   eer_k5=eer(fused_gen, probes.mean(1)),
                   frr=100 * np.mean(gen_scores < th["tau"]),
                   far=100 * np.mean(probes.ravel() >= th["tau"]),
                   n_stream=len(decisions), n_updates=n_updates,
                   gen_admitted=adm["gen"], imp_admitted=adm["imp"], atk_admitted=adm["atk"],
                   imp_injected=n_imp, upd_rate_gen=adm["gen"] / 50,
                   mean_sv=det.counters["sv_total"] / max(1, det.counters["decisions"]),
                   **{f"g_{key}": v for key, v in det.counters.items()})
        row.update(det.memory_stats(t))

        if mode == "takeover" and n_imp:
            d, block = np.array(decisions), np.flatnonzero(is_imp)
            passed = d[block] >= th["tau"]
            fused = np.convolve(d, np.ones(FUSE_K) / FUSE_K, "full")[:len(d)]
            caught = fused[block] < th["tau"]
            row["takeover_accepted_frac"] = 100 * passed.mean()
            row["takeover_first_reject"] = int(np.argmax(~passed)) if (~passed).any() else len(block)
            row["takeover_first_reject_k5"] = int(np.argmax(caught)) if caught.any() else len(block)

        if k == data.sessions[-1]:
            # the attacker's own final-session samples against the final model;
            # in non-poisoning runs this is the no-attack control
            fa = space.project(data.raw[(attacker, k)])
            ra = fa if kind == "rbf" else zz_states(fa, space.bandwidth)
            row["atk_final_accept"] = 100 * np.mean([det.score(x)[0] >= th["tau"] for x in ra])

        rows.append(row)
        if keep_scores:
            kept.append(dict(label=label, seed=seed, user=user, session=k, tau=th["tau"],
                             gen=gen_scores.astype(np.float32), probe=probes.astype(np.float32)))
    return rows, kept


def _one_user(data, user, configs, seeds, shift):
    space = UserSpace(data, user, shift=shift)
    rows, kept = [], []
    for c in configs:
        for seed in seeds:
            r, k = run_user(data, space, user, seed=seed, **c)
            rows += r
            kept += k
    return rows, kept


def run_experiment(data, name, configs, out_dir, seeds=(0, 1, 2), users=None,
                   shift=None, n_jobs=-1):
    """Run every configuration for every user and seed; save <name>.csv (and
    <name>_scores.npz if any configuration keeps scores). Finished experiments
    are loaded instead of rerun, so an interrupted batch can simply be restarted."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(path):
        print(f"{name}: found {path}, skipping")
        return pd.read_csv(path)
    users = users or data.test
    start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_user)(data, u, configs, seeds, shift) for u in users)
    rows = [r for rs, _ in results for r in rs]
    kept = [k for _, ks in results for k in ks]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    if kept:
        np.savez_compressed(
            os.path.join(out_dir, f"{name}_scores.npz"),
            meta=np.array([(k["label"], k["seed"], k["user"], k["session"], k["tau"]) for k in kept], dtype=object),
            gen=np.array([k["gen"] for k in kept], dtype=object),
            probe=np.stack([k["probe"] for k in kept]))
    print(f"{name}: {len(users)} users in {time.time() - start:.0f}s")
    return df
