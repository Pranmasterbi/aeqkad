"""Error rates and the summaries reported in the paper."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

LATE = (5, 6, 7, 8)


def eer(genuine, impostor):
    """Equal error rate in percent; higher scores mean 'more genuine'."""
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    if len(genuine) == 0 or len(impostor) == 0:
        return np.nan
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    fpr, tpr, _ = roc_curve(y, np.r_[genuine, impostor])
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fnr[i] + fpr[i]) / 2


def frr_at_far(genuine, impostor, far):
    """FRR (%) at the strictest threshold whose FAR does not exceed ``far``."""
    ranked = np.sort(impostor)[::-1]
    thr = ranked[int(np.floor(far * len(ranked)))]
    return 100 * np.mean(np.asarray(genuine) <= thr)


def per_user(df, column, sessions=LATE):
    """Late-session mean per user, averaged over seeds (users x labels)."""
    late = df[df.session.isin(sessions)]
    return (late.groupby(["label", "seed", "user"])[column].mean()
                .groupby(["label", "user"]).mean().unstack(0))


def update_safety(df):
    """Admission, contamination and update rates (%) pooled over all decisions."""
    s = df.groupby("label")[["imp_admitted", "imp_injected", "gen_admitted",
                             "n_updates", "n_stream"]].sum()
    learned = (s.imp_admitted + s.gen_admitted).clip(lower=1)
    return pd.DataFrame({
        "admitted": 100 * s.imp_admitted / s.imp_injected.clip(lower=1),
        "contamination": 100 * s.imp_admitted / learned,
        "update_rate": 100 * s.n_updates / s.n_stream,
    })


def drift_slope(df, label):
    """Least-squares slope of the mean EER over sessions (points per session)."""
    by_session = df[df.label == label].groupby("session").eer.mean()
    return np.polyfit(by_session.index, by_session.values, 1)[0]


def fused_frr(npz_path, fars=(0.05, 0.01), k=5, sessions=LATE):
    """FRR at fixed FAR under k-decision fusion, per-user thresholds pooled over
    the late sessions, averaged over seeds. Returns users x (label, far)."""
    z = np.load(npz_path, allow_pickle=True)
    meta = pd.DataFrame(list(z["meta"]), columns=["label", "seed", "user", "session", "tau"])
    meta["i"] = np.arange(len(meta))
    out = []
    for (label, seed, user), g in meta[meta.session.isin(sessions)].groupby(["label", "seed", "user"]):
        gen = np.concatenate([np.convolve(z["gen"][i].astype(float), np.ones(k) / k, "valid") for i in g.i])
        imp = np.concatenate([z["probe"][i].astype(float).mean(1) for i in g.i])
        row = dict(label=label, seed=seed, user=user)
        row.update({f"frr@{int(100 * f)}": frr_at_far(gen, imp, f) for f in fars})
        out.append(row)
    return pd.DataFrame(out).groupby(["label", "user"]).mean(numeric_only=True).drop(columns="seed")
