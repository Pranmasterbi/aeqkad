"""Rebuild the paper's tables and main statistics from a results directory.

    python experiments/analyze.py results
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aeqkad.metrics import drift_slope, fused_frr, per_user, update_safety  # noqa: E402

T_JOB, T_CIRC, T_LINK = 16.6, 0.285, 0.0   # hardware fit (Sec. VI)


def load(res, name):
    path = os.path.join(res, f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def bootstrap_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def p(a, b):
    try:
        return wilcoxon(a, b).pvalue
    except ValueError:      # identical results
        return float("nan")


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main(res):
    main_df, rbf = load(res, "main"), load(res, "rbf")
    acc = pd.concat([d for d in (main_df, rbf) if d is not None])
    E, K = per_user(acc, "eer"), per_user(acc, "eer_k5")
    frr = pd.concat([fused_frr(os.path.join(res, f"{n}_scores.npz"))
                     for n in ("main", "rbf") if os.path.exists(os.path.join(res, f"{n}_scores.npz"))])
    noise = load(res, "noise")
    N = per_user(noise, "eer") if noise is not None else None

    section("Table II: late-session accuracy")
    for lab in ["static", "naive", "window", "mm", "aeqkad", "static_rbf", "window_rbf", "aeqkad_rbf"]:
        if lab not in E:
            continue
        lo, hi = bootstrap_ci(E[lab].values)
        noisy = N[f"{lab}_noisy"].mean() if N is not None and f"{lab}_noisy" in N else float("nan")
        f1 = frr.loc[lab, "frr@1"].mean() if lab in frr.index.get_level_values(0) else float("nan")
        print(f"{lab:12s} EER {E[lab].mean():5.1f} ({lo:.1f}-{hi:.1f})  noisy {noisy:5.1f}  "
              f"k5 {K[lab].mean():5.1f}  FRR@1%FAR(k5) {f1:4.0f}  slope {drift_slope(acc, lab):+.2f}")
    for a, b in [("aeqkad", "static"), ("aeqkad", "naive"), ("aeqkad", "window"), ("aeqkad", "mm"),
                 ("aeqkad", "aeqkad_rbf")]:
        if a in E and b in E:
            print(f"  {a} vs {b}: EER p={p(E[a], E[b]):.2g}, fused p={p(K[a], K[b]):.2g}")

    section("Table III: cost and update safety (r = 0.1)")
    s = update_safety(main_df)
    late = main_df[main_df.session.isin([5, 6, 7, 8])]
    op = late.groupby(["label", "seed", "user"])[["far", "frr"]].mean().groupby("label").mean()
    circuits = main_df.groupby("label").M.mean().round(1)
    circuits["naive"] = 242   # mean over the stream (memory grows from 50 to ~437)
    for lab in ["static", "naive", "window", "mm", "aeqkad"]:
        m = circuits[lab]
        print(f"{lab:8s} circ {m:5.0f}  KiB {1.0625 * m:5.0f}  latency {T_JOB + m * T_CIRC + T_LINK:5.1f}s  "
              f"FAR {op.far[lab]:5.1f}  FRR {op.frr[lab]:5.1f}  upd {s.update_rate[lab]:5.1f}  "
              f"cont {s.contamination[lab]:4.1f}  adm {s.admitted[lab]:5.1f}")

    section("MM variant: why it fails")
    mm = pd.concat([main_df, load(res, "mm_ablation")])
    late = mm[mm.session.isin([5, 6, 7, 8])]
    print(late.groupby("label")[["anchors_in_P", "enroll_in_P", "imp_in_P", "mean_age"]].mean().round(2))
    Em = per_user(mm, "eer")
    print(f"  mm vs mm-no_anchor EER p={p(Em['mm'], Em['mm-no_anchor']):.2g}")
    print(update_safety(mm).round(1))

    section("Table IV: robustness")
    tk = load(res, "takeover")
    if tk is not None:
        print(tk.groupby(["label", "user"])[["takeover_accepted_frac", "takeover_first_reject"]]
                .mean().groupby("label").mean().round(1))
        print(update_safety(tk).contamination.round(1))
    sh = load(res, "shift")
    if sh is not None:
        by = sh.groupby(["label", "session"])[["frr", "eer"]].mean().unstack(0)
        print(by.loc[[5, 8]].round(1))
        locked = sh[sh.session == 8].groupby(["label", "user"]).upd_rate_gen.mean().lt(0.05).groupby("label").mean()
        print("locked out (%):", (100 * locked).round(1).to_dict())
    po = load(res, "poison")
    if po is not None:
        ctl = main_df[main_df.session == 8].groupby(["label", "user"]).atk_final_accept.mean().unstack(0)
        att = po[po.session == 8].groupby(["label", "user"]).atk_final_accept.mean().unstack(0)
        for m in ("window", "mm", "aeqkad"):
            print(f"  poisoning {m:7s} without {ctl[m].mean():5.1f}%  with {att[m + '_poison'].mean():5.1f}%")

    section("Table V: ablations")
    rate = load(res, "rate")
    if rate is not None:
        print(pd.concat([per_user(rate, "eer").mean().rename("eer"), update_safety(rate).contamination], axis=1).round(1))
    ab = load(res, "ablation")
    if ab is not None:
        both = pd.concat([main_df[main_df.label == "aeqkad"], ab])
        print(pd.concat([per_user(both, "eer").mean().rename("eer"), update_safety(both)], axis=1).round(1))
        g = both.groupby("label")[["g_accepted", "g_margin_binds", "g_rej_meanfid", "g_rej_drift"]].sum()
        print("gate diagnostics (% of accepted samples):")
        print((100 * g.div(g.g_accepted.clip(lower=1), axis=0)).drop(columns="g_accepted").round(2))

    section("Second round: soft margin, frontier, pinned anchors, robust threshold")
    for name in ("nu", "frontier", "reserve", "robust"):
        d = load(res, name)
        if d is None:
            continue
        late = d[d.session.isin([5, 6, 7, 8])].groupby(["label", "seed", "user"])[["far", "frr"]].mean().groupby("label").mean()
        print(pd.concat([per_user(d, "eer").mean().rename("eer"), per_user(d, "eer_k5").mean().rename("k5"),
                         update_safety(d), late], axis=1).round(1))
    rp, rs = load(res, "reserve_poison"), load(res, "reserve")
    if rp is not None and rs is not None:
        ctl = rs[rs.session == 8].groupby(["label", "user"]).atk_final_accept.mean().unstack(0)
        att = rp[rp.session == 8].groupby(["label", "user"]).atk_final_accept.mean().unstack(0)
        for lab in att:
            base = lab.replace("_poison", "")
            print(f"  poisoning {base:12s} without {ctl[base].mean():5.1f}%  with {att[lab].mean():5.1f}%")
    rsh, sh = load(res, "reserve_shift"), load(res, "shift")
    if rsh is not None and sh is not None:
        both = pd.concat([sh, rsh])
        print(both[both.session == 8].groupby("label").frr.mean().round(1).rename("FRR session 8 after change"))

    cl = load(res, "classical")
    if cl is not None:
        section("Classical static detectors")
        Ec, Kc = per_user(cl, "eer"), per_user(cl, "eer_k5")
        for lab in Ec:
            lo, hi = bootstrap_ci(Ec[lab].values)
            print(f"{lab:16s} EER {Ec[lab].mean():5.1f} ({lo:.1f}-{hi:.1f})  k5 {Kc[lab].mean():5.1f}  "
                  f"slope {drift_slope(cl, lab):+.2f}")
        if "aeqkad" in E:
            print(f"  manhattan_raw31 vs window: EER p={p(Ec['manhattan_raw31'], E['window']):.2g}; "
                  f"vs aeqkad fused p={p(Kc['manhattan_raw31'], K['aeqkad']):.2g}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
