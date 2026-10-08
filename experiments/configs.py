"""Every configuration behind the paper's tables, grouped into experiments.

Labels: ``aeqkad`` is the proposed method (confidence gate on a FIFO window),
``window`` the ungated window, ``mm`` the anchored max-min variant.
"""

AEQKAD = dict(memory="window", gate="full")

EXPERIMENTS = {
    # Tables II-III and Fig. 2; raw scores kept for fused FRR at fixed FAR
    "main": dict(configs=[
        dict(memory="static", label="static", keep_scores=True),
        dict(memory="naive", label="naive", keep_scores=True),
        dict(memory="window", label="window", keep_scores=True),
        dict(**AEQKAD, label="aeqkad", keep_scores=True),
        dict(memory="mm", gate="full", label="mm", keep_scores=True),
    ]),
    # classical RBF kernel on the same 8 features and the same budget
    "rbf": dict(configs=[
        dict(memory="static", kind="rbf", label="static_rbf", keep_scores=True),
        dict(memory="window", kind="rbf", label="window_rbf", keep_scores=True),
        dict(**AEQKAD, kind="rbf", label="aeqkad_rbf", keep_scores=True),
    ]),
    "noise": dict(configs=[
        dict(memory="static", noisy=True, label="static_noisy"),
        dict(memory="window", noisy=True, label="window_noisy"),
        dict(**AEQKAD, noisy=True, label="aeqkad_noisy"),
        dict(memory="mm", gate="full", noisy=True, label="mm_noisy"),
    ]),
    "rate": dict(configs=[
        dict(memory=m, gate=g, rate=r, label=f"{name}_r{r}")
        for r in (0.0, 0.2, 0.3)
        for m, g, name in (("naive", "none", "naive"), ("window", "none", "window"),
                           ("window", "full", "aeqkad"))
    ]),
    "ablation": dict(configs=[
        dict(memory="window", gate="no_drift", label="aeqkad-no_drift"),
        dict(memory="window", gate="score_only", label="aeqkad-score_only"),
        dict(memory="window", gate="no_margin", label="aeqkad-no_margin"),
        dict(memory="window", gate="no_meanfid", label="aeqkad-no_meanfid"),
        dict(memory="window", gate="union", label="aeqkad-union_margin"),
        dict(**AEQKAD, relative=False, label="aeqkad-fixed_thr"),
        dict(**AEQKAD, m_max=8, label="aeqkad-M8"),
        dict(**AEQKAD, m_max=32, label="aeqkad-M32"),
    ]),
    "mm_ablation": dict(configs=[
        dict(memory="mm", gate="full", anchors=False, label="mm-no_anchor"),
        dict(memory="mm", gate="no_drift", label="mm-no_drift"),
        dict(memory="mm", gate="no_margin", label="mm-no_margin"),
        dict(memory="mm", gate="no_meanfid", label="mm-no_meanfid"),
        dict(memory="mm", gate="score_only", label="mm-score_only"),
    ]),
    "takeover": dict(configs=[
        dict(memory="static", mode="takeover", label="static"),
        dict(memory="window", mode="takeover", label="window"),
        dict(**AEQKAD, mode="takeover", label="aeqkad"),
        dict(memory="mm", gate="full", mode="takeover", label="mm"),
    ]),
    "shift": dict(shift=dict(from_session=5, factor=1.3), configs=[
        dict(memory="static", label="static"),
        dict(memory="window", label="window"),
        dict(**AEQKAD, label="aeqkad"),
        dict(memory="mm", gate="full", label="mm"),
    ]),
    # the no-attack control is the 'atk_final_accept' column of 'main'
    "poison": dict(configs=[
        dict(memory="window", mode="poison", label="window_poison"),
        dict(**AEQKAD, mode="poison", label="aeqkad_poison"),
        dict(memory="mm", gate="full", mode="poison", label="mm_poison"),
    ]),
}

# --- second round: soft margin, admission frontier, pinned anchors, robust threshold
EXPERIMENTS.update({
    # nu*M < 1 makes the windowed OC-SVMs hard-margin; these restore a soft margin
    "nu": dict(configs=[
        dict(memory="window", nu=nu, label=f"window_nu{nu}", keep_scores=True)
        for nu in (0.1, 0.2)
    ] + [
        dict(**AEQKAD, nu=nu, label=f"aeqkad_nu{nu}", keep_scores=True)
        for nu in (0.1, 0.2)
    ]),
    # EER vs impostor admission over the update quantile and the budget
    "frontier": dict(configs=[
        dict(**AEQKAD, q_up=q, m_max=m, label=f"aeqkad_q{q}_M{m}")
        for q in (10, 40, 50) for m in (8, 16, 32)
    ] + [
        dict(memory="window", gate="drift_only", label="aeqkad-drift_only"),
        dict(memory="window", m_max=8, label="window_M8"),
        dict(memory="window", m_max=32, label="window_M32"),
    ]),
    # anchoring that actually reaches the prototype set: k of 16 slots pinned
    "reserve": dict(configs=[
        dict(**AEQKAD, reserve=k, label=f"aeqkad_res{k}") for k in (4, 8)
    ] + [
        dict(memory="window", reserve=k, label=f"window_res{k}") for k in (4, 8)
    ]),
    "reserve_poison": dict(configs=[
        dict(**AEQKAD, reserve=k, mode="poison", label=f"aeqkad_res{k}_poison") for k in (4, 8)
    ] + [
        dict(memory="window", reserve=8, mode="poison", label="window_res8_poison"),
    ]),
    "reserve_shift": dict(shift=dict(from_session=5, factor=1.3), configs=[
        dict(**AEQKAD, reserve=k, label=f"aeqkad_res{k}") for k in (4, 8)
    ]),
    # threshold that is not dragged down by near-zero-fidelity enrollment samples
    "robust": dict(configs=[
        dict(memory="static", robust=True, label="static_robust", keep_scores=True),
        dict(memory="window", robust=True, label="window_robust", keep_scores=True),
        dict(**AEQKAD, robust=True, label="aeqkad_robust", keep_scores=True),
    ]),
})
