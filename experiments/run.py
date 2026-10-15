"""Run the experiments behind the paper.

    python experiments/run.py                     # everything (about 2-3 h on 2 CPUs)
    python experiments/run.py main takeover       # selected experiments
    python experiments/run.py --quick             # 4 users, 1 seed, a few minutes

Results land in --out (default: results/) as one CSV per experiment; finished
experiments are skipped, so an interrupted run can be restarted as is.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aeqkad.classical import run_classical  # noqa: E402
from aeqkad.data import CMU_URL, KeystrokeData  # noqa: E402
from aeqkad.kernels import zz_states  # noqa: E402
from aeqkad.protocol import run_experiment  # noqa: E402
from configs import EXPERIMENTS  # noqa: E402

EXTRA = ("classical", "edge")


def edge_timing(n=2000, m=16, seed=0):
    """Wall-clock time of one exact kernel decision on this machine (microseconds)."""
    rng = np.random.default_rng(seed)
    protos = zz_states(rng.normal(size=(m, 8)))
    weights = np.full(m, 1 / m)
    xs = rng.normal(size=(n, 8))
    start = time.perf_counter()
    for x in xs:
        k = np.abs(np.conj(zz_states(x[None])) @ protos.T)[0] ** 2
        _ = k @ weights
    return 1e6 * (time.perf_counter() - start) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", help=f"any of: {', '.join(list(EXPERIMENTS) + list(EXTRA))}")
    ap.add_argument("--data", default=CMU_URL, help="path or URL of DSL-StrongPasswordData.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel users (joblib n_jobs)")
    ap.add_argument("--quick", action="store_true", help="4 users, 1 seed")
    args = ap.parse_args()

    names = args.experiments or list(EXPERIMENTS) + list(EXTRA)
    unknown = set(names) - set(EXPERIMENTS) - set(EXTRA)
    if unknown:
        ap.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")

    data = KeystrokeData(args.data)
    print(data)
    seeds = (0,) if args.quick else tuple(range(args.seeds))
    users = data.test[:4] if args.quick else None
    out = args.out + ("_quick" if args.quick else "")
    os.makedirs(out, exist_ok=True)

    for name in names:
        if name == "classical":
            path = os.path.join(out, "classical.csv")
            if not os.path.exists(path):
                run_classical(data, seeds=seeds, users=users).to_csv(path, index=False)
        elif name == "edge":
            us = edge_timing()
            pd.DataFrame([dict(m=16, edge_us_per_decision=us)]).to_csv(os.path.join(out, "edge.csv"), index=False)
            print(f"edge: {us:.0f} us per decision")
        else:
            spec = EXPERIMENTS[name]
            run_experiment(data, name, spec["configs"], out, seeds=seeds, users=users,
                           shift=spec.get("shift"), n_jobs=args.jobs)


if __name__ == "__main__":
    main()
