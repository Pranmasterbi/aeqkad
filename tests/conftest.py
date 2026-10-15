import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def toy_csv(tmp_path_factory):
    """A small stand-in for the CMU file: 14 subjects, 8 sessions, 50 reps, 31 features."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(14):
        mean = rng.uniform(0.05, 0.4, 31)
        trend = rng.normal(0, 0.01, 31)
        for k in range(1, 9):
            for rep in range(1, 51):
                x = np.abs(mean + trend * k + rng.normal(0, 0.04, 31))
                rows.append([f"s{s:03d}", k, rep, *x])
    cols = ["subject", "sessionIndex", "rep"] + [f"t{i}" for i in range(31)]
    path = tmp_path_factory.mktemp("data") / "toy.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return str(path)
