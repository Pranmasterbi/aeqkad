"""CMU keystroke benchmark (Killourhy & Maxion, DSN 2009)."""

import pandas as pd

CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"


class KeystrokeData:
    """Raw timing vectors indexed by (subject, session).

    The first ``n_tune`` subjects in file order were used for hyperparameter
    tuning and are excluded from every reported result.
    """

    def __init__(self, path=CMU_URL, n_tune=10):
        df = pd.read_csv(path)
        self.features = list(df.columns[3:])
        subjects = list(dict.fromkeys(df.subject))
        self.tune, self.test = subjects[:n_tune], subjects[n_tune:]
        self.sessions = sorted(int(s) for s in df.sessionIndex.unique())
        df = df.sort_values(["subject", "sessionIndex", "rep"])
        self.raw = {(s, int(k)): g[self.features].to_numpy(float)
                    for (s, k), g in df.groupby(["subject", "sessionIndex"])}

    def __repr__(self):
        n = len(self.tune) + len(self.test)
        return f"KeystrokeData({n} subjects, {len(self.features)} features, sessions {self.sessions})"
