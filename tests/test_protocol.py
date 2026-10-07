from aeqkad.data import KeystrokeData
from aeqkad.protocol import UserSpace, run_user


def test_runs_are_deterministic_and_complete(toy_csv):
    data = KeystrokeData(toy_csv)
    user = data.test[0]
    space = UserSpace(data, user)
    a, _ = run_user(data, space, user, memory="window", gate="full", label="aeqkad")
    b, _ = run_user(data, space, user, memory="window", gate="full", label="aeqkad")
    assert len(a) == 7
    assert [r["eer"] for r in a] == [r["eer"] for r in b]
    assert all(r["M"] <= 16 for r in a)


def test_static_never_updates(toy_csv):
    data = KeystrokeData(toy_csv)
    user = data.test[1]
    rows, _ = run_user(data, UserSpace(data, user), user, memory="static")
    assert sum(r["n_updates"] for r in rows) == 0
