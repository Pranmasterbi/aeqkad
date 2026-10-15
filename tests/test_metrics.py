import numpy as np

from aeqkad.metrics import eer, frr_at_far


def test_eer_separable_and_identical():
    assert eer([2, 3, 4], [0, 0.5, 1]) == 0
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    assert abs(eer(x[:2000], x[2000:]) - 50) < 3


def test_frr_at_far():
    # with 100 impostor scores, 1% FAR lets exactly one through: the threshold
    # is the second-highest impostor score (98), and only 50 falls at or below it
    imp = np.arange(100.0)
    assert np.isclose(frr_at_far([98.5, 99.5, 50], imp, 0.01), 100 / 3)
