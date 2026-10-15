import numpy as np
import pytest

from aeqkad.kernels import Kernel, zz_states


def test_states_are_normalised():
    psi = zz_states(np.random.default_rng(0).normal(size=(5, 8)))
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)


def test_kernel_is_symmetric_with_unit_diagonal():
    psi = zz_states(np.random.default_rng(1).normal(size=(6, 8)))
    K = Kernel().exact(psi, psi)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1)


def test_noisy_kernel_self_fidelity():
    psi = zz_states(np.random.default_rng(2).normal(size=(3, 8)))
    K = Kernel(noisy=True).exact(psi, psi)
    assert np.allclose(np.diag(K), 0.75)


def test_matches_qiskit_zz_feature_map():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.quantum_info import Statevector

    X = np.random.default_rng(3).normal(size=(4, 8)) * 3
    fm = ZZFeatureMap(8, reps=2, entanglement="linear")
    ref = np.array([Statevector(fm.assign_parameters(0.08 * x)).data for x in X])
    K_ref = np.abs(ref.conj() @ ref.T) ** 2
    ours = zz_states(X)
    assert np.allclose(np.abs(ours.conj() @ ours.T) ** 2, K_ref, atol=1e-12)
