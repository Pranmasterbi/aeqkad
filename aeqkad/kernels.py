"""Fidelity kernel of the ZZ feature map, and the classical RBF kernel.

Eight-qubit states are small, so we build the statevectors directly rather than
going through a circuit simulator. The construction matches
``qiskit.circuit.library.ZZFeatureMap(8, reps=2, entanglement="linear")``;
tests/test_kernels.py checks this when Qiskit is installed.
"""

import numpy as np

N_QUBITS = 8
_DIM = 2 ** N_QUBITS
# bit i of a basis index is qubit i (Qiskit's little-endian convention)
_BITS = ((np.arange(_DIM)[:, None] >> np.arange(N_QUBITS)) & 1).astype(float)
_LEFT, _RIGHT = np.arange(N_QUBITS - 1), np.arange(1, N_QUBITS)
_PARITY = np.logical_xor(_BITS[:, _LEFT], _BITS[:, _RIGHT]).astype(float)


def _hadamard_all(psi):
    """H on every qubit, i.e. a normalised fast Walsh-Hadamard transform."""
    psi = psi.copy()
    n, h = psi.shape[-1], 1
    while h < n:
        psi = psi.reshape(psi.shape[0], n // (2 * h), 2, h)
        a, b = psi[:, :, 0, :].copy(), psi[:, :, 1, :].copy()
        psi[:, :, 0, :], psi[:, :, 1, :] = a + b, a - b
        psi = psi.reshape(psi.shape[0], n)
        h *= 2
    return psi / np.sqrt(n)


def zz_states(X, bandwidth=0.08, reps=2):
    """Statevectors of the ZZ feature map applied to ``bandwidth * X``."""
    x = bandwidth * np.atleast_2d(X)
    phase = (2 * x @ _BITS.T
             + 2 * ((np.pi - x[:, _LEFT]) * (np.pi - x[:, _RIGHT])) @ _PARITY.T)
    psi = np.zeros((len(x), _DIM), complex)
    psi[:, 0] = 1.0
    for _ in range(reps):
        psi = _hadamard_all(psi) * np.exp(1j * phase)
    return psi


class Kernel:
    """Kernel evaluation as the edge server sees it.

    kind="fidelity": |<phi(a)|phi(b)>|^2 estimated from ``shots`` runs of the
        compute-uncompute circuit. With ``noisy=True`` the exact value becomes
        p*K + (1-p)/2^q. The ibm_fez noise model turned out to be a near-uniform
        rescaling of the ideal kernel (mean self-fidelity about 0.75), and this
        global-depolarising form reproduces that rescaling.
    kind="rbf": exp(-gamma ||a-b||^2), exact, no circuits.
    """

    def __init__(self, kind="fidelity", noisy=False, gamma=None, shots=1024,
                 self_fidelity=0.75):
        self.kind, self.gamma, self.shots = kind, gamma, shots
        self.p = (self_fidelity - 1 / _DIM) / (1 - 1 / _DIM) if noisy else 1.0
        self.self_value = self.p + (1 - self.p) / _DIM

    @property
    def noisy(self):
        return self.p < 1

    def exact(self, A, B):
        if self.kind == "rbf":
            sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
            return np.exp(-self.gamma * sq)
        K = np.abs(np.conj(A) @ B.T) ** 2
        return self.p * K + (1 - self.p) / _DIM if self.noisy else K

    def measure(self, K, rng):
        """Shot-noisy estimate of exact kernel values (exact for RBF)."""
        if self.kind == "rbf":
            return K
        return rng.binomial(self.shots, np.clip(K, 0, 1)) / self.shots
