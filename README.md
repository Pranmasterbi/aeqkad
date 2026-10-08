# AE-QKAD

AE-QKAD keeps a fidelity-kernel one-class SVM up to date as a user's typing
drifts, without letting impostors into the model and without spending extra
quantum circuits on updates. An edge server scores each sample against at most
16 prototypes, and a confidence gate decides separately whether an accepted
sample may also update the model. Admitted samples replace the oldest
prototype, and retraining reuses the kernel values already measured for
scoring.

## What is in here

```
aeqkad/            the method and the evaluation protocol
  kernels.py       ZZ feature-map fidelity kernel (exact statevectors + shot sampling), RBF kernel
  ocsvm.py         one-class SVM on a precomputed Gram matrix, update margins
  model.py         prototype memories (static, naive, window, mm) and the gate
  protocol.py      calibration, prequential streams, impostor scenarios
  classical.py     static scaled-Manhattan and RBF detectors
  metrics.py       EER, FRR at fixed FAR, admission/contamination
experiments/
  configs.py       every configuration behind the paper's tables
  run.py           runs them
  analyze.py       rebuilds the tables and statistics from a results folder
results/           per-user, per-session outputs of the runs reported in the paper
hardware/          IBM Quantum job identifiers of the ibm_fez runs
notebooks/         Colab notebook that clones this repo and runs everything
tests/
```

## Setup

Python 3.10 or later.

```bash
git clone https://github.com/Pranmasterbi/aeqkad.git
cd aeqkad
pip install -r requirements.txt
```

The CMU keystroke data (`DSL-StrongPasswordData.csv`) is downloaded from
<https://www.cs.cmu.edu/~keystroke/> on first use. If you already have it,
pass `--data path/to/DSL-StrongPasswordData.csv`.

## Reproducing the paper

The numbers in the paper come from the CSV files in `results/`. To print the
tables from them:

```bash
python experiments/analyze.py results
```

To rerun the experiments from scratch:

```bash
python experiments/run.py --quick          # 4 users, 1 seed: a few minutes, checks the setup
python experiments/run.py --out my_results # everything: about 4 hours on 2 CPU cores
python experiments/analyze.py my_results
```

Runs are deterministic, so `my_results` should match `results/` exactly.
Individual experiments can be selected by name, e.g. `python experiments/run.py
main takeover`. Finished experiments are skipped, so an interrupted run can be
restarted with the same command.

| Experiment | Paper | What it varies |
|---|---|---|
| `main` | Tables II, III; Fig. 2 | static, naive, window, MM variant, AE-QKAD |
| `rbf` | Table II | the same memories with a classical RBF kernel |
| `noise` | Table II, Sec. VI | depolarising stand-in for the ibm_fez noise model |
| `takeover`, `shift`, `poison` | Table IV | contiguous takeover, abrupt behaviour change, slow poisoning |
| `rate`, `ablation` | Table V | impostor rate; gate criteria, threshold form, budget |
| `mm_ablation` | Sec. VI | anchors and gate criteria of the MM variant |
| `classical` | Table II | static detectors on all 31 features and on the 8 PCA features |
| `nu` | Sec. VI | soft margin (nu = 0.1, 0.2) for the windowed models |
| `frontier` | Sec. VI, Table V | EER vs impostor admission over the update quantile and the budget; drift-only gate |
| `reserve`, `reserve_poison`, `reserve_shift` | Sec. VI, Table V | 4 or 8 of the 16 prototype slots pinned to enrollment samples |
| `robust` | Sec. VI | median/MAD access threshold instead of the 5th percentile |
| `edge` | Sec. VI | time of one exact kernel decision on the local CPU |

On Colab, open `notebooks/colab.ipynb`; it clones the repository and runs the
same commands.

## Notes on the implementation

- Eight-qubit states are simulated exactly. The construction matches Qiskit's
  `ZZFeatureMap(8, reps=2, entanglement="linear")` to machine precision;
  `tests/test_kernels.py` checks this when Qiskit is installed.
- Shot noise is modelled by binomial sampling of the all-zero outcome
  (S = 1024). The noisy runs replace the kernel by p K + (1 - p)/256 with a
  self-fidelity of 0.75, which reproduces the near-uniform rescaling we
  measured with the ibm_fez noise model.
- Every configuration sees the same streams for a given seed and user, so
  comparisons between methods are paired. Per-user results are averaged over
  the three seeds before the Wilcoxon tests.
- The hardware validation (kernel estimation and static scoring for two users
  on ibm_fez) was run separately with Qiskit Runtime; its job identifiers are
  in `hardware/`.


## License

MIT, see `LICENSE`. The CMU keystroke dataset is distributed by its authors
under their own terms and is not included here.
