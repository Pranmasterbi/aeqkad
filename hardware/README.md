# Hardware runs

Kernel-estimation and static-scoring validation on `ibm_fez` (IBM Heron) through
Qiskit Runtime, S = 1024 shots, for two test users: 16 prototypes, their Gram
matrix, and 40 scored samples per user (20 genuine, 20 impostor). The 1,528
circuits were split into 13 jobs of 8 to 264 circuits so that a least-squares
fit separates the per-job overhead from the per-circuit time
(T_job = 16.6 s, t_circ = 0.285 s).

`jobs.csv` lists the 13 job identifiers (submitted 23 September 2026). The
jobs ran in two identical blocks of six (A and B) followed by one short job
(C). `qpu_seconds` is the QPU usage reported by IBM Quantum; the wall-clock
times include queueing. Results can be retrieved with

```python
from qiskit_ibm_runtime import QiskitRuntimeService
job = QiskitRuntimeService().job("<job_id>")
result = job.result()
```
