# mamba-minimal

Small, explicit PyTorch implementations of Mamba-1, Mamba-2, and a **Mamba-3 SISO
educational subset**. The modules provide recurrent state updates, streaming APIs,
numerical tests, and bounded selective-copy experiments. They do not reproduce
paper-scale models, results, or efficiency. Modules are organized as Python
subpackages; adaptations and their sources are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Setup and commands

Windows PowerShell, Python **3.12**, no administrator access or environment activation
needed. The tested environment used PyTorch 2.7.1 and pytest 8.3.3; CPU wheels suffice.
From the `skills-practice` repository root, change into this directory:

```powershell
cd Practice/aiml/mamba-implementations
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install pytest==8.3.3
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Run the test and training commands from this directory; no editable install is required.
No external datasets, compiler tools,
mamba-ssm, Docker, WSL, notebooks, or paid service is needed. Dependency installation
needs internet; all training data are generated locally.

In the commands below, `python` means the same environment's Python; in PowerShell
you can substitute `.\.venv\Scripts\python.exe` directly:

```powershell
# Fast mathematical checks; skip only fixed-batch optimization tests.
python -m pytest -q -p no:cacheprovider -m "not slow"
# Full suite, including all four fixed-batch overfit checks.
python -m pytest -q -s -p no:cacheprovider --durations=5
# Five updates per model: CPU smoke.
python train.py train --preset smoke --output results/my-smoke
# Measure before selecting a training budget; the benchmark model is discarded.
python train.py benchmark --preset bounded --output results/my-bounded
# Equal bounded budget for the three models and Mamba-1 constant-SSM ablation.
python train.py train --preset bounded --steps 300 --output results/my-bounded
# Load one saved checkpoint and evaluate at both lengths.
python train.py evaluate --checkpoint results/my-bounded/mamba3/checkpoint.pt --output results/my-reload
# Optional separately logged fixed-batch run; fails if loss >= .05 or accuracy < 1.
python train.py train --preset overfit --output results/my-overfit
```

`--model mamba1|mamba2|mamba3|mamba1_constant` selects one model; default is all.
`--seed 1 --device cpu --threads 1` are defaults. CUDA is optional if an existing
PyTorch installation supports it; the included evidence is CPU-only. All presets
are ordinary dictionaries in `train.py`, not a generic configuration framework.
Reusing an output directory overwrites its run files; use a new directory to keep
previous runs. Checkpoints contain final weights and model/task configuration,
and support evaluation rather than optimizer-state resume.

## Architecture

| Location | Purpose |
|---|---|
| `mamba_minimal/mamba1/` | Selective sequential recurrence and causal-convolution block; constant-SSM ablation |
| `mamba_minimal/mamba2/` | Recurrent, dense and chunked SSD; shared B/C group and streaming block |
| `mamba_minimal/mamba3/` | Exponential-trapezoidal SISO recurrence, real rotation pairs, complete streaming history |
| `mamba_minimal/model.py`, `norm.py` | Embedding, one/two pre-norm residual mixers, final norm and output head |
| `mamba_minimal/task.py`, `train.py` | Synthetic task, recall-only loss, bounded training, saving and evaluation |
| `tests/` | Independent mathematical references, gradients, streaming, causality and optimization |

Example streaming use (all blocks share this API):

```python
import torch
from mamba_minimal.mamba2 import Mamba2

block = Mamba2(width=32, state_size=8).eval()
x = torch.randn(2, 35, 32)
with torch.no_grad():
    y, final = block(x)
    first, state = block(x[:, :17])
    second, continued = block(x[:, 17:], state)
    next_y, next_state = block.step(torch.randn(2, 32), continued)
    torch.testing.assert_close(y, torch.cat((first, second), dim=1))
```

[research_notes.md](research_notes.md) gives the equation-to-function map,
initialization, rotation sign convention, state shapes, actual compute/storage
costs, tolerances, and implemented/omitted features. [sources.json](sources.json)
pins papers **2312.00752v2**, **2405.21060v1**, **2603.15569v1**, and official code
commit **e9594ce1c732d97440f0332fdc43170a2294dbfa**. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).

## Task and measured evidence

For K=4, relevant tokens are IDs 1–4, distractors 5–8, delimiter 9, and blank 0.
Three relevant tokens occupy randomly chosen positions in a distractor prefix.
A delimiter is followed by three blank recall inputs. The target is the relevant
tokens in order. No target or shifted target is fed back. Cross-entropy and accuracy
use **only recall positions**. Predictions have four classes; uniform chance is 25%.

This differs from Mamba-1 §E.1: that setup used length 4096, vocabulary 16 including
a white noise token, 16 data tokens, two width-64 layers, and 400,000 updates.
Here the distractor vocabulary is disjoint and varied, total lengths are 32/64,
and the training budget and models are much smaller.

Actual Windows CPU run on 2026-09-11: width 32, expansion 2, N=8, one residual
block, head dimension 8 for Mamba-2/3, Adam lr .003, no weight decay, gradient norm
clip 1, batch 16, **300 updates** (4,800 generated examples / 153,600 input tokens).
Model seed 1; independent training/evaluation generators use 1001/2001/3001.
Each model sees the same generated training stream and 256 fresh held-out examples
at each length (768 recall labels per length). Held-out data are generated only for
evaluation, and are not used for model selection. Length 64 uses its own seed.

| Model | Parameters | Benchmark ms/update | Train seconds | Final train loss | Recall L=32 | Recall L=64 |
|---|---:|---:|---:|---:|---:|---:|
| Mamba-1 | 8,896 | 45.12 | 15.89 | 1.3953 | 24.87% | 24.87% |
| Mamba-2 | 7,912 | 25.15 | 5.54 | 0.8149 | 58.20% | 57.94% |
| Mamba-3 SISO subset | 8,160 | 89.26 | 27.83 | 0.3228 | 89.19% | 82.42% |
| Mamba-1 constant delta/B/C | 7,762 | 54.12 | 14.02 | 1.3959 | 26.04% | 25.91% |

The benchmark used two warmups and five measured forward/backward/optimizer
updates, one CPU thread, before the 300-update run was selected. It excludes data
generation; training elapsed time includes it and loop logging. These short
measurements vary with scheduling and warmup; they are not reliable performance
rankings. Per-run elapsed time including evaluation is also saved.

All **48 tests passed in 36.33 s**, including four fixed-batch overfit checks.
Those use four length-16 examples, two recall tokens, lr .01 and 400 updates:

| Model | Final fixed-batch loss | Recall |
|---|---:|---:|
| Mamba-1 | 0.00005790 | 100% |
| Mamba-2 | 0.00003916 | 100% |
| Mamba-3 SISO subset | 0.00009271 | 100% |
| Constant-SSM ablation | 0.00004934 | 100% |

Raw evidence: [test transcript](results/tests.txt),
[fixed-batch measurements](results/overfit.json),
[environment](results/environment.json), [benchmark](results/bounded/benchmark.json),
[all run records](results/bounded/summary.json). Each model directory contains
`training.csv`, `result.json`, and a small `checkpoint.pt`. These record the actual
configuration, seeds, device, versions, parameter count, losses, accuracies and
elapsed times. Smoke results live in `results/smoke/`; a checkpoint reload
evaluation lives in `results/reloaded/mamba3/` and reproduced both held-out results
exactly. Each bounded checkpoint is 37–42 KB (decimal, rounded).

This is one exploratory seed, not a fair architectural ranking: parameters differ,
the shared wrapper omits parts of the paper architectures, and the toy task and
short common budget favor some initialization/optimization dynamics. Mamba-1 and
its ablation stayed around chance within 300 updates; Mamba-1 did **not** beat the
ablation. Fixed-batch success alongside this failure distinguishes ability to
memorize a tiny set from learning the generated task. No hyperparameter search or
extra run was used to turn that observation into a success claim. Mamba-3's lower
longer-length accuracy is also an observed limitation, not proof of robust length
generalization. No GPU, MIMO, paper-scale experiment or throughput reproduction was run.

The documented test commands disable pytest's cache. Serialization unit tests use
an in-memory stream; the experiment saves and reloads actual checkpoint files.
