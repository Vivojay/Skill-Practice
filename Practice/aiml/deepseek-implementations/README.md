# deepseek-minimal

Small, readable PyTorch labs for reconstructing and testing DeepSeek mechanisms.
The core implements: fine/shared MoE and balancing, V2 MLA with
compressed-cache decoding, V3 sequential MTP, and a DeepSeekMath-v1 GRPO subset.
The repository contains small mechanism implementations and generated-task
experiments. It does not contain pretrained checkpoints or reproduce every paper.

[Paper catalogue](papers.md) · [Actual results](docs/evidence.md) ·
[Source audit](docs/discovery.md) · [Remaining coverage](docs/roadmap.md) ·
[Attribution/license](THIRD_PARTY.md)

## Run locally

Windows, Python 3.12, no admin access required. Run from this directory. Only
PyTorch, pytest and the standard library are required; experiments are explicitly
CPU, even if an installed torch wheel supports CUDA. No weights, datasets, API keys,
GPU toolchains, Docker or cloud account are needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch pytest
.\.venv\Scripts\python -m pytest
```

The commands below use `python`; substitute `.\.venv\Scripts\python` if using the
environment above. The existing environment was validated with torch 2.7.1 and
pytest 8.3.3. For the CPU-only PyTorch wheel, use its official CPU package index:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
No editable install is necessary when running the module commands from this root.

```powershell
python -m examples.moe --variant fine
python -m examples.moe --variant coarse
foreach ($seed in 11,22,33) {
    python -m examples.moe --variant instant --seed $seed
    python -m examples.moe --variant ema --seed $seed
}
python -m examples.mla
python -m examples.mtp --variant ntp
python -m examples.mtp --variant mtp
python -m examples.grpo
python scripts/summarize.py
```

Each command has `--help`. Defaults are bounded at 120 updates for MoE/MLA/MTP,
160 SFT + 60 GRPO rounds for arithmetic. Training wall guards are 120 or 180 seconds
per run. These caps exclude startup/evaluation and do not promise machine-specific
wall time. All planned runs completed locally. Results show tradeoffs and failures:
smaller MLA cache did not speed up this CPU path, MTP slightly improved loss but
not accuracy, and arithmetic GRPO worsened held-out correctness.

## Core modules

| Module | Mechanism | Paper note |
|---|---|---|
| [moe.py](deepseek_minimal/moe.py) | Explicit weighted expert sum, f/P balance loss, training-boundary bias update | [MoE](notes/moe.md), [loss-free](notes/balance.md) |
| [mla.py](deepseek_minimal/mla.py) | Content/positional split, both projection absorptions, causal prefix cache | [V2](notes/v2.md) |
| [mtp.py](deepseek_minimal/mtp.py) | Sequential next-token embedding merge, shifted targets/masks and denominator | [V3](notes/v3.md) |
| [grpo.py](deepseek_minimal/grpo.py) | Group advantage, clipped ratio, k3 KL, response mean then group mean | [Math](notes/math.md), [R1 boundary](notes/r1.md) |

The numerical tests are intended as falsifiable checks, not just shape checks.
Tests include independently expressed sums/objectives, gradients, EOS/padding,
causality, cache state and overfit checks. No MTP speculative decoder is included;
there is no claim of draft/verify or stochastic-sampling equivalence.

## Checkpoints and artifacts

Summaries are in `results/`; full JSON logs and `.pt` checkpoints are excluded under
`artifacts/`. Each summary records its artifact paths and source commit. Models,
optimizer/configuration, global Python/PyTorch RNG, task-generator RNG, step,
controller buffers, and GRPO reference state are saved for continuation. Load only
your own trusted checkpoints. Regenerating a run replaces that run's default
checkpoint/summary; use `--output` for a separate summary.

Example of a deliberate interruption and exact continuation:

```powershell
python -m examples.moe --variant ema --seed 44 --stop-after 60
python -m examples.moe --variant ema --seed 44 --resume artifacts/moe-ema-44/checkpoint.pt
```

Keep all configuration flags identical on resume, including the total step budget.
The one-step serialization test also checks exact optimizer/controller/RNG recovery.
Resuming a completed checkpoint can regenerate evaluation. Repeated selection on
held-out results would bias the evaluation. Artifact export is manual.

Optional Colab/Kaggle use is unrun: clone the same committed source, install torch
and pytest, run the same module commands, then explicitly export `results/` and
`artifacts/`. The current scripts intentionally use CPU; a separately documented
GPU experiment would need device/RNG/synchronization and timing changes. GPU
availability/type/VRAM is not assumed and no tunnel is needed.

## Next work

33 distinct reports are catalogued with revisions, attribution, reusable mechanisms,
scope, evidence and omissions. V4.1's official report file is confirmed but its
contents were inaccessible; no internals are invented. Discovery gaps are explicit.
NSA and the other mechanism and training-recipe extensions in the remaining-coverage
table are not implemented in this core snapshot.
