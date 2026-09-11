# Executed core evidence — 2026-09-11

All 12 planned bounded training runs completed on Windows CPU, Python 3.12.10,
torch 2.7.1+cu118, float32, one PyTorch thread. CUDA was not used. Experiment
implementation commit: `55af5caaf1a267aa831e72ca295c30fd5c8cad88`.
The tests use float64 where relevant. No remote compute, weights or datasets were
downloaded. Summary JSON/CSV is committed; full logs and checkpoints stay in the
excluded `artifacts/` paths recorded by each summary.

## Architectural MoE comparison

| Variant, seed 11 | Held-out MSE | Late training imbalance | Total / active parameters |
|---|---:|---:|---:|
| Fine + shared | 0.081837 | 0.722656 | 12,512 / 6,368 |
| Coarse | 0.077558 | 0.560938 | 12,416 / 6,272 |

120 updates each, 64 tokens/update; expert capacity and active expert work match,
router overhead differs. The coarse baseline did better in this single-seed
regression run. This does not establish a general architecture ranking. Full
expert loads and memory/step-time measures are saved. [Fine](../results/moe-fine-11.json),
[coarse](../results/moe-coarse-11.json), [budget derivation](../notes/moe.md).

## Routing feedback investigation

The [hypothesis and budget](../hypothesis.md) were committed before these runs.
Same starting weights, Adam settings, generated streams, update/token budget and
final evaluation RNG within each pair; no parameter search or winner selection.

| Seed | MSE instant / EMA | Late imbalance instant / EMA | Adaptation updates instant / EMA |
|---:|---:|---:|---:|
| 11 | 0.078595 / 0.078979 | 0.476563 / 0.487500 | not reached / not reached |
| 22 | 0.080751 / 0.081562 | 0.246875 / 0.201302 | 19 / 19 |
| 33 | 0.083362 / 0.084599 | 0.308854 / 0.223177 | 32 / 22 |

Three-pair average MSE increases **1.00%**, late imbalance falls **11.65%**, and
adjacent controller-direction switching falls from **0.3960 to 0.0576**.
Mean-of-seed median step times: 5.965 ms instantaneous, 5.863 ms EMA; controller
times: 0.1273 versus 0.1301 ms. The variation and fixed arm order preclude a robust
speed claim. EMA adds one expert-sized history tensor and O(N) work; the baseline
implementation also keeps normalized loads for diagnostics.

The prespecified mean quality/balance/switching criteria pass. Seed 11's adaptation
is right censored in both arms, so the **overall hypothesis remains inconclusive**.
EMA worsens task MSE in all three seeds and worsens imbalance in seed 11. The
average therefore masks variation between seeds. Three seeds are not a scaling study or
novel-method evidence. [All paired values](../results/routing-pairs.csv),
[summary](../results/routing-summary.json). Rebuild with `python scripts/summarize.py`.

## MLA

| Quantity | Expanded | Compressed |
|---|---:|---:|
| Persistent cache bytes, B=2/S=16 | 10,240 | 1,536 |
| Prefill median / IQR ms | 0.985 / 0.162 | 1.095 / 0.260 |
| One-token decode median / IQR ms | 0.868 / 0.141 | 0.989 / 0.122 |

Five warmups + 25 measurements per phase/path; decode starts with 15 cached tokens.
Maximum float32 output difference: 7.15e-7. Fixed tiny-batch previous-vector MSE:
0.911386 -> 0.000014845 after 120 updates. The cache is **85% smaller**, but slower
on this CPU workload. Process peak working set about 427 MB includes Python,
PyTorch and all allocations, not just cache. [Result](../results/mla.json).

## MTP

| Training objective | Evaluation NTP loss | Predictable-token accuracy | Median step ms |
|---|---:|---:|---:|
| NTP only | 0.341536 | 97.77% | 5.971 |
| NTP + 0.3 MTP | 0.332511 | 97.54% | 11.918 |

Same shared initialization and 30,720 backbone positions. MTP adds 28,800 block
positions and uses 22,176 parameters versus 11,584 used by NTP. Both constructed
objects allocate the MTP module to ensure identical initialization; NTP never
executes it or creates optimizer state for it. Evaluation enumerates the same
64-combination distribution available during training, so it is **not held-out
generalization**. MTP improves loss slightly while reducing accuracy slightly,
at about 2x measured step time. Single paired seed; no decoding acceleration claim.
[NTP](../results/mtp-ntp.json), [MTP](../results/mtp-mtp.json).

## Math-v1 GRPO

160 SFT updates, then 60 fresh groups of 8 questions x 8 completions; two policy
updates/group, 3,840 RL completions. 12,096 policy parameters, frozen old and SFT
reference copies. SFT training greedy correctness is **86.25%** and sampled
correctness **51.56%**, establishing nondegenerate reward variation.

| Held-out metric (20 distinct operand pairs) | SFT | After GRPO |
|---|---:|---:|
| Greedy exact correctness | 5.00% | 0.00% |
| Sampled exact correctness | 8.75% | 4.375% |
| Mean response length | 2.00 | 2.00 |
| Mean within-group reward std | 0.20279 | 0.11200 |
| Sample k3 KL to SFT reference | 0.00000 | 0.10398 |

This is a negative held-out result on a small compositional split: the model fits
training pairs and does not learn robust addition. Greedy counts are 1/20 versus
0/20, so precision is limited. This experiment does not demonstrate emergent
reasoning, reproduce R1 or perform DeepSeek distillation. SFT median step 4.982 ms; each RL round (sampling + two
updates) 27.165 ms. [Full summary](../results/grpo.json).

## Verification and provenance notes

- `python -m pytest`: **19 passed** (5.33 seconds on the final integration run).
  Numerical/gradient checks, masks, causality, cache continuation, bias boundaries,
  frozen evaluation, old/reference policy freezing, per-group reductions and
  tiny overfits are in `tests/`.
- Checkpoint roundtrip test resumes model, Adam state, task generator, global RNG
  and controller and requires exact subsequent tensors. A GRPO CLI one-step
  checkpoint smoke run also succeeded; it is not counted as a planned experiment.
- An actual 10-update EMA CLI run (seed 44) was interrupted after update 5 and
  resumed. It exactly matched an uninterrupted run's model, Adam state, global/task
  RNG, controller and quality metrics. This diagnostic used separate excluded
  verification outputs; it was not used to tune the routing investigation.
- A follow-up controller fix preserves exact raw-count ties for the instantaneous
  rule. Every one of the 360 saved instantaneous update directions was checked
  against the corrected rule and matched; trial metrics did not need rerunning.
- Original logging expanded `python -m` into a script path. The summary `command`
  fields were corrected from the actual executed command log, with a metadata
  note; numerical results and source commit are unchanged. Future logging uses
  `sys.orig_argv`. Later checkpoint-on-budget guards do not change completed runs.
- CPU timings measure these ordinary implementations in this session; they are
  not fused-kernel throughput, exact total-FLOP budgets or a promise of future
  wall time. Memory helpers report tensor bytes and Windows process working set.
