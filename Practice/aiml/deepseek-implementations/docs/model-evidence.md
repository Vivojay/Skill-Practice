# Model extension: measured evidence

All retained model-family runs used CPU float32, one PyTorch thread, seed 17, two layers and
width 32. Each family ran 60 updates on batches of eight length-12 modular
progressions. V3.2 used its first eight updates for indexer-only warm-up. The
initial and final loss use the same validation set and include the ambiguous
first transition. Predictable accuracy excludes that first transition.

Training starts were 0..11; validation starts were 12..15, with strides 1..3
modulo 16. Validation contains overlapping subsequences and the same arithmetic
rule. These runs show that the implementations learn this task; they do not
establish general reasoning, long-context quality or a ranking of released models.

| Family | Parameters including MTP | Initial loss | Final loss | Predictable accuracy | Median step ms | Cache bytes at 12 tokens, B=1 |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 27,808 | 2.9770 | 0.5208 | 91.67% | 11.30 | 6,144 |
| moe | 32,672 | 2.8651 | 0.4381 | 95.83% | 18.28 | 6,144 |
| v2 | 32,464 | 2.7821 | 0.5107 | 97.50% | 22.76 | 1,152 |
| v3 | 30,928 | 2.9400 | 0.4974 | 97.50% | 22.62 | 1,152 |
| v32 | 32,112 | 2.9586 | 0.7258 | 84.17% | 28.90 | 1,920 |
| v4 | 37,982 | 2.8323 | 0.4412 | 95.83% | 119.70 | 1,440 |
| v3-mtp | 50,408 | 2.9400 | 0.4705 | 98.33% | 45.51 | 1,152 |
| v4-mtp | 57,690 | 2.8323 | 0.4771 | 94.17% | 162.32 | 1,440 |
| qwen2 | 25,888 | 2.9714 | 0.4444 | 97.50% | 12.46 | 3,072 |
| qwen3 | 25,792 | 2.8318 | 0.4488 | 93.33% | 14.56 | 3,072 |
| llama3 | 25,760 | 2.8321 | 0.5033 | 90.83% | 14.21 | 3,072 |

The models differ in parameter count, expert compute and training work. Timings
come from sequential local runs, include no GPU work, and are sensitive to machine
load. They are diagnostic measurements, not throughput comparisons under a matched
budget. V4's Python token loop is substantially slower than the simpler models.
Cache counts include indexer keys and unfinished compression state where needed;
they exclude model parameters and temporary score tensors. V3.2 still materializes
dense scores. V4's short-prefix overhead can exceed MLA's despite compression.

MTP and NTP begin with equal base weights within each V3 or V4 pair, and use the
same training token batches. MTP adds a whole block and extra shifted supervision.
The result summaries report total parameters; the cache is for the base decoder.
MTP is used for training only. Greedy generation uses a fixed token budget and
has no speculative verification, released tokenizer or conversation template.

Exact configurations, commands, environment, source revisions, generated samples,
memory and timings are saved in the eleven `results/model-*.json` files. Logs and
checkpoints remain under ignored `artifacts/` directories.

## R1 training-stage analogue

The arithmetic split is the existing 80 training / 20 held-out operand pairs.
Answers are a sum token followed by EOS, not reasoning traces. The R1 run uses:
30 generic-sequence pretraining steps; 120 answer-SFT steps; 20 first-stage RL
rounds; 640 rejection candidates; 60 mixed-data SFT steps after resetting to the
saved base; 20 mixed-task RL rounds; then three students, each with 20 generic
pretraining steps and 100 supervised steps. Each RL round samples eight questions
and eight responses, and takes two optimizer updates.

### r1

| Stage | Held-out greedy accuracy | Sampled accuracy | Held-out k3 KL |
|---|---:|---:|---:|
| base_pretrain | 0.0% | 0.00% | 0.00000 |
| cold_sft | 45.0% | 28.12% | 0.00000 |
| reasoning_rl | 30.0% | 28.75% | 0.11399 |
| rejection | 30.0% | 28.75% | 0.00000 |
| reset_mixed_sft | 10.0% | 10.00% | 0.00000 |
| mixed_rl | 15.0% | 10.00% | 0.08729 |
| student_qwen2 | 5.0% | 5.31% | 1.98006 |
| student_qwen3 | 5.0% | 6.88% | 1.83042 |
| student_llama3 | 5.0% | 6.56% | 1.84443 |

### zero

| Stage | Held-out greedy accuracy | Sampled accuracy | Held-out k3 KL |
|---|---:|---:|---:|
| base_pretrain | 0.0% | 0.00% | 0.00000 |
| reasoning_rl | 0.0% | 0.00% | 0.01382 |

For RL rows, held-out KL uses the frozen reference at the beginning of that
stage. Non-RL teacher rows use a self-reference, so their zero KL is a measurement
convention, not a claim of zero drift from pretraining. Student KL is relative to
that student's generic-pretraining checkpoint. The final `mixed_rl` stage is a
correctness-reward analogue, not human preference alignment.

The R1 run retained 79 unique verified addition responses from 640 candidates.
Reset SFT and student SFT use those responses plus 80 scripted echo examples.
The pipeline keeps the earlier rejection dataset for students; it does not silently
replace it with ground-truth labels from held-out prompts. All three students
finished at 5% held-out greedy addition accuracy. Their backbones are randomly
initialized small Qwen2, Qwen3 and Llama3 variants, with local generic pretraining.

The Zero path performs 30 generic-pretraining steps and 40 RL rounds without
answer SFT. It observed no positive rewards among 2,560 sampled training responses,
and remained at 0% held-out accuracy. This illustrates a sparse-reward failure with
a weak base. It does not contradict the released R1-Zero results. Even when policy
advantages are zero, KL, retained optimizer momentum, weight decay and routing
feedback can still change the local policy.

These are negative or mixed outcomes, retained without tuning on the held-out
results. They support discussion of initialization, reward sparsity, distribution
shift and the difference between correctness of an optimizer and usefulness of its
learned policy. The original 12 core experiment summaries remain separate.

## Verification and corrections

The test suite covers explicit expert sums, rotary identities, cache continuation,
future-token gradients, a gathered sparse-attention oracle, channelwise compression,
Sinkhorn constraints/gradients, the zero-valued sink, shared MTP gradients, response
masking and reference immutability. CLI continuation checks additionally matched
uninterrupted model, optimizer and global/task RNG states exactly for:

- V3.2 interrupted during indexer warm-up (12-step diagnostic, seed 29).
- V4 interrupted before a compression boundary (8-step diagnostic, seed 29).
- R1 interrupted after rejection collection and resumed through base reset,
  both later teacher stages and all three student checkpoints (seed 17).

Two source corrections were made after initial runs. V4's compressed rotary base
was corrected from the generic 10000 to the published 40000; both V4 experiments
were rerun from source `ed1d485`. V2's optional extended-context logit coefficient
was corrected to .707; default factor-one experiments are unchanged. Other model
summaries record source `b998028`; harmless validation/doc changes were present
during the final student-family runs and do not change their equations.

RL evaluation initially used a self-reference for every teacher row. The corrected
script uses the training-stage reference for RL rows; both reasoning runs were
rerun from `d8eb5dd`. Accuracies and training decisions are unchanged. Superseded
summaries are not presented as final evidence. Checkpoint diagnostics are separate
from the retained experiment set and were not used to select hyperparameters.

V4.1 remains unimplemented because its full report could not be retrieved. None
of these results establish V4.1, million-token inference, production quantization,
distributed performance, or reproduction of released model capabilities.
