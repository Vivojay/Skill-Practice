# Model architecture and implementation notes

The family builders assemble small text decoders from shared attention and
feed-forward modules. Each builder specifies its routing, positional encoding
and residual structure. Parameters are initialized locally; released checkpoints
and their tokenizers are outside the current scope.

## What each family means here

| Family | Runnable implementation | Deliberate boundary |
|---|---|---|
| Original dense generation | `models/v1.py`: embeddings, pre-RMSNorm residuals, MHA/GQA, RoPE, SwiGLU, output head | “V1” is a local shorthand for the original DeepSeek LLM family; tiny depth/width and generated tokens |
| Early MoE | `v1.build(moe=True)`: first dense block, then routed/shared experts | Local expert loops; no capacity dropping or expert parallelism |
| V2 | `models/v2.py`: MLA, two shared experts, softmax routing, group-maximum selection | Eight routed experts and scale 4 instead of the published 160 and scale 16; no device/communication loss in the model experiment |
| V3 / V3.1 | `models/v3.py`: sigmoid affinity, sum-of-two group scores, selection bias, normalized weights, sequential MTP wrapper | One initial dense block at depth two; release training/tokenizer updates are not separate architectures |
| V3.2 / Exp | `models/v32.py`: V3 plus lightning indexing, top-k masks and detached indexer supervision | Dense score tensors make the math inspectable; no sparse-kernel throughput claim |
| V4 | `models/v4.py`: CSA/HCA, shared-KV MQA, partial rotation and inverse output rotation, sink, grouped projection, mHC, hash MoE, MTP | Two streams; compression 2/8 and window 4; AdamW; float32; text only |
| R1-Zero / R1 / R1-0528 | `models/r1.py`: shared V3 teacher backbone | These names describe trained releases, not randomly initialized weights; 0528-specific recipe not reconstructed |
| R1 distilled students | `r1.build_student`: Qwen2, Qwen3 and Llama3 dense families | Reduced configurations; no checkpoint importer, released tokenizer or teacher knowledge |
| V4.1-Flash | No builder | Report access failed. Public inference code is visible, but its full release methods were not verified |

Size variants that share an architecture do not get duplicated implementations.
The six original R1 students map to Qwen2.5 and Llama3; the later 0528 student maps
to Qwen3. The local settings are architecture demonstrations, not official-size
configuration files. None of the builders loads downloaded weights.

The model experiments use vocabulary 16, width 32, four query heads and two
blocks. Dense FFNs use hidden width `3D`; routed experts use `D/2`. These easy-to-read
integer sizes keep both dense and routed computation visible on CPU; they are not
the original capacity ratios. V4 uses compression 2/8 so a length-12 example
exercises both compressors. Reasoning students use width 16 and one block.
Dropout is zero. Inputs are generated token IDs with fixed-length prompts; the
decoder assumes every input position is real. Response loss handles right padding,
but arbitrary padded prompt batches and text tokenization are not implemented.

## Implementation choices

**Decoder structure.** `Decoder` follows embedding -> blocks ->
final normalization -> vocabulary projection. A block applies attention and a
feed-forward update, each after normalization. Its hidden state is `[B,T,D]`.
The alternative was separate copies per release; that would obscure the actual
differences. V4 has its own block because its state is `[B,T,S,D]`.

**Attention and cache layout.** GQA stores `[B,T,Hkv,d]` keys and values and
repeats them only for the attention calculation. MLA stores the compressed latent
and positional key separately. Cache tests compare continued prefixes with a full
pass; gradient tests check that future tokens cannot affect earlier outputs.
These tests establish causality and consistency, not model quality.

**Rotary configuration.** Changing the frequency schedule
halfway through a request invalidates cached keys. `rotary.py` supports adjacent
and split-half layouts, the YaRN frequency ramp, and Llama3 wavelength scaling.
V2's optional logit scaling uses coefficient .707; V3 uses 1. The DeepSeek model
experiments use factor 1; the Llama3 student uses wavelength scaling with factor 8.
V4 compressed layers use base 40000, while pure local window layers use 10000.
No long-context accuracy has been established.

**Expert selection and weighting.** Bias and group masks choose experts;
unbiased affinity scores weight their outputs. V2 uses softmax without top-k
renormalization. V3 uses sigmoid with renormalization and bias feedback after an
optimizer step. V4 uses square-root softplus, unrestricted local groups, and a
deterministic token hash in the first block. Its hash table is generated locally,
not copied from a released model. The table is a buffer and survives checkpoints.

**Sparse training.** The V3.2 example first freezes the backbone
and controller and trains only indexers against dense attention. It then trains
the backbone on selected entries and keeps the detached indexer objective.
The local KL is averaged over queries rather than summed, with an explicit loss
weight. The indexer has a dense quadratic cost, and this implementation's main
attention also uses a dense mask. The gathered-head oracle tests the masked math.

**Compression boundaries.** V4's current token may use a block when it
completes that block, following the published reference's zero-based positions.
Incomplete blocks stay in bounded tail state. CSA combines separately projected
previous/current blocks using per-channel softmax weights; HCA uses one larger
block. Both attend to the local window as well. A learned sink consumes probability
mass but contributes a zero value. Because keys are also values, the output's
rotary slice is rotated back by the query position. Tests include this cancellation.

**Residual mixing.** `HyperConnection` collapses streams
with nonnegative input weights, adds the sublayer output through output weights,
and mixes residual streams with a Sinkhorn-normalized matrix. Twenty iterations
approximate double stochasticity; the separate gradient test uses finite steps.
This constrains residual mixing, not the complete nonlinear network. It does not
prove that training cannot diverge. The initialization is a local training choice.

**Optimization.** AdamW, short loops, explicit dictionaries and small
checkpoints keep the experiments understandable. Muon, quantization-aware training,
FP8/FP4 formats, fused kernels and distributed communication are omitted. That is
a training/systems boundary, not an assertion that those details are unimportant.

## R-series stage decisions

`examples/reasoning.py` shows these observable transitions:

1. Pretrain on generated modular sequences, then save that base state.
2. For the `r1` path, train on scripted addition answers. The `zero` path skips this.
3. Sample fresh response groups from a frozen policy and apply the existing
   DeepSeekMath-v1 token-level GRPO objective. Each RL stage has a frozen reference.
4. Retain only exactly verified training-prompt responses and remove duplicates.
5. Reset to the saved base and train on verified responses plus a separate echo task.
6. Apply another RL stage on the two tasks, then train small dense students on the
   collected mixture. Student checkpoints are saved separately.

This preserves useful stage distinctions from the R1 report. It replaces its
reasoning traces, language-consistency rewards and human preference reward models
with much smaller tasks. The student stage is hard-target supervised training,
not vocabulary-logit KL distillation. If rejection produces no correct responses,
the log shows zero accepted rows; the code never fabricates teacher data.
No held-out operand pair enters training or rejection collection.
Within the teacher run, optimizer moments are retained across the first SFT/RL
transition; the documented reset-to-base stage also resets its optimizer. This is
a local choice, not a recovered unpublished training setting. Routing diagnostics
are temporary forward-pass values and are excluded from policy deep copies;
trainable weights and balancing buffers remain in the copied state.

## Primary references

- [Original architecture, section 2.2](https://arxiv.org/html/2401.02954v1#S2.SS2).
- [V2 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/44f7caf/config.json)
  and [V3 reference](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py).
- [V3.2 methods](https://arxiv.org/html/2512.02556v1#S2) and
  [Exp reference](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py).
- [V4 architecture](https://arxiv.org/html/2606.19348v1#S2) and
  [V4 reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/2b2bebc/inference/model.py).
- [R1 stages](https://arxiv.org/html/2501.12948v1#S2),
  [R1-0528 release](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528),
  [Qwen2 reference](https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/qwen2/modeling_qwen2.py),
  [Qwen3 reference](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py),
  [Llama3 scaling reference](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_rope_utils.py).
