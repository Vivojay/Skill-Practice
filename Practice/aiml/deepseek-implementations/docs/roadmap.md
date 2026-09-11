# Remaining research coverage

The repository includes the four original mechanism groups and small text-model
assemblies through V4, plus an R1 training-stage analogue. The table records what
further evidence or implementation is still needed. Model-specific decisions are
explained in [notes/models.md](../notes/models.md).

| Next lab | Required mechanism / evidence | Explicit boundary |
|---|---|---|
| M2a NSA | Compression, selection, sliding window and actual gating; causal block boundaries; gather versus dense-masked oracle; one block/window ablation | Educational tensor math, no sparse-kernel speed claim |
| DSA follow-up | Dense-mask implementation, supervision and warm-up are present; a real gather kernel and long-context study remain | Indexer and current core both retain dense score work |
| mHC follow-up | Maps and finite Sinkhorn are present; deeper stability and ordinary-residual matched ablations remain | Finite iterations approximate the constraints; the nonlinear update is not globally non-expansive |
| M2d ENGRAM | Bounded n-gram hashes, deterministic multi-head tables, context gate and local refinement; collision/causal tests | Explain omitted token canonicalization; embeddings are not cached complete answers |
| V4 follow-up | Text decoder, CSA/HCA, bounded cache, hash routing and MTP are present; production ratios, optimizer and precision studies remain | AdamW and reduced dimensions are explicit substitutions |
| M2f DSPARK | Read semi-AR draft and confidence-scheduled verification; implement correct greedy target verifier first; acceptance/calibration ablation | Any scheduler simulation must expose cost/contention assumptions |
| M2g V4_1_FLASH or newer report | First resolve source access, then identify new disclosed mechanics | Remain source-blocked until full methods can be read |
| M3a LLM / CODER / CODER_V2 / MATH recipes | Tiny filtering/deduplication/FIM transformations, leakage checks, paper-specific recipe deltas | Original data/scaling studies remain absent |
| M3b ESFT | Calibration routes select experts; freeze specified parameters; assert actual changed tensors; domain-shift comparison | No unverified full-checkpoint adaptation |
| R1 follow-up | Stage-order analogue and three student families are present; reasoning traces, preference reward models and real data studies remain | Local teacher samples are not released DeepSeek distillation data |
| M3d MATH_V2 / GRM | Independent truth, generation/self-assessment/evaluation separated; deliberately incorrect candidates; false acceptance and exploitation | Rubric/classifier alone is a toy analogue, not SPCT reproduction |
| M3e PROVER / PROVER_1_5 / PROVER_V2 | Separate synthetic data, checker feedback/search and subgoals; sound symbolic checker allowed | Lean claims require an actual Lean run; unchecked generated programs are not executed |
| M3f VL / VL2 | Generated images, small adapter and paper-specific dynamic-resolution/tiling subset | Substitute vision representations explicitly labelled |
| M3g JANUS / JANUS_FLOW / JANUS_PRO | Decoupled visual paths, separate AR/rectified-flow losses and recipe deltas | Pretrained encoders/tokenizers are substitutions, not reproductions |
| M3h OCR / OCR2 | Read actual optical-compression and later causal-flow encoders; rendered labels; character error/token-budget study | Generic downsampling is insufficient |
| M3i DREAMCRAFT | Study hierarchy and learned priors; determine actual asset/rendering feasibility | No procedural-shape fake reproduction; asset-dependent experiment may remain deferred |

Systems coverage is analytical. Distributed training, cloud execution, fused
kernels and large pretrained weights are outside the implementation scope. The
additional experiment ideas remain hypotheses; EMA is the implemented core ablation.
