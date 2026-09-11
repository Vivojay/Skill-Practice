# Remaining research coverage

The core includes the catalogue, four mechanism groups, bounded experiments,
numerical checks and one EMA routing ablation. The table describes the mechanisms
and evidence needed for further coverage; those extensions remain unimplemented.

## Validation requirements

Further implementations require the full methods and current reference code, an
explicitly scoped subset, an independently expressed numerical oracle, and a
targeted ablation. The catalogue and experiment records should distinguish
implemented behavior from remaining limitations.

| Next lab | Required mechanism / evidence | Explicit boundary |
|---|---|---|
| M2a NSA | Compression, selection, sliding window and actual gating; causal block boundaries; gather versus dense-masked oracle; one block/window ablation | Educational tensor math, no sparse-kernel speed claim |
| M2b DSA / V3_2_EXP / V3_2 | Obtain full Exp PDF; current corrected indexer positions; lightning indexer, supervision and training phases; indexed attention oracle | Count dense indexer work separately; not NSA |
| M2c MHC | Residual stream maps and finite Sinkhorn; ordinary residual and externally attributed HC baselines; sums/nonnegativity/gradients/growth | Doubly stochastic does not mean identity or guarantee nonlinear stability |
| M2d ENGRAM | Bounded n-gram hashes, deterministic multi-head tables, context gate and local refinement; collision/causal tests | Explain omitted token canonicalization; embeddings are not cached complete answers |
| M2e V4 | Read CSA/HCA methods; tiny hybrid with cache/compression/window tests; reuse applicable existing cores | Muon/precision externally attributed; no million-token reproduction |
| M2f DSPARK | Read semi-AR draft and confidence-scheduled verification; implement correct greedy target verifier first; acceptance/calibration ablation | Any scheduler simulation must expose cost/contention assumptions |
| M2g V4_1_FLASH or newer report | First resolve source access, then identify new disclosed mechanics | Remain source-blocked until full methods can be read |
| M3a LLM / CODER / CODER_V2 / MATH recipes | Tiny filtering/deduplication/FIM transformations, leakage checks, paper-specific recipe deltas | Original data/scaling studies remain absent |
| M3b ESFT | Calibration routes select experts; freeze specified parameters; assert actual changed tensors; domain-shift comparison | No unverified full-checkpoint adaptation |
| M3c R1 | Locally labelled SFT -> RL -> verified rejection selection -> student illustration | Teacher provenance explicit; algorithmic labels are not DeepSeek distillation |
| M3d MATH_V2 / GRM | Independent truth, generation/self-assessment/evaluation separated; deliberately incorrect candidates; false acceptance and exploitation | Rubric/classifier alone is a toy analogue, not SPCT reproduction |
| M3e PROVER / PROVER_1_5 / PROVER_V2 | Separate synthetic data, checker feedback/search and subgoals; sound symbolic checker allowed | Lean claims require an actual Lean run; unchecked generated programs are not executed |
| M3f VL / VL2 | Generated images, small adapter and paper-specific dynamic-resolution/tiling subset | Substitute vision representations explicitly labelled |
| M3g JANUS / JANUS_FLOW / JANUS_PRO | Decoupled visual paths, separate AR/rectified-flow losses and recipe deltas | Pretrained encoders/tokenizers are substitutions, not reproductions |
| M3h OCR / OCR2 | Read actual optical-compression and later causal-flow encoders; rendered labels; character error/token-budget study | Generic downsampling is insufficient |
| M3i DREAMCRAFT | Study hierarchy and learned priors; determine actual asset/rendering feasibility | No procedural-shape fake reproduction; asset-dependent experiment may remain deferred |

Systems coverage is analytical. Distributed training, cloud execution, fused
kernels and large pretrained weights are outside the implementation scope. The
additional experiment ideas remain hypotheses; EMA is the implemented core ablation.
