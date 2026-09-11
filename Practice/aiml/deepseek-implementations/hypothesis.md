# Prespecified core investigation — 2026-09-11

Written before running the routing investigation. This is a small ablation of
known feedback smoothing, **not evidence of research novelty**.

Baseline: Auxiliary-Loss-Free Load Balancing v1, Algorithm 1:
`b <- b + u sign(mean(c)-c)` after each training batch. Scores for top-k are
`sigmoid(router(x))+b`; mixture weights are the original sigmoid values.
The paper's v1 Algorithm 1 uses the preceding batch, not an EMA. Its section 4.3
already tests update-rate, proportional-feedback and multiplicative-bias variants.

Modification: replace current normalized loads with an EMA (decay 0.9); initialize
the EMA to the first batch's load fractions. Keep u=0.001, all weights, optimizer,
streams and token budgets identical. Smoothing might reduce noisy switching, but
lag might harm adaptation after a distribution shift. This changes training and
routing decisions; it is not an output-equivalent implementation optimization.

Prior-work search on 2026-09-11: `MoE loss free balancing EMA smoothing expert bias
exponential moving average arxiv`, and `site:arxiv.org DeepSeek moving average load
bias`. Found [phi-balancing](https://arxiv.org/abs/2605.15403) and
[Expert Threshold Routing](https://arxiv.org/abs/2603.11535). These study related
EMA adjustments/thresholds, with different objectives or routing. Phi-balancing's
related-methods paragraph calls DeepSeek's method EMA-based; that characterization
differs from the primary v1 algorithm inspected here. We follow the primary
algorithm. No claim of being first, or of reproducing phi-balancing/ET.

Fixed budget: three paired seeds 11, 22, 33; two arms per seed; 120 updates/arm,
64 generated tokens/update, width 32, 7 routed + 1 shared experts, hidden 16,
top-3 routed; Adam lr=0.003. No hyperparameter search. Batch/task seeds are disjoint
from final evaluation seeds. No development results will select a winning setting.
At update 60 the mixture probability for input cluster +1 changes from 0.85 to
0.15. The mapping from inputs to regression targets stays fixed. This controls
covariate shift without relabeling the problem.

Each paired arm starts with identical state and consumes identical training
examples. One final held-out batch of 512 tokens from the shifted distribution
per arm; its generator is separate and evaluation cannot update controller state.
Log task MSE, all expert loads, max/mean-1 imbalance, adjacent bias-update direction
changes, median step time and IQR, controller time, tensor bytes and process memory.
Adaptation time is the first post-shift update ending three consecutive five-step
rolling imbalance windows <=0.3. If never reached, report null (right censored).

Hypothesis survives this limited test only if the three-seed average late-shift
imbalance and direction-switch rate decrease, held-out MSE degrades by <=5%, and
adaptation does not slow by >10 updates. A violation falsifies this operational
hypothesis; censoring makes the adaptation claim inconclusive. Report all paired
outcomes and overhead, even when the EMA loses. No large-model extrapolation.

Separate architectural comparison: seed 11, same 120 updates and task stream,
fine/shared versus coarse experts, both softmax + expert auxiliary coefficient
0.001. Coarse=4 experts, hidden 32, top-2, no shared expert. Expert parameter
capacity and active expert linear work match; router parameter/work overhead
differs and must be counted. This is not an exactly equal total-FLOP comparison.
