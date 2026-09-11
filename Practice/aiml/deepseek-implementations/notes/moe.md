# MOE — DeepSeekMoE

Source: [2401.06066v1](https://arxiv.org/html/2401.06066v1), sections 2–3,
equations 3–17. Official V3 shared-expert/dispatch reference inspected; see
[attribution](../THIRD_PARTY.md). The original MoE HF modeling file was inaccessible.

## Equation to code

For inputs X:[B,T,D], flatten tokens to [BT,D]. `MoE.route` produces S:[BT,N]
by softmax of the router projection. I:[BT,K] selects the largest scores;
W gathers those scores **without renormalization** for the v1 paper equations.
`MoE.forward` computes sum(shared experts) + sum_j W_j E_{I_j}(x), equations 9–11.
The caller adds the residual if embedding this in a Transformer. Each expert is
a bias-free SwiGLU: down(silu(gate(x))*up(x)). No token is dropped.

`balance_losses` implements equations 12–17: f_i=N c_i/(KT), P_i=mean_t S_ti,
expert loss=sum_i f_i P_i. For device groups, average f within each group and sum
P within each group, then sum their products. Coefficients belong to the caller.
The CPU experiment uses only expert coefficient 0.001; device loss is numerically
tested without claiming multi-device execution.

## Scope and evidence

Implemented: fine/shared experts, top-k weighted dispatch, differentiable mixture
scores and selected experts, expert/device balance objectives. Simplified: Python
expert loops, one regression layer, no capacity limit, no communication. Omitted:
pretraining, token dropping, checkpoint compatibility and published scale results.

Run `python -m pytest tests/test_moe.py` and
`python -m examples.moe --variant fine`, then `--variant coarse`.
The float64 expert-sum oracle and input gradients use 1e-12 tolerance. Tests also
cover empty experts, dispatch accounting, objective gradients and tiny overfit.
[Fine results](../results/moe-fine-11.json) and
[coarse results](../results/moe-coarse-11.json) record the actual run.

Fine: 7 routed + 1 shared, hidden=16, K=3; coarse: 4 routed, hidden=32, K=2.
Both have 12,288 expert parameters and 6,144 active expert parameters/token.
Routers differ: 224 versus 128 parameters. The two models have independent
architecture-specific initializations, the same seed and stream, and equal expert
capacity/work; this is not a matched-total-FLOP or same-function initialization.
Parameter counts estimate active linear work; wall time includes dispatch overhead.

## Limitations and further experiments

Inferred limitation: small batches can starve experts despite an average load
objective. Proposed modification: history-smoothed bias feedback, retaining
unbiased combine weights. Extra cost: one expert-sized EMA and update work.
Failure mode: stale feedback after distribution shift. Compare paired streams and
held-out task loss plus imbalance/adaptation; quality regression can falsify the
claim even if balance improves. Related EMA routing work exists. This is the only
implemented extension; see [prespecification](../hypothesis.md) and
[balance note](balance.md), not a second claimed invention.
