# LOSS_FREE — auxiliary-loss-free load balancing

Source: [2408.15664v1](https://arxiv.org/html/2408.15664v1), section 3,
Algorithm 1 and section 4.3. The inspected official V3 gate separately stores
original scores; its inference code does not supply the training bias controller.

## Equation to code

`MoE.route`: I=topk(sigmoid(X R^T)+b); combine weights gather the original sigmoid
scores. This is the paper's main sigmoid-gate setting, with raw selected scores.
`pending_load` accumulates counts across training forwards. `finish_step`, called
**after optimizer.step**, applies b_i <- b_i + u sign(mean(c)-c_i), then clears
the counts. Biases are buffers, not learned parameters. The current sequence
cannot change its own routing decisions through batch statistics.

The EMA ablation updates m <- 0.9 m + 0.1 c/sum(c), initialized to the first batch.
It then uses sign(mean(m)-m_i). With decay=0 the controller is the original rule.
Evaluation forward and `finish_step` both preserve all controller buffers.

## Version boundaries

MoE v1 uses softmax affinities. The loss-free paper primarily uses sigmoid.
V3 also uses sigmoid, **normalizes selected unbiased scores**, and adds group/node
restrictions and a complementary sequence-level balance objective. This module's
optional `normalize=True` exposes only that normalization choice; it is not a V3
router reproduction. Neither the V3 grouping policy nor complementary sequence
loss is claimed here. V2's device/communication routing is also outside core.

## Scope, commands and evidence

Run `python -m examples.moe --variant instant --seed 11` and the same with `ema`;
repeat seeds 22 and 33 exactly as [hypothesis.md](../hypothesis.md) specifies.
Use `python -m pytest tests/test_moe.py tests/test_checkpoint.py` for direction,
boundary, evaluation-freeze and continuation tests. Results are the six
`results/moe-{instant,ema}-{11,22,33}.json` summaries; full logs/checkpoints are
excluded under `artifacts/`. [Evidence](../docs/evidence.md) reports all arms.
No training data or evaluation leakage, auxiliary gradient, GPU scheduling or
expert parallelism is introduced by this controller.

## Limitations and further experiments

Author-reported limitation: too-small bias update rates adapt slowly; too-large
rates cause fluctuations (section 4.3). Our hypothesis is that EMA feedback reduces
oscillation, at the risk of adaptation lag. Cost is O(N) state/update work.
Prespecified quality, balance, switching and adaptation criteria can falsify it;
one metric alone is insufficient. Phi-balancing and Expert Threshold Routing
already study related EMA ideas; this is a replication-style ablation, not a
novel-method claim. No hyperparameters were selected using these held-out runs.
