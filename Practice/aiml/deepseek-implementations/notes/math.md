# MATH — DeepSeekMath v1 outcome GRPO

Source: [2402.03300v1](https://arxiv.org/html/2402.03300v1), section 4.1,
equations 3–4, outcome supervision and Algorithm 1. v3 is catalogued as the latest
observed revision, but **v1 defines this lab**. The official repository README
provides inference usage, not the original GRPO training implementation.

## Equation to code

For each of Q questions, `sample_completions` draws G independent unfiltered,
temperature-one responses from a frozen copy of the current policy. Old log
probabilities are detached and held fixed during two optimizer steps. Reference
is a separate frozen SFT snapshot throughout this single outer iteration.

`group_advantages` returns (r_i-mean(r))/std(r), constant over each response's tokens.
Population std is an explicit convention; v1 does not specify its correction.
Identical rewards give exactly zero advantages. `grpo_loss` uses r=exp(logp-old)
and min(r*A,clip(r,1-eps,1+eps)*A). It subtracts beta times
k3=exp(logp_ref-logp)-(logp_ref-logp)-1 from the objective. The returned loss is the
negative mean of per-response token means, then group/question means. This keeps
the paper's response-length normalization and includes EOS. Prompt positions,
post-EOS positions and explicit padding are excluded. Sampled PAD before EOS is a
real invalid answer action, not silently masked. Length caps are truncations, not
forced EOS events.

The nonnegative k3 expression estimates forward KL under current-policy sampling;
after multiple updates on old samples it is a sample surrogate, not exact KL.
There is no extra importance multiplier on the KL term. Zero advantage removes
the reward term, **not** a potentially nonzero KL gradient.

## Scope and evidence

`python -m pytest tests/test_grpo.py tests/test_checkpoint.py` checks independent
float64 objective arithmetic, gradient signs, detached old/reference probabilities,
EOS/padding, zero reward dispersion and continuation. `python -m examples.grpo`
uses 160 SFT updates before 60 GRPO rounds, 8 questions x 8 completions, two updates
per group, beta=.04 and clipping=.2. Width 32, 12,096 policy parameters, CPU.

Task: operands 0..9; answer is one token representing the sum, followed by EOS.
All rewards are exactly checked arithmetic, not a learned reward model. Training
uses 80 operand pairs; 20 held-out pairs satisfy (a+3b)%5=0. SFT training greedy
correctness is 86.25%, so sampling is nondegenerate. [Actual result](../results/grpo.json):
held-out greedy 5% -> 0%, sampled 8.75% -> 4.375%; no generalization improvement.
Completion length, group reward dispersion, sample KL and timing are recorded.

Omitted: process rewards, reward-model retraining/replay, iterative reference resets,
data mining, natural-language reasoning and large-model effects. This is not a
later GRPO variant, R1-Zero, R1's multi-stage pipeline or a distilled checkpoint.

## Limitations and further experiments

Inferred limitation: tiny groups can have identical rewards and no reward-gradient
signal, while reference regularization remains. Proposed future study: increase
group size while keeping total sampled-token budget fixed by reducing question
batches. Cost: fewer distinct prompts/update and different gradient variance.
Failure: reduced prompt diversity harms generalization. Falsify via held-out exact
correctness, zero-advantage frequency and time jointly. Group-relative estimation
is already studied in GRPO; novelty is not claimed. **Not implemented** here.
