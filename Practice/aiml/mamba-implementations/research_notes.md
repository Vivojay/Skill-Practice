# Research notes

## Evidence and equation map

Read versions and code revision are pinned in [sources.json](sources.json);
attribution and original notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The three arXiv HTML papers and official Python references were accessible.
No model kernels were imported or run. Inspection focused on Mamba-1 §2–3,
Algorithm 2, §E.1; Mamba-2 §3, §5–7 and Listing 1; Mamba-3 §3.1–3.4,
Propositions 1–4 and appendices A/B, plus official SISO reference and step code.

| Paper mathematics | Local function | Implemented / omitted |
|---|---|---|
| Mamba-1 Algorithm 2, recurrent §2 equations | `mamba1.core.selective_scan` | Input-dependent delta/B/C, diagonal negative A; sequential autograd; fused scan omitted |
| Mamba-1 §3.4, official `Mamba` initialization | `mamba1.block.Mamba1` | Projection, causal depthwise conv, SiLU, scan, D, SiLU gate, output projection |
| Mamba-2 §3 semiseparable SSM | `mamba2.core.ssd_recurrent`, `ssd_dense` | Scalar transition per head; dense path for verification only |
| Mamba-2 §6 / Listing 1 decomposition | `mamba2.core.ssd_chunked` | Diagonal chunk blocks, incoming-state output, chunk summary and carry; chunk loop is sequential |
| Mamba-2 §7 / `Mamba2Simple` | `mamba2.block.Mamba2` | Parallel projections; conv/SiLU on x/B/C; D, gate, RMSNorm, output; one B/C group |
| Mamba-3 Eq. (5), (11), Propositions 1/4 | `mamba3.core.siso_step`, `siso_scan` | Generalized exponential-trapezoidal update, previous rotated B and x, cumulative phase |
| Mamba-3 Propositions 2–4 | `rotate_pairs`; `tests/complex_oracle.py` | Real interleaved rotations versus direct complex physical-state recurrence |
| Mamba-3 §3.4 / official SISO defaults | `mamba3.block.Mamba3._project` | BC RMSNorm then biases, data-dependent decay/delta/lambda/angles; no explicit conv |

All three block APIs return `(output, state)` from `forward(u, state=None)` and
`step(u_t, state=None)`. Inputs are `(batch,T,width)` or `(batch,width)` respectively.
States are NamedTuples and are never mutated in place. Nonempty sequences only.
Caller-provided states must have matching shape, dtype, and device. For inference,
use `torch.no_grad()`; retaining a differentiable state across calls intentionally
retains the graph. There is no hidden cache or implicit detach.

## Mamba-1

Let inner width be I and state width N. The implemented update is

```
h[b,i,n] = exp(delta[b,t,i] * A[i,n]) * h[b,i,n]
           + delta[b,t,i] * B[b,t,n] * x[b,t,i]
y[b,t,i] = sum_n C[b,t,n] * h[b,i,n]
```

The released implementation uses `delta * B` for the input coefficient. Exact ZOH
for frozen A/B would use `expm1(delta*A)/A * B`. These differ even in the scalar
case; the paper's exact-ZOH gating identity is therefore not an identity for this
implemented input term. Mamba-3 Eq. (4) calls the implemented rule exponential-Euler.

`A=-exp(A_log)`, initialized to `-(1,...,N)` in every channel, ensures negative
diagonal transitions. The low-rank delta projection has rank `ceil(width/16)` and
uniform weights in `[-rank**(-1/2), rank**(-1/2)]`. Its bias is inverse-softplus of
log-uniform delta in `[0.001,0.1]`, floored at `1e-4`. This starts decay timescales
small and varied without making softplus numerically unstable. D starts at one.
Other linear/conv weights use PyTorch defaults; conv has bias, input/output
projections do not. No later initialization pass overwrites the delta bias.
All training uses Adam with zero weight decay, thus also respecting upstream's
intent not to decay A_log, D, or delta biases. Parameters preserve their chosen
dtype; mixed precision policy from fused kernels is omitted.

The constant ablation replaces the projected low-rank delta features, B, and C
with one learned constant vector. Delta still passes through its projection and
softplus. Convolution, activations, gating, residuals and normalization remain:
selection inside the SSM is removed, but the entire network remains nonlinear.
It also has fewer parameters, so this is not a parameter-matched causal study.

## Mamba-2

The core takes discretized inputs `X=delta*x` and log decay `a=delta*A`. Each head
shares scalar A across its P value channels and N state coordinates:

```
H_t = exp(a_t) H_(t-1) + X_t outer B_t       # (P,N) per head
Y_t = H_t C_t                              # (P)
L[t,s] = exp(sum_{j=s+1}^t a_j), s <= t; 0 otherwise
Y = (L * (C B^T)) X + initial-state contribution
```

This is structured causal masked linear attention: C is query, B is key, X is
value. There is no softmax or row-sum normalization; entries can be signed.
The block's RMSNorm is a separate operation after the SiLU gate, not softmax.
B/C are shared across all value heads, after their own channels of the depthwise
convolution and SiLU. Delta is projected directly from the block input and bypasses
the convolution. A starts negative with magnitude uniform on `[1,16]`, D at one,
and delta biases use the same log-uniform inverse-softplus recipe as Mamba-1.

Each chunk computes `(L * C B^T) X`, the incoming-state contribution, and its
weighted terminal input summary. The summary and total chunk decay update the
carried state. Final partial chunks are sliced to their actual length; no padded
decays advance the state. Segment sums avoid subtracting nearby large prefixes,
and the upper triangle is masked to negative infinity before exponentiation.
The implementation loops over chunks. The paper permits parallel summary scans;
mathematical parallelizability is not a claim that this Python loop is parallel,
or that tiny CPU timings predict fused GPU performance.

## Mamba-3 SISO subset

The current official SISO path rotates B/C by a **positive** cumulative angle:

```
phase_t = (phase_(t-1) + delta_t * theta_t) modulo 2*pi
K_t = R(phase_t) B_t; Q_t = R(phase_t) C_t
alpha_t = exp(delta_t*A_t)
beta_t  = (1-lambda_t)*delta_t*alpha_t
gamma_t = lambda_t*delta_t
H_t = alpha_t*H_(t-1) + beta_t*(x_(t-1) outer K_(t-1))
      + gamma_t*(x_t outer K_t)
y_t = H_t Q_t
```

An interleaved pair `(u,v)` rotates to `(u*cos(phi)-v*sin(phi),
u*sin(phi)+v*cos(phi))`. This code rotates half the N real coordinates (N/4 pairs),
following default `rope_fraction=0.5`; N must be divisible by four here.
The projected angular velocity `theta=pi*tanh(raw_angle)` is shared across heads
before multiplication by their separate deltas. Lambda is sigmoid of its projection.
There is no extra delta on the cached previous x or B.

For an independent check, pair the real coordinates as complex numbers and define
physical state `h_t = exp(-i*phase_t)*H_t`. Then the complex transition is
`exp(delta*A - i*delta*theta)`, with the trapezoidal previous-input contribution
inside that transition. Readout is `Re(sum(conj(C_complex)*h_t))`. The paper writes
rotary factors using transposes; the official positive B/C angle convention is
equivalent to negating the learnable physical angular velocity. The oracle uses
this explicit sign and converts final states back to compare both outputs and
state. No extra factor of two is used: N real coordinates encode N/2 complex ones.

BC projections are RMS-normalized over N (epsilon `1e-5`, learned scale initialized
to one), then expanded to heads, then given head-specific biases initialized to
one, then rotated. Following the pinned current reference, real decay is
`A=-max(heavy_tail(raw_A),1e-4)` where heavy_tail is `1+a` for nonnegative a and
`1/(1-a)` otherwise. This is a **current reference choice**, not a claim that the
paper requires this particular activation. Delta biases follow the log-uniform
recipe. D is one. Input/output projections are bias-free with PyTorch defaults.
The block does not use a convolution or x activation. It applies D, SiLU(z) gating,
then output projection, following `is_outproj_norm=False`. Optional headwise
pre-gate output normalization from the reference is not implemented.

Limits checked at the core, with identical supplied parameters and initial state:

- `lambda=1`: beta vanishes; the result is complex exponential-Euler, independent
  of previous key/value. It becomes Mamba-2's core only when rotation is also zero
  and the decay/inputs match (not an equivalence of the complete blocks).
- `theta=0`, initial phase zero: real exponential-trapezoidal recurrence, including
  the previous input. With lambda=1 it agrees with SSD recurrence. A nonzero fixed
  phase is also a coordinate change, but caches must be transformed consistently.
- Lambda near 1/2 under smoothness assumptions gives the trapezoidal accuracy
  interpretation; an unconstrained learned sigmoid lambda need not yield a
  second-order integrator. Discretization accuracy is not a training guarantee.

MIMO would replace rank-one input/output interactions with rank-R matrix products,
increasing compute per state read and potentially improving arithmetic intensity.
It is omitted, along with optimized SISO/MIMO training kernels, paper-scale
benchmarks, SwiGLU alternation, hybrid attention, and pretrained-weight compatibility.
The shared small residual wrapper is not the paper's full network architecture.

## State and actual complexity

Notation: b=batch, T=length, d=model width, I=expand*d=H*P, N=real state size,
k=conv width, Q=chunk length, r=Mamba-1 delta rank, R=N/4 rotation pairs.
The following are **this code's** core costs, excluding stored inputs/outputs where
stated. Autograd training retains intermediates; constant inference state does not
mean constant training memory.

| Core | Arithmetic work | Additional autograd storage (order) | Persistent inference state |
|---|---|---|---|
| Mamba-1 recurrence | b*T*I*N | b*T*I*N | `(b,I,N)` plus conv `(b,I,k-1)` |
| Mamba-2 recurrence | b*T*H*P*N | b*T*H*P*N | `(b,H,P,N)` plus conv `(b,I+2N,k-1)` |
| Mamba-2 dense | b*H*[T²*(N+P)+T*P*N] | b*H*[T²+T*(N+P)+P*N] | Same final state; quadratic workspace makes this verification-only |
| Mamba-2 chunked | b*H*[T*Q*(N+P)+T*P*N+(T/Q)*P*N] | b*H*[T*Q+T*(N+P)+ceil(T/Q)*P*N] | Same recurrent state; no-grad chunk workspace scales with Q² and P*N |
| Mamba-3 SISO recurrence | b*T*H*[P*N+N+R] | b*T*H*[P*N+N+P+R] | phase `(b,H,R)`, H `(b,H,P,N)`, key `(b,H,N)`, value `(b,H,P)` |

The dense/chunk contractions use matrix products; PyTorch may retain additional
constant-factor contraction temporaries. Chunk input/output tensors also occupy
O(b*T*H*(N+P)) storage; returned outputs always occupy O(b*T*I).
Sequential inference with `step` uses O(b*I*N) work/state per token for all models,
plus block projections and the indicated histories. No sequence-length cache grows.

Block projection work adds O(b*T*[d*I+I*(r+N)]) for Mamba-1 and
O(b*T*d*(I+N+H+R)) for Mamba-2/3 (R=0 for Mamba-2). Convolution adds
O(b*T*k*I) or O(b*T*k*(I+2N)), respectively. Norms/gates add O(b*T*I); Mamba-3 BC
normalization/bias/rotation processing additionally costs O(b*T*H*N).
Parameters are O(d*I+I*(r+N+k)) for Mamba-1; O(d*(I+N+H)+k*(I+2N))
for Mamba-2; O(d*(I+N+H+R)+H*N) for Mamba-3. Model embedding/output head
add O(vocab*d) parameters and O(b*T*d*classes) output work. Two blocks double
the block terms. No fused scan, activation recomputation, compiler, or custom
backward is used; Python dispatch overhead matters at these dimensions.

## Verification choices

Core comparisons use tiny float64 tensors: `atol=1e-11, rtol=1e-10`, allowing
roundoff from different contraction orders while detecting indexing/sign errors.
Path gradients use `1e-10/1e-9`; block forward/streaming checks use `1e-10/1e-9`
because projections/convolutions accumulate differently. Numerical gradcheck uses
central differences with `eps=1e-6, atol=1e-5, rtol=1e-4`, looser because finite
differences incur cancellation/truncation. Inputs avoid phase-wrap boundaries and
nondifferentiable activation breakpoints for numerical checks. No tolerance was
relaxed after a mathematical failure.

Tests include independently unrolled/scalar references, direct complex recurrence,
gradient checks including initial/final state and all histories, partial chunks,
large negative decays, state immutability, two-segment continuation, causality,
task leakage/loss masking, serialization, finite gradients for every parameter,
and fixed-batch overfitting (loss <0.05 and 100% recall). Overfitting demonstrates
optimization connectivity and capacity on a tiny fixed set, not generalization.
