# Systems study and hardware boundaries

[Fire-Flyer AI-HPC](https://arxiv.org/html/2408.14158v2) studies co-design of
compute, reduction, network/storage and operations. Its author block says DeepSeek-AI;
HAI-Platform/HFAiLab is predecessor infrastructure lineage. [Insights into V3](https://arxiv.org/abs/2505.09343)
examines the systems implications of V3 and hardware limitations. These are distinct
reports. This note provides small analytical calculations, not their measured
hardware results. [Official project index](https://github.com/deepseek-ai/open-infra-index).

## Calculations anchored to this repository

The fine MoE runs route T=64 tokens to K=3 experts each, giving exactly 192
assignments. With width D=32 and float32 payloads, sending each assignment once
costs T*K*D*4 = **24,576 bytes**. Returning expert outputs of the same width adds
24,576 bytes: **49,152 bytes** before routing metadata, padding or collective
protocol overhead. Actual remote traffic is multiplied by the remote-assignment
fraction. If all experts are local as in our code, remote traffic is zero. If
multiple selected experts share a destination and dispatch deduplicates tokens,
this simple per-assignment estimate overcounts outbound bytes. Accounting must
preserve every assignment and matching combine weight. Routing tests check that
conservation; the saved experiment logs contain 192 assignments per batch.

With 8 experts of hidden size 16, the three expert linear maps contain
8*3*32*16=12,288 weights; four active experts use 6,144 weights/token. Counting a
multiply-add as two FLOPs gives approximately 12,288 active expert linear FLOPs
per token. This excludes SiLU, elementwise products, router, gathering and combining.
The coarse variant matches this expert work but has a smaller router. Thus neither
equal expert capacity nor active parameter counts imply equal measured latency.

MLA's measured [B=2,S=16,H=4,Dc=8,R=4,Dv=8,C=8] cache stores
2*16*(8+4)*4=**1,536 bytes**; expanded K/V stores
2*16*4*(8+4+8)*4=**10,240 bytes**. The difference concerns persistent tensors.
Attention scores, concatenation copies and absorption work can dominate tiny CPU
calls. [Actual timings](../results/mla.json) show this smaller cache was slower.

For a hypothetical device with compute rate F and sustained bandwidth B, a
roofline lower bound for work W and traffic M is max(W/F,M/B). This assumes perfect
overlap and attainable sustained rates. Without overlap a simple bound is
W/F+M/B; shared buses, dependent operations, launches and contention can add time.
No rate F or B is assumed from a published GPU table for our CPU. DualPipe's
overlap cannot be established by this equation alone. There is no mock scheduler
or simulated network presented as a kernel/system implementation.

## Project coverage

| Official project | Role / relation to study | Coverage here |
|---|---|---|
| [DualPipe](https://github.com/deepseek-ai/DualPipe) | Pipeline scheduling/overlap | Analytical context; distributed schedule unimplemented |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | Expert dispatch/combine communication | Payload accounting above; no mocked DeepEP execution |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Optimized attention kernels | Cache comparison only; ordinary MLA code is not FlashMLA |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | Dense/MoE GEMM kernels | Excluded hardware implementation; no emulated FP8/FP4 throughput |
| [3FS](https://github.com/deepseek-ai/3FS) | Distributed storage | Analytical context; local checkpoints are not 3FS |
| [EPLB](https://github.com/deepseek-ai/EPLB) | Physical expert placement/replication balance | Distinct from training selection-bias controller; no placement lab |
| [LPLB](https://github.com/deepseek-ai/LPLB) | Official load-balancing project | Catalogue only; internals not audited or implemented |
| [smallpond](https://github.com/deepseek-ai/smallpond) | Data processing on 3FS | Discovered beyond seed anchors; catalogue only |
| [profile-data](https://github.com/deepseek-ai/profile-data) | Published profile data | Reference-only; not a local benchmark |
| [TileKernels](https://github.com/deepseek-ai/TileKernels) | TileLang kernel library | Official README read; compilation excluded |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | Speculative-decoding training/evaluation; links DSpark | DSpark report counted once; code work deferred |
| [DeepJIT](https://github.com/deepseek-ai/DeepJIT) | GPU/NPU kernel JIT support | Newly discovered official tooling; not installed |
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect) | Top-k kernels for DSA/sampling | Newly discovered official tooling; not installed |
| [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) | Official agent harness | Catalogue only; no agent/deployment expansion |
| [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | Official application/recipe repository | README inspected for new report links; none found there |

Limits of this study: no contention trace, actual collective, distributed file
system, fused kernel, hardware throughput or cluster-cost reproduction. No project
is counted as a standalone paper solely because it has a repository.

Inferred limitation/hypothesis for future study: ideal overlap estimates can be
optimistic when compute and communication share resources. A dependency-respecting
schedule with explicit resource occupancy could improve estimates at the cost of
more measured inputs; comparison against a real trace could falsify it. This is
standard systems modelling, not new research, and is not implemented here.
