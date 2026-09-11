# Attribution and modifications

These small implementations adapt the research and reference code listed below.
The local experiments do not reproduce published experimental results.

Architecture, initialization, and some tensor-operation patterns are adapted from
[state-spaces/mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa),
commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`, inspected 2026-09-11.
That repository is Apache-2.0 licensed; its complete license is preserved in
[LICENSE](LICENSE). No upstream `NOTICE` file was present in the inspected tree.
Applicable original notices:

- Copyright (c) 2023, Tri Dao, Albert Gu. (`mamba_simple.py`, selective scan)
- Copyright (c) 2024, Tri Dao, Albert Gu. (`mamba2_simple.py`)
- Copyright (c) 2024, Albert Gu and Tri Dao. (`ssd_minimal.py`)
- Copyright (c) 2026, Dao AI Lab, Goombalab. (`modules/mamba3.py`)
- Copyright (c) 2025, Dao AI Lab, Goombalab. (Mamba-3 SISO reference tests)

Modified files here explicitly identify adaptations in their headers. Changes
include batch-first layouts, ordinary PyTorch operations, functional state returns,
float64 preservation, sequential autograd scans, one B/C group, reduced dimension
options, and partial SSD chunks. No CUDA, Triton, TileLang, einops, or mamba-ssm code
is executed. The complex oracle and scalar-product tests express the mathematics
independently of the real rotation/chunk implementations.

Research attribution:

- Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*, [arXiv:2312.00752v2](https://arxiv.org/abs/2312.00752v2), 2024-05-31.
- Tri Dao and Albert Gu, *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality*,
  [arXiv:2405.21060v1](https://arxiv.org/abs/2405.21060v1), 2024-05-31.
- Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter,
  Tri Dao, and Albert Gu, *Mamba-3: Improved Sequence Modeling using State Space
  Principles*, [arXiv:2603.15569v1](https://arxiv.org/abs/2603.15569v1), 2026-03-16.

[sources.json](sources.json) records the inspected versions, reference files,
and SHA-256 hashes. Downloaded research caches are ignored by Git; the project
works offline after installing PyTorch and pytest.
