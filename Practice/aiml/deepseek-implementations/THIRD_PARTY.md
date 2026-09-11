# Attribution

The mathematical mechanisms originate in the papers linked in papers.md.
This educational project does not reproduce DeepSeek's model training.

MLA absorption, adjacent-pair rotary layout, SwiGLU experts, and separation of
selection scores from weights were checked against DeepSeek-V3
`inference/model.py` at commit `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`:
https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py

Our small PyTorch expression of these routines uses separate projections, explicit
cache objects, and no distributed or quantized kernels. The reference's MIT notice
is preserved below. The V2 and MoE linked Hugging Face modeling files returned 403;
the public V3 implementation supplies the inspected shared-mechanism reference.
The inspected Math README exposes inference usage, not the original GRPO trainer;
GRPO is expressed from its fully disclosed v1 equations. V3's inspected inference
file does not implement the training MTP objective; that comes from report v2.

RoPE originates with Su et al.; SwiGLU with Shazeer; PPO and the KL estimator with
Schulman and collaborators. MTP draws on Gloeckle et al. and EAGLE, with DeepSeek's
sequential construction distinguished in the lab. DPO, LoRA, speculative decoding,
Muon and Hyper-Connections retain their external attribution (see papers.md).

## DeepSeek reference code license

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
