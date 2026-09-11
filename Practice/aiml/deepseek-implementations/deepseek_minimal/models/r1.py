"""R1 backbones: the teacher reuses V3; dense students retain their own families.

These builders initialize small random models. A release name does not supply
its pretrained knowledge or its post-training data.
"""
from . import v3
from .decoder import Decoder, Block, GroupedAttention
from ..moe import Expert


def build(vocab=32,width=32,layers=2):
    """Shared teacher architecture for R1-Zero, R1 and R1-0528."""
    return v3.build(vocab,width,layers)


def build_student(family='qwen2',vocab=32,width=32,layers=2):
    if family not in ('qwen2','qwen3','llama3') or layers < 1:
        raise ValueError('Choose qwen2, qwen3 or llama3 and positive depth')
    eps = 1e-5 if family == 'llama3' else 1e-6
    rotary = dict(base=10000.)
    if family == 'qwen3':
        rotary = dict(base=1000000.)
    elif family == 'llama3':
        rotary = dict(base=500000.,factor=8,scheme='llama3',original_length=8192)
    blocks = []
    for _ in range(layers):
        attention = GroupedAttention(width,heads=4,kv_heads=2,rotary=rotary,
                                     qkv_bias=family=='qwen2',qk_norm=family=='qwen3',norm_eps=eps)
        blocks.append(Block(width,attention,Expert(width,3*width),eps))
    return Decoder(vocab,width,blocks,eps)
