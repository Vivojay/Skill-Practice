"""Rotary positions and the frequency ramp used by the V3 reference's YaRN.

The base and scaling are fixed for a run: changing them halfway through decoding
would make old cached keys inconsistent with new queries.
"""
import math
import torch


def inverse_frequencies(dim, *, base=10000., factor=1., original_length=4096,
                        beta_fast=32., beta_slow=1., scheme='yarn',
                        low_frequency=1., high_frequency=4., device=None, dtype=None):
    if dim < 2 or dim % 2 or base <= 1 or factor < 1 or original_length < 1:
        raise ValueError('Need even rotary width, base > 1, factor >= 1 and positive context')
    if not 0 < beta_slow < beta_fast:
        raise ValueError('Need 0 < beta_slow < beta_fast')
    index = torch.arange(dim // 2, device=device, dtype=dtype)
    frequencies = base ** (-2 * index / dim)
    if factor == 1:
        return frequencies
    if scheme == 'llama3':
        if not 0 < low_frequency < high_frequency:
            raise ValueError('Need increasing positive wavelength thresholds')
        rotations = original_length*frequencies/(2*math.pi)
        blend = ((rotations-low_frequency)/(high_frequency-low_frequency)).clamp(0,1)
        return frequencies*(blend+(1-blend)/factor)
    if scheme != 'yarn':
        raise ValueError('Unknown rotary scaling scheme')
    def boundary(rotations):
        return dim * math.log(original_length / (2 * math.pi * rotations)) / (2 * math.log(base))
    low = max(0, math.floor(boundary(beta_fast)))
    high = min(dim - 1, math.ceil(boundary(beta_slow)))
    ramp = ((index - low) / max(high - low, .001)).clamp(0, 1)
    return frequencies * (1 - ramp) + frequencies / factor * ramp


def rotate(x, positions, *, interleaved=True, **options):
    """Rotate [batch, time, heads, rotary_width]; positions is [time]."""
    frequencies = inverse_frequencies(x.shape[-1], device=x.device, dtype=x.dtype, **options)
    angles = positions.to(x.dtype)[:, None] * frequencies
    c, s = angles.cos()[None, :, None], angles.sin()[None, :, None]
    if interleaved:
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a*c-b*s, a*s+b*c), -1)


def attention_scale(factor):
    """V3's all-dimension logit multiplier; not a claim of long-context quality."""
    if factor < 1:
        raise ValueError('Extension factor must be >= 1')
    return (1 + .1 * math.log(factor)) ** 2
