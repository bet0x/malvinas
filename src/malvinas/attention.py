import torch
import torch.nn as nn
from torch.nn import functional as F

from malvinas.rope import apply_rotary_emb


class Attention(nn.Module):
    """Causal multi-head attention with RoPE. Full O(T^2) attention, no
    windowing/key-reuse/GQA yet — those are separate, later increments."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, self.n_heads, 3 * self.d_k)
        q, k, v = qkv.chunk(3, dim=-1)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_k ** -0.5)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = (attn_weights @ v).permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out)
