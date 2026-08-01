import torch
from torch import nn
from torch.nn import functional as F

from malvinas.norm import RMSNorm
from malvinas.rope import apply_rotary_emb


class Attention(nn.Module):
    """Causal multi-head attention with RoPE and QK-Norm. Full O(T^2)
    attention, no windowing/key-reuse/GQA yet — those are separate, later
    increments."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        reuse_key_as_value: bool = False,
        window_size: int | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.reuse_key_as_value = reuse_key_as_value
        self.window_size = window_size
        qkv_out = 2 * d_model if reuse_key_as_value else 3 * d_model
        self.qkv = nn.Linear(d_model, qkv_out, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.d_k)
        self.k_norm = RMSNorm(self.d_k)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.reuse_key_as_value:
            qkv = self.qkv(x).view(B, T, self.n_heads, 2 * self.d_k)
            q, k = qkv.chunk(2, dim=-1)
        else:
            qkv = self.qkv(x).view(B, T, self.n_heads, 3 * self.d_k)
            q, k, v = qkv.chunk(3, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if self.reuse_key_as_value:
            v = k

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_k ** -0.5)
        allowed_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        if self.window_size is not None:
            allowed_mask &= torch.triu(allowed_mask, diagonal=-self.window_size)
        attn_scores = attn_scores.masked_fill(~allowed_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = (attn_weights @ v).permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out)
