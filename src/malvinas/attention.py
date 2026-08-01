import torch
from torch import nn
from torch.nn import functional as F

from malvinas.norm import RMSNorm
from malvinas.rope import apply_rotary_emb


class Attention(nn.Module):
    """Causal multi-head attention with RoPE, QK-Norm and optional windowing."""

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
        if window_size is not None and window_size < 0:
            raise ValueError("window_size must be non-negative")
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

        if self.window_size is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            left_context = min(self.window_size, T - 1)
            window_length = left_context + 1
            padded_k = F.pad(k, (0, 0, left_context, 0))
            padded_v = F.pad(v, (0, 0, left_context, 0))
            k_windows = padded_k.unfold(2, window_length, 1).transpose(-2, -1)
            v_windows = padded_v.unfold(2, window_length, 1).transpose(-2, -1)

            scores = (q.unsqueeze(-2) * k_windows).sum(dim=-1) * (self.d_k ** -0.5)
            offsets = torch.arange(-left_context, 1, device=x.device)
            positions = torch.arange(T, device=x.device).unsqueeze(-1)
            valid = positions + offsets >= 0
            scores = scores.masked_fill(~valid, float("-inf"))
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = (weights.unsqueeze(-1) * v_windows).sum(dim=-2)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out)
