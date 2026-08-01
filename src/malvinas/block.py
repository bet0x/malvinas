import torch
from torch import nn

from malvinas.attention import Attention
from malvinas.moe import MoEFeedForward
from malvinas.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm attention + pre-norm MoE FFN, each with a residual connection."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_dim: int,
        window_size: int | None = None,
        reuse_key_as_value: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(
            d_model, n_heads, reuse_key_as_value=reuse_key_as_value, window_size=window_size
        )
        self.moe_norm = RMSNorm(d_model)
        self.moe = MoEFeedForward(d_model, num_experts, top_k, expert_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.moe(self.moe_norm(x))
        return x
