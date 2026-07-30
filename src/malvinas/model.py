import torch
import torch.nn as nn

from malvinas.block import TransformerBlock
from malvinas.norm import RMSNorm
from malvinas.rope import precompute_freqs_cis


class MalvinasModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_dim: int,
        max_seq_len: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, num_experts, top_k, expert_dim)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # tied

        freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len, rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        T = token_ids.shape[1]
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, self.freqs_cis[:T])
        x = self.final_norm(x)
        return self.output_head(x)
