import torch
from torch import nn

from malvinas.block import TransformerBlock
from malvinas.norm import RMSNorm


class MTPHead(nn.Module):
    """DeepSeek-V3-style Multi-Token Prediction head (plan 09): combines the
    main trunk's hidden state at position t with the embedding of the
    already-known ground-truth token t+1, runs one more transformer block,
    and predicts t+2 through the (shared, tied) output head."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_dim: int,
        token_embedding: nn.Embedding,
        output_head: nn.Linear,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.output_head = output_head
        self.combine_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.combine_norm = RMSNorm(d_model)
        self.block = TransformerBlock(d_model, n_heads, num_experts, top_k, expert_dim)
        self.final_norm = RMSNorm(d_model)

    def forward(
        self, hidden_state: torch.Tensor, next_token_ids: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        next_emb = self.token_embedding(next_token_ids)
        combined = self.combine_norm(self.combine_proj(torch.cat([hidden_state, next_emb], dim=-1)))
        out = self.block(combined, freqs_cis)
        out = self.final_norm(out)
        return self.output_head(out)
