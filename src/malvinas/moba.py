import math

import torch
from torch import nn
from torch.nn import functional as F

from malvinas.rope import apply_rotary_emb


class MoBAAttention(nn.Module):
    """Mixture of Block Attention (Moonshot AI, arXiv:2502.13189;
    reference implementation fetched directly from
    github.com/MoonshotAI/MoBA, moba/moba_naive.py -- "a clean version of
    moba implementation for educational purposes") -- plan 12.

    Adapted from that reference to this project's padded (B, T, n_heads,
    d_head) tensor convention instead of the original's variable-length
    (cu_seqlens) batching, and with RoPE applied to q/k before the block
    routing (as a real caller would). The routing algorithm itself --
    per-chunk mean-key gate, forced inclusion of the query's own chunk,
    exclusion of future chunks, top-k selection over the rest -- is kept
    faithful to the source.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int,
        top_k: int,
        query_chunk_size: int = 16,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        if chunk_size <= 0 or top_k <= 0 or query_chunk_size <= 0:
            raise ValueError("chunk_size, top_k and query_chunk_size must be positive")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.query_chunk_size = query_chunk_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _select_blocks(
        self,
        q: torch.Tensor,
        chunk_keys: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return selected block ids as (H, Q, K), never a token-level mask."""
        H, Q, _ = q.shape
        num_blocks = chunk_keys.shape[1]
        gate = torch.einsum("hqd,hnd->hqn", q.float(), chunk_keys.float())

        current_blocks = torch.div(query_positions, self.chunk_size, rounding_mode="floor")
        block_ids = torch.arange(num_blocks, device=q.device)
        allowed = block_ids.view(1, 1, -1) <= current_blocks.view(1, Q, 1)
        gate = gate.masked_fill(~allowed, float("-inf"))
        current_idx = current_blocks.view(1, Q, 1).expand(H, -1, -1)
        gate.scatter_(-1, current_idx, float("inf"))

        return torch.topk(gate, k=min(self.top_k, num_blocks), dim=-1).indices

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n_h, d_h = self.n_heads, self.d_head

        q = self.W_q(x).view(B, T, n_h, d_h)
        k = self.W_k(x).view(B, T, n_h, d_h)
        v = self.W_v(x).view(B, T, n_h, d_h)
        q = apply_rotary_emb(q, freqs_cis).permute(0, 2, 1, 3)  # (B, n_h, T, d_h)
        k = apply_rotary_emb(k, freqs_cis).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        num_blocks = math.ceil(T / self.chunk_size)
        token_offsets = torch.arange(self.chunk_size, device=x.device)
        outputs = []
        for b in range(B):
            chunk_keys = torch.stack(
                [
                    k[b, :, i * self.chunk_size : min(T, (i + 1) * self.chunk_size)].mean(
                        dim=1
                    )
                    for i in range(num_blocks)
                ],
                dim=1,
            )
            batch_output = []
            for start in range(0, T, self.query_chunk_size):
                end = min(T, start + self.query_chunk_size)
                query_positions = torch.arange(start, end, device=x.device)
                q_chunk = q[b, :, start:end]
                selected_blocks = self._select_blocks(q_chunk, chunk_keys, query_positions)

                token_idx = selected_blocks.unsqueeze(-1) * self.chunk_size + token_offsets
                token_idx = token_idx.flatten(start_dim=-2)
                valid = (token_idx < T) & (token_idx <= query_positions.view(1, -1, 1))
                safe_idx = token_idx.clamp(max=T - 1)

                gather_shape = (*safe_idx.shape, d_h)
                k_source = k[b].unsqueeze(1).expand(-1, end - start, -1, -1)
                v_source = v[b].unsqueeze(1).expand(-1, end - start, -1, -1)
                selected_k = torch.gather(
                    k_source, 2, safe_idx.unsqueeze(-1).expand(gather_shape)
                )
                selected_v = torch.gather(
                    v_source, 2, safe_idx.unsqueeze(-1).expand(gather_shape)
                )

                scores = (q_chunk.unsqueeze(-2) * selected_k).sum(dim=-1) * (d_h ** -0.5)
                scores = scores.masked_fill(~valid, float("-inf"))
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
                batch_output.append((weights.unsqueeze(-1) * selected_v).sum(dim=-2))

            outputs.append(torch.cat(batch_output, dim=1))

        out = torch.stack(outputs, dim=0).permute(0, 2, 1, 3).reshape(B, T, n_h * d_h)
        return self.out_proj(out)
