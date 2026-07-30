import math

import torch
import torch.nn as nn
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

    def __init__(self, d_model: int, n_heads: int, chunk_size: int, top_k: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.top_k = top_k

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _block_gate(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q, k: (H, T, D). Returns an additive gate (H, T, T): 0 where a
        key position's chunk is selected for that query, -inf otherwise."""
        H, T, D = q.shape
        C = self.chunk_size
        num_blocks = math.ceil(T / C)

        chunk_keys = torch.stack(
            [k[:, i * C : min(T, (i + 1) * C)].mean(dim=1) for i in range(num_blocks)], dim=1
        )  # (H, num_blocks, D) -- one representative key per chunk

        gate = torch.einsum("htd,hnd->htn", q.float(), chunk_keys.float())  # (H, T, num_blocks)

        for i in range(num_blocks):
            block_start, block_end = i * C, min(T, (i + 1) * C)
            gate[:, : block_end, i] = float("-inf")  # exclude: query is <= end of chunk i
            gate[:, block_start:block_end, i] = float("inf")  # force-include: query's own chunk

        top_k = min(self.top_k, num_blocks)
        top_vals, top_idx = torch.topk(gate, k=top_k, dim=-1, largest=True)
        threshold = top_vals.min(dim=-1).values  # (H, T)
        selected = gate >= threshold.unsqueeze(-1)
        selected_mask = torch.zeros_like(selected)
        selected_mask.scatter_(-1, top_idx, True)
        selected = selected & selected_mask

        block_gate = torch.where(selected, torch.zeros_like(gate), torch.full_like(gate, float("-inf")))
        return block_gate.repeat_interleave(C, dim=-1)[:, :, :T]  # (H, T, T), token-level

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n_h, d_h = self.n_heads, self.d_head

        q = self.W_q(x).view(B, T, n_h, d_h)
        k = self.W_k(x).view(B, T, n_h, d_h)
        v = self.W_v(x).view(B, T, n_h, d_h)
        q = apply_rotary_emb(q, freqs_cis).permute(0, 2, 1, 3)  # (B, n_h, T, d_h)
        k = apply_rotary_emb(k, freqs_cis).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        outputs = []
        for b in range(B):
            block_gate = self._block_gate(q[b], k[b])  # (n_h, T, T)
            scores = (q[b] @ k[b].transpose(-2, -1)) * (d_h ** -0.5) + block_gate
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1).type_as(v)
            outputs.append(weights @ v[b])  # (n_h, T, d_h)

        out = torch.stack(outputs, dim=0).permute(0, 2, 1, 3).reshape(B, T, n_h * d_h)
        return self.out_proj(out)
