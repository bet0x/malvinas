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
        local_global_ratio: int | None = None,
        local_window_size: int | None = None,
        global_rope_theta: float | None = None,
        global_rotary_pct: float = 1.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # every (local_global_ratio + 1)-th layer is global (Gemma 4's 5:1 split);
        # if no ratio is given, every layer is "local" with full RoPE, no windowing.
        self.is_global_layer = [
            local_global_ratio is not None and (i + 1) % (local_global_ratio + 1) == 0
            for i in range(n_layers)
        ]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    num_experts,
                    top_k,
                    expert_dim,
                    window_size=None if is_global else local_window_size,
                    reuse_key_as_value=is_global,
                )
                for is_global in self.is_global_layer
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # tied

        d_k = d_model // n_heads
        freqs_cis_local = precompute_freqs_cis(d_k, max_seq_len, rope_theta)
        self.register_buffer("freqs_cis_local", freqs_cis_local, persistent=False)
        freqs_cis_global = precompute_freqs_cis(
            d_k, max_seq_len, global_rope_theta or rope_theta, rotary_pct=global_rotary_pct
        )
        self.register_buffer("freqs_cis_global", freqs_cis_global, persistent=False)

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Grow the vocab (e.g. adding tool-call special tokens, plan 00
        §9), keeping every existing row's weights and the tied output head."""
        old_embedding = self.token_embedding
        old_vocab_size, d_model = old_embedding.weight.shape

        new_embedding = nn.Embedding(new_vocab_size, d_model)
        with torch.no_grad():
            new_embedding.weight[:old_vocab_size] = old_embedding.weight

        self.token_embedding = new_embedding
        self.output_head = nn.Linear(d_model, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # re-tie

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        T = token_ids.shape[1]
        x = self.token_embedding(token_ids)
        for block, is_global in zip(self.blocks, self.is_global_layer):
            freqs_cis = self.freqs_cis_global if is_global else self.freqs_cis_local
            x = block(x, freqs_cis[:T])
        x = self.final_norm(x)
        return self.output_head(x)
