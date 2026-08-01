import torch
from torch import nn
from torch.nn import functional as F

from malvinas.rope import apply_rotary_emb


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, arXiv:2405.04434, Section
    2.1, eq. 9-19). Implemented directly against the paper's own equations
    (fetched and transcribed, not from memory) -- plan 10.

    KV is compressed through a shared low-rank latent c_t^{KV} (eq. 9) and
    reconstructed per-head via up-projections (eq. 10-11). Q gets the same
    treatment for training-memory savings (eq. 12-13). RoPE is "decoupled":
    a small per-head rotary query (eq. 14) is concatenated with the
    per-head content query, and a *single, shared-across-heads* rotary key
    (eq. 16 -- no head index) is concatenated with each head's own content
    key (eq. 15, 17). Attention and output projection follow eq. 18-19.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_rope_head: int,
        d_kv_latent: int,
        d_q_latent: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_rope_head = d_rope_head

        # eq. 9-11: KV compression
        self.W_DKV = nn.Linear(d_model, d_kv_latent, bias=False)
        self.W_UK = nn.Linear(d_kv_latent, n_heads * d_head, bias=False)
        self.W_UV = nn.Linear(d_kv_latent, n_heads * d_head, bias=False)

        # eq. 12-13: Q compression
        self.W_DQ = nn.Linear(d_model, d_q_latent, bias=False)
        self.W_UQ = nn.Linear(d_q_latent, n_heads * d_head, bias=False)

        # eq. 14, 16: decoupled RoPE -- per-head query, single shared key
        self.W_QR = nn.Linear(d_q_latent, n_heads * d_rope_head, bias=False)
        self.W_KR = nn.Linear(d_model, d_rope_head, bias=False)

        # eq. 19
        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        n_h, d_h, d_r = self.n_heads, self.d_head, self.d_rope_head

        c_kv = self.W_DKV(x)
        k_content = self.W_UK(c_kv).view(B, T, n_h, d_h)
        v_content = self.W_UV(c_kv).view(B, T, n_h, d_h)

        c_q = self.W_DQ(x)
        q_content = self.W_UQ(c_q).view(B, T, n_h, d_h)

        q_rope = apply_rotary_emb(self.W_QR(c_q).view(B, T, n_h, d_r), freqs_cis)
        k_rope = apply_rotary_emb(self.W_KR(x).view(B, T, 1, d_r), freqs_cis)
        k_rope = k_rope.expand(B, T, n_h, d_r)  # eq. 17: same rope key, every head

        q = torch.cat([q_content, q_rope], dim=-1).permute(0, 2, 1, 3)  # (B, n_h, T, d_h+d_r)
        k = torch.cat([k_content, k_rope], dim=-1).permute(0, 2, 1, 3)
        v = v_content.permute(0, 2, 1, 3)  # (B, n_h, T, d_h) -- no rope part

        attn_scores = (q @ k.transpose(-2, -1)) * ((d_h + d_r) ** -0.5)  # eq. 18 denominator
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = (attn_weights @ v).permute(0, 2, 1, 3).contiguous().view(B, T, n_h * d_h)
        return self.W_O(out)
