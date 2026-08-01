import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - supported PyTorch versions provide it
    create_block_mask = None
    flex_attention = None

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
        document_attention_backend: str = "auto",
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.reuse_key_as_value = reuse_key_as_value
        if window_size is not None and window_size < 0:
            raise ValueError("window_size must be non-negative")
        self.window_size = window_size
        if document_attention_backend not in {"auto", "flex", "sdpa"}:
            raise ValueError("document_attention_backend must be auto, flex, or sdpa")
        self.document_attention_backend = document_attention_backend
        qkv_out = 2 * d_model if reuse_key_as_value else 3 * d_model
        self.qkv = nn.Linear(d_model, qkv_out, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.d_k)
        self.k_norm = RMSNorm(self.d_k)

    def _project_qkv(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if self.reuse_key_as_value:
            qkv = self.qkv(x).view(B, T, self.n_heads, 2 * self.d_k)
            q, k = qkv.chunk(2, dim=-1)
        else:
            qkv = self.qkv(x).view(B, T, self.n_heads, 3 * self.d_k)
            q, k, v = qkv.chunk(3, dim=-1)

        q = apply_rotary_emb(self.q_norm(q), freqs_cis)
        k = apply_rotary_emb(self.k_norm(k), freqs_cis)
        if self.reuse_key_as_value:
            v = k
        return (
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
        )

    def _global_document_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        document_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, _, T, _ = q.shape
        use_flex = self.document_attention_backend == "flex" or (
            self.document_attention_backend == "auto" and q.is_cuda
        )
        if use_flex:
            if flex_attention is None or create_block_mask is None:
                if self.document_attention_backend == "flex":
                    raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
            else:
                def document_causal_mask(batch, _head, query, key):
                    return (query >= key) & (
                        document_ids[batch, query] == document_ids[batch, key]
                    )

                block_mask = create_block_mask(
                    document_causal_mask,
                    B=B,
                    H=None,
                    Q_LEN=T,
                    KV_LEN=T,
                    device=q.device,
                    _compile=q.is_cuda,
                )
                return flex_attention(q, k, v, block_mask=block_mask)

        positions = torch.arange(T, device=q.device)
        causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
        same_document = document_ids.unsqueeze(1) == document_ids.unsqueeze(2)
        attention_mask = (same_document & causal).unsqueeze(1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self._project_qkv(x, freqs_cis)

        if self.window_size is None:
            if document_ids is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = self._global_document_attention(q, k, v, document_ids)
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
            if document_ids is not None:
                padded_document_ids = F.pad(document_ids, (left_context, 0), value=-1)
                document_windows = padded_document_ids.unfold(1, window_length, 1)
                valid = (
                    valid.unsqueeze(0)
                    & (document_windows == document_ids.unsqueeze(-1))
                ).unsqueeze(1)
            scores = scores.masked_fill(~valid, float("-inf"))
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = (weights.unsqueeze(-1) * v_windows).sum(dim=-2)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend new tokens to cached keys and values during autoregressive decoding."""
        B, T, C = x.shape
        q, new_k, new_v = self._project_qkv(x, freqs_cis)
        if kv_cache is None:
            past_length = 0
            k, v = new_k, new_v
        else:
            cached_k, cached_v = kv_cache
            if cached_k.shape[:2] != new_k.shape[:2]:
                raise ValueError("KV cache batch or head count does not match the input")
            past_length = cached_k.shape[2]
            k = torch.cat((cached_k, new_k), dim=2)
            v = torch.cat((cached_v, new_v), dim=2)

        key_positions = torch.arange(k.shape[2], device=x.device)
        query_positions = past_length + torch.arange(T, device=x.device)
        valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if self.window_size is not None:
            valid &= key_positions.unsqueeze(0) >= (
                query_positions.unsqueeze(1) - self.window_size
            )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid.unsqueeze(0).unsqueeze(0),
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out), (k, v)
