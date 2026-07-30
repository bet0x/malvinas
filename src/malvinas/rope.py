import torch


def freqs_cis_from_inv_freq(inv_freq: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Complex-exponential RoPE frequencies from an explicit inv_freq tensor
    (e.g. YaRN-rescaled), one row per position: (max_seq_len, len(inv_freq))."""
    positions = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float, rotary_pct: float = 1.0
) -> torch.Tensor:
    """Complex-exponential RoPE frequencies, one row per position:
    (max_seq_len, (dim * rotary_pct) / 2). Only the first `dim * rotary_pct`
    dims of a head get rotated when this is consumed by apply_rotary_emb —
    partial-rotary RoPE (plan 08)."""
    rotary_dim = int(dim * rotary_pct)
    freq_indices = torch.arange(0, rotary_dim, 2, dtype=torch.float)
    inv_freq = 1.0 / (theta ** (freq_indices / rotary_dim))
    return freqs_cis_from_inv_freq(inv_freq, max_seq_len)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate x (B, T, n_heads, d_k) using per-position freqs_cis (T, rotary_dim/2).
    If rotary_dim < d_k (partial-rotary), the remaining dims pass through
    unrotated."""
    rotary_dim = freqs_cis.shape[-1] * 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x_complex = torch.view_as_complex(x_rot.float().reshape(*x_rot.shape[:-1], -1, 2))
    freqs_cis_bthd = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, T, 1, rotary_dim/2)
    x_rotated = x_complex * freqs_cis_bthd
    x_rotated = torch.view_as_real(x_rotated).flatten(-2).type_as(x)

    return torch.cat([x_rotated, x_pass], dim=-1)
