import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """Complex-exponential RoPE frequencies, one row per position: (max_seq_len, dim/2)."""
    freq_indices = torch.arange(0, dim, 2, dtype=torch.float)
    inv_freq = 1.0 / (theta ** (freq_indices / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate x (B, T, n_heads, d_k) using per-position freqs_cis (T, d_k/2)."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis_bthd = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, T, 1, d_k/2)
    x_rotated = x_complex * freqs_cis_bthd
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
