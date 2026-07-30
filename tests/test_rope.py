import torch

from malvinas.rope import apply_rotary_emb, precompute_freqs_cis


def test_position_zero_is_unrotated():
    d_k = 4
    freqs_cis = precompute_freqs_cis(d_k, max_seq_len=8, theta=10000.0)

    x = torch.randn(1, 1, 1, d_k)  # (B, T=1 at position 0, n_heads, d_k)
    out = apply_rotary_emb(x, freqs_cis[:1])

    assert torch.allclose(out, x, atol=1e-6)


def test_rotation_preserves_vector_norm():
    d_k = 4
    seq_len = 8
    freqs_cis = precompute_freqs_cis(d_k, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, 2, d_k)  # (B, T, n_heads, d_k)
    out = apply_rotary_emb(x, freqs_cis)

    in_norm = x.pow(2).sum(-1).sqrt()
    out_norm = out.pow(2).sum(-1).sqrt()
    assert torch.allclose(in_norm, out_norm, atol=1e-5)
