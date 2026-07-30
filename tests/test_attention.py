import torch

from malvinas.attention import Attention
from malvinas.rope import precompute_freqs_cis


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 2, 5
    attn = Attention(d_model, n_heads)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out = attn(x, freqs_cis)

    assert out.shape == x.shape


def test_causal_masking_blocks_future_tokens():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 2, 5
    attn = Attention(d_model, n_heads)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out_a = attn(x, freqs_cis)

    x_changed_future = x.clone()
    x_changed_future[0, -1] = torch.randn(d_model)  # change only the last (future-most) token
    out_b = attn(x_changed_future, freqs_cis)

    # every position except the last must be unaffected by a change to the last token
    assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)
    assert not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-6)
