import torch

from malvinas.mla import MLAAttention
from malvinas.rope import precompute_freqs_cis


def make_mla(d_model=16, n_heads=4, d_head=4, d_rope_head=2, d_kv_latent=6, d_q_latent=8):
    return MLAAttention(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_rope_head=d_rope_head,
        d_kv_latent=d_kv_latent,
        d_q_latent=d_q_latent,
    )


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, d_rope_head, seq_len = 16, 2, 5
    mla = make_mla(d_model=d_model, d_rope_head=d_rope_head)
    freqs_cis = precompute_freqs_cis(d_rope_head, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out = mla(x, freqs_cis)

    assert out.shape == x.shape


def test_causal_masking_blocks_future_tokens():
    torch.manual_seed(0)
    d_model, d_rope_head, seq_len = 16, 2, 5
    mla = make_mla(d_model=d_model, d_rope_head=d_rope_head)
    freqs_cis = precompute_freqs_cis(d_rope_head, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out_a = mla(x, freqs_cis)

    x_changed_future = x.clone()
    x_changed_future[0, -1] = torch.randn(d_model)
    out_b = mla(x_changed_future, freqs_cis)

    assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)
    assert not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-6)


def test_kv_latent_is_genuinely_compressed_below_full_kv_width():
    """DeepSeek-V2 eq. 9: d_c must be << n_h * d_h for MLA to actually save
    KV-cache space -- this checks the down-projection's own output width,
    not just that we passed a smaller number somewhere."""
    d_model, n_heads, d_head, d_kv_latent = 16, 4, 4, 6
    mla = make_mla(d_model=d_model, n_heads=n_heads, d_head=d_head, d_kv_latent=d_kv_latent)

    assert mla.W_DKV.weight.shape == (d_kv_latent, d_model)
    assert d_kv_latent < n_heads * d_head


def test_rope_key_projection_is_shared_across_heads_not_per_head():
    """DeepSeek-V2 eq. 16: k_t^R = RoPE(W^{KR} h_t) has no head index --
    one shared rope-key vector for every head, unlike the per-head content
    key. The weight matrix's output width must be d_rope_head, not
    n_heads * d_rope_head."""
    d_model, n_heads, d_rope_head = 16, 4, 2
    mla = make_mla(d_model=d_model, n_heads=n_heads, d_rope_head=d_rope_head)

    assert mla.W_KR.weight.shape == (d_rope_head, d_model)
