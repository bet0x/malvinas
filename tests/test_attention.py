import torch
import pytest

from malvinas.attention import Attention
from malvinas.rope import apply_rotary_emb, precompute_freqs_cis


@pytest.mark.parametrize("window_size", [None, 3])
def test_document_mask_isolates_packed_sequences(window_size):
    torch.manual_seed(0)
    attn = Attention(8, 2, window_size=window_size)
    all_freqs = precompute_freqs_cis(4, max_seq_len=4, theta=10000.0)
    position_ids = torch.tensor([[0, 1, 0, 1]])
    freqs_cis = all_freqs[position_ids]
    document_ids = torch.tensor([[0, 0, 1, 1]])
    x = torch.randn(1, 4, 8)

    output = attn(x, freqs_cis, document_ids=document_ids)
    changed = x.clone()
    changed[:, :2] = torch.randn_like(changed[:, :2])
    changed_output = attn(changed, freqs_cis, document_ids=document_ids)

    assert torch.allclose(output[:, 2:], changed_output[:, 2:], atol=1e-6)


def test_flex_document_attention_matches_sdpa_fallback():
    torch.manual_seed(0)
    sdpa = Attention(8, 2, document_attention_backend="sdpa")
    flex = Attention(8, 2, document_attention_backend="flex")
    flex.load_state_dict(sdpa.state_dict())
    all_freqs = precompute_freqs_cis(4, max_seq_len=8, theta=10000.0)
    document_ids = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 1, 2, 2]]
    )
    position_ids = torch.tensor(
        [[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 0, 1, 2, 3, 0, 1]]
    )
    x = torch.randn(2, 8, 8)

    expected = sdpa(x, all_freqs[position_ids], document_ids=document_ids)
    actual = flex(x, all_freqs[position_ids], document_ids=document_ids)

    assert torch.allclose(actual, expected, atol=1e-5)


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


def test_qk_norm_zero_weight_gives_uniform_causal_attention():
    """With q_norm.weight zeroed, q becomes all-zero, so every attn score is 0
    and softmax over the causal-masked row is uniform -> output at position t
    is exactly the causal mean of v up to t."""
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 2, 4
    attn = Attention(d_model, n_heads)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    with torch.no_grad():
        attn.q_norm.weight.zero_()

    x = torch.randn(1, seq_len, d_model)
    out = attn(x, freqs_cis)

    with torch.no_grad():
        qkv = attn.qkv(x).view(1, seq_len, n_heads, 3 * attn.d_k)
        _, _, v = qkv.chunk(3, dim=-1)
        v = v.permute(0, 2, 1, 3)  # (1, n_heads, T, d_k)

    for t in range(seq_len):
        expected = v[:, :, : t + 1].mean(dim=2)  # (1, n_heads, d_k)
        expected = expected.reshape(1, d_model)
        expected_out = attn.out_proj(expected)
        assert torch.allclose(out[0, t], expected_out[0], atol=1e-5)


def test_reuse_key_as_value_shrinks_qkv_projection():
    d_model, n_heads = 8, 2
    full_attn = Attention(d_model, n_heads, reuse_key_as_value=False)
    reused_attn = Attention(d_model, n_heads, reuse_key_as_value=True)

    assert full_attn.qkv.weight.shape == (3 * d_model, d_model)
    assert reused_attn.qkv.weight.shape == (2 * d_model, d_model)


def test_reuse_key_as_value_uses_key_projection_as_value():
    """With a single-token sequence, causal attention weight is exactly 1.0
    on itself, so the output must equal out_proj(k) when v is defined as k."""
    torch.manual_seed(0)
    d_model, n_heads = 8, 2
    attn = Attention(d_model, n_heads, reuse_key_as_value=True)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=1, theta=10000.0)

    x = torch.randn(1, 1, d_model)
    out = attn(x, freqs_cis)

    with torch.no_grad():
        qkv = attn.qkv(x).view(1, 1, n_heads, 2 * attn.d_k)
        q, k = qkv.chunk(2, dim=-1)
        k = attn.k_norm(k)
        k = apply_rotary_emb(k, freqs_cis)
        expected = attn.out_proj(k.reshape(1, 1, d_model))

    assert torch.allclose(out, expected, atol=1e-5)


def test_window_size_blocks_positions_further_back_than_window():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 2, 5
    attn = Attention(d_model, n_heads, window_size=1)  # only self + 1 position back
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out_a = attn(x, freqs_cis)

    x_changed_pos0 = x.clone()
    x_changed_pos0[0, 0] = torch.randn(d_model)
    out_b = attn(x_changed_pos0, freqs_cis)

    # position 0 is outside the window (>1 step back) for positions 2, 3, 4
    assert torch.allclose(out_a[0, 2:], out_b[0, 2:], atol=1e-6)
    # but position 1 is within the window (exactly 1 step back) -> must change
    assert not torch.allclose(out_a[0, 1], out_b[0, 1], atol=1e-6)
