import torch

from malvinas.moba import MoBAAttention
from malvinas.rope import precompute_freqs_cis


def make_moba(d_model=16, n_heads=2, chunk_size=2, top_k=1):
    return MoBAAttention(d_model=d_model, n_heads=n_heads, chunk_size=chunk_size, top_k=top_k)


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 16, 2, 8
    moba = make_moba(d_model=d_model, n_heads=n_heads, chunk_size=2, top_k=1)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out = moba(x, freqs_cis)

    assert out.shape == x.shape


def test_causal_masking_blocks_future_tokens():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 16, 2, 8
    moba = make_moba(d_model=d_model, n_heads=n_heads, chunk_size=2, top_k=1)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out_a = moba(x, freqs_cis)

    x_changed_future = x.clone()
    x_changed_future[0, -1] = torch.randn(d_model)
    out_b = moba(x_changed_future, freqs_cis)

    assert torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)
    assert not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-6)


def test_own_block_is_always_attended_even_with_low_relevance_score():
    """MoBA (moba_naive.py reference): a query's own chunk is force-included
    (gate set to +inf) regardless of the computed relevance score, so it's
    never dropped by top-k even when top_k=1 and another block scores far
    higher. Verify by zeroing the query projection (making every relevance
    score identically 0, a tie) with chunk_size=2, top_k=1, seq_len=4 --
    a query in the second chunk (positions 2-3) must still be influenced by
    its own chunk-mate's value, not only by chunk 0."""
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 1, 4
    moba = make_moba(d_model=d_model, n_heads=n_heads, chunk_size=2, top_k=1)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    with torch.no_grad():
        moba.W_q.weight.zero_()  # every query becomes the zero vector -> tied relevance scores

    x = torch.randn(1, seq_len, d_model)
    out_a = moba(x, freqs_cis)

    x_changed = x.clone()
    x_changed[0, 2] = torch.randn(d_model)  # change the chunk-1-mate of query position 3
    out_b = moba(x_changed, freqs_cis)

    # position 3 (query) shares chunk 1 with position 2 -- if chunk 1 were
    # ever dropped in favor of only chunk 0, this would stay unchanged
    assert not torch.allclose(out_a[0, 3], out_b[0, 3], atol=1e-6)


def test_block_routing_returns_compact_block_ids_instead_of_token_mask():
    torch.manual_seed(0)
    moba = make_moba(d_model=16, n_heads=2, chunk_size=2, top_k=2)
    q = torch.randn(2, 8, 8)
    k = torch.randn(2, 8, 8)
    chunk_keys = torch.stack(
        [k[:, start : start + 2].mean(dim=1) for start in range(0, 8, 2)], dim=1
    )

    selected = moba._select_blocks(q, chunk_keys, torch.arange(8))

    assert selected.shape == (2, 8, 2)
    current_blocks = torch.arange(8) // 2
    assert torch.all((selected == current_blocks.view(1, 8, 1)).any(dim=-1))
