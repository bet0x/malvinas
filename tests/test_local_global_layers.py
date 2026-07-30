import torch

from malvinas.model import MalvinasModel


def test_layers_alternate_local_global_at_configured_ratio():
    """5:1 local:global -> layers 0-4 local (windowed, full RoPE), layer 5
    global (full attention, key-as-value reuse, partial-rotary RoPE)."""
    torch.manual_seed(0)
    model = MalvinasModel(
        vocab_size=16,
        d_model=8,
        n_layers=6,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=16,
        max_seq_len=8,
        rope_theta=10000.0,
        local_global_ratio=5,
        local_window_size=2,
        global_rope_theta=1_000_000.0,
        global_rotary_pct=0.5,
    )

    for i in range(5):
        assert model.blocks[i].attn.window_size == 2
        assert model.blocks[i].attn.reuse_key_as_value is False

    assert model.blocks[5].attn.window_size is None
    assert model.blocks[5].attn.reuse_key_as_value is True


def test_forward_still_works_with_local_global_layers():
    torch.manual_seed(0)
    vocab_size, seq_len = 16, 5
    model = MalvinasModel(
        vocab_size=vocab_size,
        d_model=8,
        n_layers=6,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=16,
        max_seq_len=seq_len,
        rope_theta=10000.0,
        local_global_ratio=5,
        local_window_size=2,
        global_rope_theta=1_000_000.0,
        global_rotary_pct=0.5,
    )

    token_ids = torch.randint(0, vocab_size, (1, seq_len))
    logits = model(token_ids)

    assert logits.shape == (1, seq_len, vocab_size)
