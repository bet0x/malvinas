import torch

from malvinas.block import TransformerBlock
from malvinas.rope import precompute_freqs_cis


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 2, 5
    block = TransformerBlock(d_model, n_heads, num_experts=4, top_k=2, expert_dim=16)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out = block(x, freqs_cis)

    assert out.shape == x.shape


def test_gradients_flow_into_both_attention_and_moe():
    torch.manual_seed(0)
    d_model, n_heads, seq_len = 8, 2, 5
    block = TransformerBlock(d_model, n_heads, num_experts=4, top_k=2, expert_dim=16)
    freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len=seq_len, theta=10000.0)

    x = torch.randn(1, seq_len, d_model)
    out = block(x, freqs_cis)
    out.sum().backward()

    assert block.attn.qkv.weight.grad is not None
    assert block.attn.qkv.weight.grad.abs().sum() > 0
    assert block.moe.router.weight.grad is not None
    assert block.moe.router.weight.grad.abs().sum() > 0
