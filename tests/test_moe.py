import torch

from malvinas.moe import MoEFeedForward


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, seq_len = 8, 3
    moe = MoEFeedForward(d_model, num_experts=4, top_k=2, expert_dim=16)

    x = torch.randn(1, seq_len, d_model)
    out = moe(x)

    assert out.shape == x.shape


def test_zeroed_routed_experts_isolates_shared_expert_contribution():
    torch.manual_seed(0)
    d_model, seq_len = 8, 3
    moe = MoEFeedForward(d_model, num_experts=4, top_k=2, expert_dim=16)

    with torch.no_grad():
        moe.expert_gate_up_proj.zero_()
        moe.expert_down_proj.zero_()

    x = torch.randn(1, seq_len, d_model)
    out = moe(x)
    expected = moe.shared_expert(x)

    assert torch.allclose(out, expected, atol=1e-6)
