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


def test_expert_bias_is_added_to_router_logits_before_topk():
    """A large bias on one expert must force it to always be selected,
    regardless of the router's learned weights."""
    torch.manual_seed(0)
    d_model, seq_len = 8, 5
    moe = MoEFeedForward(d_model, num_experts=4, top_k=1, expert_dim=16)

    with torch.no_grad():
        moe.expert_bias[2] = 1000.0

    x = torch.randn(1, seq_len, d_model)
    moe(x)

    assert torch.all(moe.last_selected_experts == 2)


def test_update_expert_bias_nudges_by_update_rate_based_on_load():
    """DeepSeek-V3-style: overloaded experts (selection fraction above the
    uniform target) get their bias decreased, underloaded ones increased,
    by a fixed step -- no gradient involved."""
    torch.manual_seed(0)
    d_model = 8
    moe = MoEFeedForward(d_model, num_experts=4, top_k=1, expert_dim=16)

    # simulate a batch where every one of 4 tokens picked expert 0
    moe.last_selected_experts = torch.tensor([0, 0, 0, 0])

    before = moe.expert_bias.clone()
    moe.update_expert_bias(update_rate=0.01)

    assert torch.isclose(moe.expert_bias[0], before[0] - 0.01)
    assert torch.isclose(moe.expert_bias[1], before[1] + 0.01)
    assert torch.isclose(moe.expert_bias[2], before[2] + 0.01)
    assert torch.isclose(moe.expert_bias[3], before[3] + 0.01)
