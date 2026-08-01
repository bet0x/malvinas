import torch

from malvinas.moe import MoEFeedForward


def test_output_shape_matches_input():
    torch.manual_seed(0)
    d_model, seq_len = 8, 3
    moe = MoEFeedForward(d_model, num_experts=4, top_k=2, expert_dim=16)

    x = torch.randn(1, seq_len, d_model)
    out = moe(x)

    assert out.shape == x.shape


def test_grouped_experts_match_per_assignment_reference():
    torch.manual_seed(0)
    d_model, seq_len = 8, 4
    moe = MoEFeedForward(d_model, num_experts=4, top_k=2, expert_dim=16)
    x = torch.randn(2, seq_len, d_model)

    actual = moe(x)

    x_flat = x.reshape(-1, d_model)
    router_logits = moe.router(x_flat)
    selected = torch.topk(router_logits + moe.expert_bias, moe.top_k, dim=-1).indices
    weights = torch.sigmoid(torch.gather(router_logits, -1, selected)).reshape(-1)
    token_idx = torch.arange(x_flat.shape[0]).repeat_interleave(moe.top_k)
    expert_idx = selected.reshape(-1)
    expert_inputs = x_flat[token_idx]
    gate_up = torch.bmm(
        expert_inputs.unsqueeze(1), moe.expert_gate_up_proj[expert_idx]
    )
    gate, up = gate_up.chunk(2, dim=-1)
    routed = torch.bmm(
        (moe.act(gate) * up), moe.expert_down_proj[expert_idx]
    ).squeeze(1)
    routed = routed * weights.unsqueeze(-1)
    expected_routed = torch.zeros_like(x_flat)
    expected_routed.index_add_(0, token_idx, routed)
    expected = expected_routed.view_as(x) + moe.shared_expert(x)

    assert torch.allclose(actual, expected, atol=1e-6)


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


def test_forward_supports_cuda_bfloat16_autocast():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return

    layer = MoEFeedForward(8, num_experts=4, top_k=2, expert_dim=16).cuda()
    x = torch.randn(2, 3, 8, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
