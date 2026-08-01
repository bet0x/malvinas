import pytest
import torch
from torch.nn import functional as F

from malvinas.model import MalvinasModel
from malvinas.train import (
    WarmupCosineScheduler,
    accumulate_expert_counts,
    build_optimizer,
    compute_loss,
    compute_training_loss,
    optimizer_step,
    train_step,
    update_expert_bias,
)


def make_tiny_model():
    torch.manual_seed(0)
    return MalvinasModel(
        vocab_size=16,
        d_model=16,
        n_layers=2,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=32,
        max_seq_len=8,
        rope_theta=10000.0,
    )


def test_loss_decreases_when_overfitting_a_single_batch():
    torch.manual_seed(0)
    model = make_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    input_ids = torch.randint(0, 16, (2, 6))
    target_ids = torch.randint(0, 16, (2, 6))

    first_loss = train_step(model, optimizer, input_ids, target_ids)
    for _ in range(30):
        last_loss = train_step(model, optimizer, input_ids, target_ids)

    assert last_loss < first_loss


def test_train_step_updates_expert_bias():
    torch.manual_seed(0)
    model = make_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    input_ids = torch.randint(0, 16, (2, 6))
    target_ids = torch.randint(0, 16, (2, 6))

    biases_before = [block.moe.expert_bias.clone() for block in model.blocks]
    train_step(model, optimizer, input_ids, target_ids)
    biases_after = [block.moe.expert_bias.clone() for block in model.blocks]

    for before, after in zip(biases_before, biases_after):
        assert not torch.equal(before, after)


def test_train_step_with_mtp_target_trains_the_mtp_head():
    """plan 09: when the model has an mtp_head and mtp_target_ids (the
    ground-truth t+2 tokens) is passed, train_step must also backprop into
    the MTP head -- not silently ignore it."""
    torch.manual_seed(0)
    model = MalvinasModel(
        vocab_size=16,
        d_model=16,
        n_layers=2,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=32,
        max_seq_len=8,
        rope_theta=10000.0,
        mtp_depth=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    input_ids = torch.randint(0, 16, (2, 6))
    target_ids = torch.randint(0, 16, (2, 6))  # t+1
    mtp_target_ids = torch.randint(0, 16, (2, 6))  # t+2

    train_step(model, optimizer, input_ids, target_ids, mtp_target_ids=mtp_target_ids)

    assert model.mtp_head.combine_proj.weight.grad is not None
    assert model.mtp_head.combine_proj.weight.grad.abs().sum() > 0


def test_expert_count_accumulation_skips_unused_mtp_head():
    model = MalvinasModel(
        vocab_size=16,
        d_model=16,
        n_layers=2,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=32,
        max_seq_len=8,
        rope_theta=10000.0,
        mtp_depth=1,
    )
    model(torch.randint(0, 16, (2, 6)))

    counts = accumulate_expert_counts(model)
    assert len(counts) == len(model.blocks)
    assert model.mtp_head.block.moe not in counts

    update_expert_bias(model, counts)
    assert torch.equal(
        model.mtp_head.block.moe.expert_bias,
        torch.zeros_like(model.mtp_head.block.moe.expert_bias),
    )


def test_compute_loss_without_mask_matches_plain_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16)
    target_ids = torch.randint(0, 16, (2, 4))

    loss = compute_loss(logits, target_ids)
    expected = F.cross_entropy(logits.view(-1, 16), target_ids.view(-1))

    assert torch.isclose(loss, expected)


def test_compute_loss_with_mask_ignores_masked_positions():
    """SFT loss masking: positions where loss_mask is False (e.g. the user's
    prompt tokens) must not influence the loss at all -- corrupting their
    targets should change nothing."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 16)
    target_ids = torch.tensor([[3, 7, 2, 9]])
    loss_mask = torch.tensor([[False, False, True, True]])

    loss_a = compute_loss(logits, target_ids, loss_mask)

    corrupted_targets = torch.tensor([[15, 15, 2, 9]])  # only masked-out positions changed
    loss_b = compute_loss(logits, corrupted_targets, loss_mask)

    expected = F.cross_entropy(logits[0, 2:], target_ids[0, 2:])

    assert torch.isclose(loss_a, loss_b)
    assert torch.isclose(loss_a, expected)


def test_compute_loss_with_fully_masked_batch_is_differentiable_zero():
    logits = torch.randn(2, 3, 8, requires_grad=True)
    target_ids = torch.randint(0, 8, (2, 3))
    loss_mask = torch.zeros(2, 3, dtype=torch.bool)

    loss = compute_loss(logits, target_ids, loss_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_warmup_cosine_scheduler_reaches_floor_and_restores_state():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_lr=1.0,
        min_lr=0.1,
        warmup_steps=2,
        decay_steps=6,
    )

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert rates[-1] == pytest.approx(0.1)

    restored = WarmupCosineScheduler(
        optimizer,
        max_lr=2.0,
        min_lr=0.0,
        warmup_steps=0,
        decay_steps=1,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.step_num == scheduler.step_num
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_build_optimizer_groups_parameters_once_with_expected_decay():
    model = make_tiny_model()
    optimizer = build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.1,
        embedding_weight_decay=0.02,
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    parameter_ids = [
        id(parameter)
        for group in groups.values()
        for parameter in group["params"]
    ]

    assert len(parameter_ids) == len(set(parameter_ids)) == len(
        list(model.parameters())
    )
    assert model.token_embedding.weight in groups["embedding"]["params"]
    assert groups["matrix"]["weight_decay"] == pytest.approx(0.1)
    assert groups["embedding"]["weight_decay"] == pytest.approx(0.02)
    assert groups["scalar"]["weight_decay"] == 0.0


def test_optimizer_step_clips_gradients_and_rejects_nonfinite_values():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    original_norm = optimizer_step(model, optimizer, max_grad_norm=0.5)
    clipped_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    )
    assert original_norm > 0.5
    assert clipped_norm <= 0.50001

    optimizer.zero_grad(set_to_none=True)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite gradients"):
        optimizer_step(model, optimizer, max_grad_norm=1.0)


def test_compute_training_loss_rejects_nonfinite_loss():
    class NonFiniteModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            shape = (*input_ids.shape, 4)
            return torch.full(shape, float("nan"), device=input_ids.device)

    input_ids = torch.tensor([[0, 1]])
    target_ids = torch.tensor([[1, 2]])
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        compute_training_loss(
            NonFiniteModel(),
            input_ids,
            target_ids,
            loss_mask,
        )
