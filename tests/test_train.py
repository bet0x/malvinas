import torch
from torch.nn import functional as F

from malvinas.model import MalvinasModel
from malvinas.train import compute_loss, train_step


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
