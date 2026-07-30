import torch

from malvinas.model import MalvinasModel


def make_model():
    torch.manual_seed(0)
    return MalvinasModel(
        vocab_size=16,
        d_model=8,
        n_layers=2,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=16,
        max_seq_len=8,
        rope_theta=10000.0,
        mtp_depth=1,
    )


def test_forward_with_mtp_returns_main_and_mtp_logits():
    vocab_size, seq_len = 16, 5
    model = make_model()

    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    next_token_ids = torch.randint(0, vocab_size, (1, seq_len))  # ground-truth t+1 targets

    logits, mtp_logits = model.forward_with_mtp(input_ids, next_token_ids)

    assert logits.shape == (1, seq_len, vocab_size)
    assert mtp_logits.shape == (1, seq_len, vocab_size)


def test_mtp_head_receives_gradient():
    vocab_size, seq_len = 16, 5
    model = make_model()

    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    next_token_ids = torch.randint(0, vocab_size, (1, seq_len))

    _, mtp_logits = model.forward_with_mtp(input_ids, next_token_ids)
    mtp_logits.sum().backward()

    assert model.mtp_head.combine_proj.weight.grad is not None
    assert model.mtp_head.combine_proj.weight.grad.abs().sum() > 0


def test_resize_token_embeddings_keeps_mtp_head_in_sync():
    """resize_token_embeddings replaces token_embedding/output_head with new
    objects -- the mtp_head must be re-pointed at those, not left holding
    stale references to the old (smaller) ones."""
    model = make_model()
    model.resize_token_embeddings(20)

    assert model.mtp_head.token_embedding is model.token_embedding
    assert model.mtp_head.output_head is model.output_head


def test_model_without_mtp_depth_has_no_mtp_head():
    model = MalvinasModel(
        vocab_size=16,
        d_model=8,
        n_layers=2,
        n_heads=2,
        num_experts=4,
        top_k=2,
        expert_dim=16,
        max_seq_len=8,
        rope_theta=10000.0,
    )
    assert model.mtp_head is None
