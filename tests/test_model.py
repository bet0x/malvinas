import torch

from malvinas.model import MalvinasModel


def make_model(vocab_size=16, d_model=8, n_heads=2, seq_len=5):
    return MalvinasModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
        num_experts=4,
        top_k=2,
        expert_dim=16,
        max_seq_len=seq_len,
        rope_theta=10000.0,
    )


def test_forward_returns_logits_over_vocab():
    torch.manual_seed(0)
    vocab_size, seq_len = 16, 5
    model = make_model(vocab_size=vocab_size, seq_len=seq_len)

    token_ids = torch.randint(0, vocab_size, (1, seq_len))
    logits = model(token_ids)

    assert logits.shape == (1, seq_len, vocab_size)


def test_embedding_and_output_head_are_tied():
    torch.manual_seed(0)
    model = make_model()

    assert model.output_head.weight is model.token_embedding.weight


def test_resize_token_embeddings_preserves_existing_rows_and_stays_tied():
    """Growing the vocab (e.g. to add tool-call special tokens, plan 00 §9)
    must keep every existing row's weights exactly as they were, keep the
    output head tied to the (new) embedding, and still work in a forward
    pass on token ids in the newly added range."""
    torch.manual_seed(0)
    old_vocab_size = 16
    model = make_model(vocab_size=old_vocab_size)
    old_weights = model.token_embedding.weight.data.clone()

    model.resize_token_embeddings(old_vocab_size + 4)

    assert model.token_embedding.weight.shape == (old_vocab_size + 4, 8)
    assert torch.equal(model.token_embedding.weight.data[:old_vocab_size], old_weights)
    assert model.output_head.weight is model.token_embedding.weight

    token_ids = torch.tensor([[old_vocab_size, old_vocab_size + 1]])  # newly added ids
    logits = model(token_ids)
    assert logits.shape == (1, 2, old_vocab_size + 4)
