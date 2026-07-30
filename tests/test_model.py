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
