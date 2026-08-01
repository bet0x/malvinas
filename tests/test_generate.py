import pytest
import torch

from malvinas.generate import _sample_token, build_parser, generate_tokens
from malvinas.model import MalvinasModel


def make_model():
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
    )


def test_greedy_generation_uses_cache_and_restores_training_state():
    torch.manual_seed(0)
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])

    output = generate_tokens(model, prompt, max_new_tokens=3, temperature=0)

    assert output.shape == (1, 6)
    assert torch.equal(output[:, :3], prompt)
    assert model.training


def test_generation_rejects_context_overflow():
    model = make_model()

    with pytest.raises(ValueError, match="context length"):
        generate_tokens(model, torch.ones((1, 7), dtype=torch.long), max_new_tokens=2)


def test_top_p_keeps_the_smallest_probability_nucleus():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    generator = torch.Generator().manual_seed(0)

    samples = {
        int(_sample_token(logits, 1.0, 0, 0.7, generator).item())
        for _ in range(20)
    }

    assert samples <= {0, 1}


def test_generation_validates_top_p_and_accepts_eos_alias():
    model = make_model()
    with pytest.raises(ValueError, match="top_p"):
        generate_tokens(model, torch.tensor([[1]]), max_new_tokens=1, top_p=0)

    args = build_parser().parse_args(
        ["--model", "model.pt", "--prompt", "hola", "--eos-token-id", "7"]
    )
    assert args.stop_token_id == 7
