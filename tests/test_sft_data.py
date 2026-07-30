import torch

from malvinas.data import build_sft_example, stream_sft_examples
from malvinas.tokenizer import Tokenizer
from malvinas.train import compute_loss

# Integration-adjacent: uses the real SmolLM2 tokenizer (already used
# elsewhere), no network needed beyond that.


def test_loss_mask_is_true_only_for_assistant_turns():
    tok = Tokenizer()
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]

    input_ids, loss_mask = build_sft_example(messages, tok)

    assert len(input_ids) == len(loss_mask)
    assert any(loss_mask)  # some assistant tokens are marked
    assert not all(loss_mask)  # but not the whole sequence (user turn excluded)

    # exact accounting: the number of True positions equals exactly how many
    # tokens the assistant turn's own formatted text encodes to
    assistant_turn_ids = tok.encode("<|assistant|>\n4\n")
    assert sum(loss_mask) == len(assistant_turn_ids)


def test_masked_example_plugs_into_compute_loss():
    tok = Tokenizer()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    input_ids, loss_mask = build_sft_example(messages, tok)

    vocab_size = tok.vocab_size
    T = len(input_ids)
    logits = torch.randn(1, T, vocab_size)
    target_ids = torch.tensor([input_ids])
    mask = torch.tensor([loss_mask])

    loss = compute_loss(logits, target_ids, mask)
    assert torch.isfinite(loss)


def test_stream_sft_examples_yields_real_smoltalk_conversations():
    """Integration test against the real SmolTalk stream (network): each
    yielded example must be a valid (input_ids, loss_mask) pair with at
    least one assistant-turn token marked for loss."""
    tok = Tokenizer()
    examples = list(
        stream_sft_examples("HuggingFaceTB/smoltalk", "all", tok, max_examples=3)
    )

    assert len(examples) == 3
    for input_ids, loss_mask in examples:
        assert len(input_ids) == len(loss_mask)
        assert any(loss_mask)
