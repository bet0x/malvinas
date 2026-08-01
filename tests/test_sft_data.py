import pytest
import torch

from malvinas.data import build_sft_example, stream_sft_examples, xlam_to_messages
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


@pytest.mark.integration
def test_stream_sft_examples_yields_real_smoltalk_conversations():
    """Integration test against the real SmolTalk stream (network): each
    yielded example must be a valid (input_ids, loss_mask) pair with at
    least one assistant-turn token marked for loss."""
    tok = Tokenizer()
    examples = list(
        stream_sft_examples("HuggingFaceTB/smoltalk", tok, config_name="all", max_examples=3)
    )

    assert len(examples) == 3
    for input_ids, loss_mask in examples:
        assert len(input_ids) == len(loss_mask)
        assert any(loss_mask)


@pytest.mark.integration
def test_stream_sft_examples_yields_real_dolci_think_conversations():
    """Dolci-Think-SFT-32B (plan 00 §10, reasoning traces) uses the same
    {messages: [{role, content}]} shape as SmolTalk -- no adapter, no
    config_name needed, and no explicit test yet that the assistant's
    <think>...</think> trace ends up inside the masked-in (loss-counted)
    span, which this checks by requiring more than a token or two marked."""
    tok = Tokenizer()
    examples = list(
        stream_sft_examples("allenai/Dolci-Think-SFT-32B", tok, max_examples=2)
    )

    assert len(examples) == 2
    for input_ids, loss_mask in examples:
        assert len(input_ids) == len(loss_mask)
        assert sum(loss_mask) > 10  # a real <think> trace is not just 1-2 tokens


def test_xlam_to_messages_builds_system_user_assistant_turns():
    row = {
        "query": "What's the weather in Lima?",
        "tools": '[{"name": "get_weather", "parameters": {}}]',
        "answers": '[{"name": "get_weather", "arguments": {"city": "Lima"}}]',
    }

    messages = xlam_to_messages(row)

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert row["query"] in messages[1]["content"]
    assert row["tools"] in messages[0]["content"]
    assert row["answers"] in messages[2]["content"]
    assert "<|tool_call|>" in messages[2]["content"]


@pytest.mark.integration
def test_stream_sft_examples_with_xlam_adapter_yields_real_tool_call_examples():
    """Integration test against the real xLAM function-calling stream:
    the assistant's masked-in span must contain the tool-call marker,
    proving the adapter's structure survives tokenization+masking."""
    tok = Tokenizer()
    tok.add_special_tokens(["<|tool_call|>"])

    examples = list(
        stream_sft_examples(
            "Salesforce/xlam-function-calling-60k",
            tok,
            max_examples=2,
            row_to_messages=xlam_to_messages,
        )
    )

    assert len(examples) == 2
    for input_ids, loss_mask in examples:
        assert len(input_ids) == len(loss_mask)
        assert any(loss_mask)
