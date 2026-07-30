from malvinas.tokenizer import Tokenizer

# Integration test: downloads (and caches) the real SmolLM2 tokenizer from
# Hugging Face on first run. Requires network access.


def test_vocab_size_matches_smollm2():
    tok = Tokenizer()
    assert tok.vocab_size == 49152


def test_encode_decode_roundtrip():
    tok = Tokenizer()
    text = "Hello Malvinas"

    ids = tok.encode(text)
    decoded = tok.decode(ids)

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert decoded == text


def test_add_special_tokens_grows_vocab_and_encodes_atomically():
    """Tool-call markers (plan 00 §9, Llama 3.1 style: <|python_tag|>,
    <|eom_id|>) must become new, single, dedicated token ids -- not get
    split into subword pieces."""
    tok = Tokenizer()
    before = tok.vocab_size

    added = tok.add_special_tokens(["<|tool_call|>", "<|tool_response|>"])

    assert added == 2
    assert tok.vocab_size == before + 2

    ids = tok.encode("<|tool_call|>")
    assert len(ids) == 1
    assert ids[0] >= before
