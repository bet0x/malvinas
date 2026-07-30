from tokenizers import Tokenizer as HFTokenizer

DEFAULT_REPO_ID = "HuggingFaceTB/SmolLM2-135M"


class Tokenizer:
    """Thin wrapper around the SmolLM2 tokenizer (plan 01): a modern,
    appropriately-sized BPE vocab (49,152) for a sub-1B model, instead of
    GPT-2's outdated one or a needlessly huge 128-200K vocab."""

    def __init__(self, repo_id: str = DEFAULT_REPO_ID):
        self._hf_tokenizer = HFTokenizer.from_pretrained(repo_id)

    @property
    def vocab_size(self) -> int:
        return self._hf_tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._hf_tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._hf_tokenizer.decode(ids)

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Add new atomic tokens (e.g. tool-call markers, plan 00 §9).
        Returns how many new ids were actually added."""
        return self._hf_tokenizer.add_special_tokens(tokens)
