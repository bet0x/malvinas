import pytest

from malvinas.config import PRESET_NAMES, model_config_from_preset


def test_all_model_presets_produce_valid_configs():
    for preset in PRESET_NAMES:
        config = model_config_from_preset(preset, vocab_size=128, max_seq_len=64)

        assert config.vocab_size == 128
        assert config.max_seq_len == 64
        assert config.d_model % config.n_heads == 0
        assert config.top_k <= config.num_experts


def test_tiny_preset_builds_a_model():
    config = model_config_from_preset("tiny", vocab_size=128, max_seq_len=64)

    model = config.build()

    assert model.token_embedding.num_embeddings == 128


def test_unknown_model_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown model preset"):
        model_config_from_preset("unknown", vocab_size=128, max_seq_len=64)
