from dataclasses import asdict, dataclass

from malvinas.model import MalvinasModel


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    num_experts: int
    top_k: int
    expert_dim: int
    max_seq_len: int
    rope_theta: float = 10000.0
    local_global_ratio: int | None = 5
    local_window_size: int | None = 1024
    global_rope_theta: float | None = None
    global_rotary_pct: float = 1.0
    mtp_depth: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        return cls(**values)

    def build(self) -> MalvinasModel:
        return MalvinasModel(**self.to_dict())


PRESET_NAMES = ("tiny", "0.5b", "1b-deep", "1b-wide")


def model_config_from_preset(
    preset: str, vocab_size: int, max_seq_len: int
) -> ModelConfig:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    dimensions = {
        "tiny": dict(
            d_model=64, n_layers=2, n_heads=4, num_experts=4, top_k=2, expert_dim=128
        ),
        "0.5b": dict(
            d_model=512, n_layers=16, n_heads=8, num_experts=32, top_k=4, expert_dim=512
        ),
        "1b-deep": dict(
            d_model=512, n_layers=32, n_heads=8, num_experts=32, top_k=4, expert_dim=512
        ),
        "1b-wide": dict(
            d_model=768, n_layers=20, n_heads=12, num_experts=32, top_k=4, expert_dim=768
        ),
    }
    try:
        selected = dimensions[preset]
    except KeyError as exc:
        raise ValueError(f"unknown model preset: {preset}") from exc

    return ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        local_window_size=min(1024, max_seq_len),
        **selected,
    )
