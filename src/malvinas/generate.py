import argparse
from pathlib import Path

import torch

from malvinas.config import ModelConfig
from malvinas.tokenizer import DEFAULT_REPO_ID, Tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malvinas-generate",
        description="Generate text from a trained Malvinas model using a KV cache.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eos-token-id", "--stop-token-id", dest="stop_token_id", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        k = min(top_k, logits.shape[-1])
        cutoff = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_indices, sorted_logits
        )
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.inference_mode()
def generate_tokens(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    stop_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape [1, time] with a non-empty prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if prompt_ids.shape[1] + max_new_tokens > model.max_seq_len:
        raise ValueError("prompt plus generated tokens exceed the model context length")
    if max_new_tokens == 0:
        return prompt_ids

    was_training = model.training
    model.eval()
    try:
        positions = torch.arange(prompt_ids.shape[1], device=prompt_ids.device).unsqueeze(0)
        logits, cache = model.forward_cached(prompt_ids, positions)
        output = prompt_ids
        for index in range(max_new_tokens):
            next_token = _sample_token(
                logits[:, -1], temperature, top_k, top_p, generator
            )
            output = torch.cat((output, next_token), dim=1)
            if stop_token_id is not None and int(next_token.item()) == stop_token_id:
                break
            if index + 1 == max_new_tokens:
                break
            next_position = torch.full_like(next_token, prompt_ids.shape[1] + index)
            logits, cache = model.forward_cached(next_token, next_position, cache)
        return output
    finally:
        model.train(was_training)


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    device = _resolve_device(args.device)
    precision = args.precision
    if precision == "auto":
        precision = (
            "bfloat16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "float16" if device.type == "cuda" else "float32"
        )
    if device.type == "cpu" and precision != "float32":
        raise ValueError("CPU generation requires --precision float32")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[precision]

    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    required = {"model_config", "model_state_dict"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"invalid model artifact, missing: {', '.join(sorted(missing))}")
    model = ModelConfig.from_dict(payload["model_config"]).build()
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device=device, dtype=dtype)

    tokenizer_repo = payload.get("run_config", {}).get("tokenizer", DEFAULT_REPO_ID)
    tokenizer = Tokenizer(tokenizer_repo)
    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_token_id=args.stop_token_id,
        generator=generator,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
