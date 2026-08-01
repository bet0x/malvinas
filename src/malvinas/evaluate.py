import argparse
import json
from pathlib import Path

import torch

from malvinas.cli import (
    DEFAULT_PRETRAIN_DATASET,
    _block_stream,
    _evaluate,
    _infer_separator_id,
    _resolve_autocast_dtype,
    _resolve_device,
)
from malvinas.config import ModelConfig
from malvinas.tokenizer import DEFAULT_REPO_ID, Tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malvinas-evaluate",
        description="Evaluate a saved Malvinas model on a streaming dataset.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--mode", choices=("pretrain", "sft"))
    parser.add_argument("--sft-format", choices=("messages", "xlam"), default="messages")
    parser.add_argument("--tokenizer")
    parser.add_argument("--separator-id", type=int)
    parser.add_argument("--min-quality-score", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.batch_size <= 0 or args.max_batches <= 0:
        raise ValueError("--batch-size and --max-batches must be positive")

    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    required = {"model_config", "model_state_dict"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"invalid model artifact, missing: {', '.join(sorted(missing))}")

    model_config = ModelConfig.from_dict(payload["model_config"])
    saved_run_config = payload.get("run_config", {})
    mode = args.mode or payload.get("mode", saved_run_config.get("mode", "pretrain"))
    tokenizer_repo = args.tokenizer or saved_run_config.get("tokenizer", DEFAULT_REPO_ID)
    tokenizer = Tokenizer(tokenizer_repo)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the model vocabulary")

    block_size = args.block_size or min(
        saved_run_config.get("block_size", model_config.max_seq_len),
        model_config.max_seq_len,
    )
    if block_size <= 0 or block_size > model_config.max_seq_len:
        raise ValueError("--block-size must fit within the model context length")
    separator_id = (
        args.separator_id
        if args.separator_id is not None
        else saved_run_config.get("separator_id")
    )
    if separator_id is None:
        separator_id = _infer_separator_id(tokenizer)

    dataset = args.dataset
    dataset_config = args.dataset_config
    if dataset is None:
        dataset = saved_run_config.get("dataset")
    if dataset is None and mode == "pretrain":
        dataset = DEFAULT_PRETRAIN_DATASET
    if dataset is None:
        raise ValueError("--dataset is required when the artifact has no saved dataset")
    if dataset_config is None:
        dataset_config = saved_run_config.get("dataset_config")

    run_config = {
        "mode": mode,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "sft_format": args.sft_format,
        "separator_id": separator_id,
        "min_quality_score": args.min_quality_score,
        "max_examples": args.max_examples,
        "block_size": block_size,
        "batch_size": args.batch_size,
        "split": args.split,
    }
    device = _resolve_device(args.device)
    autocast_dtype = _resolve_autocast_dtype(args.precision, device)
    model = model_config.build()
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)
    metrics = _evaluate(
        model,
        _block_stream(run_config, tokenizer),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    print(
        json.dumps(
            {
                "model": str(args.model),
                "dataset": dataset,
                "split": args.split,
                "mode": mode,
                "block_size": block_size,
                **metrics,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
