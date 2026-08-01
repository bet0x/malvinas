import argparse
from collections.abc import Iterator
from pathlib import Path

import torch

from malvinas.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from malvinas.config import PRESET_NAMES, ModelConfig, model_config_from_preset
from malvinas.data import (
    pack_sft_tokens,
    pack_tokens,
    stream_pretrain_documents,
    stream_sft_examples,
    xlam_to_messages,
)
from malvinas.tokenizer import DEFAULT_REPO_ID, Tokenizer
from malvinas.train import train_step

DEFAULT_PRETRAIN_DATASET = "allenai/dolma3_mix-150B-1025"
DEFAULT_SFT_DATASET = "HuggingFaceTB/smoltalk"

Block = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malvinas-train",
        description="Run resumable Malvinas pretraining or supervised fine-tuning.",
    )
    parser.add_argument("--mode", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--preset", choices=PRESET_NAMES, default="tiny")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-config")
    parser.add_argument("--sft-format", choices=("messages", "xlam"), default="messages")
    parser.add_argument("--tokenizer", default=DEFAULT_REPO_ID)
    parser.add_argument("--separator-id", type=int)
    parser.add_argument("--min-quality-score", type=float, default=0.5)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--resume",
        help="Resume a run checkpoint, or use 'latest' for the newest checkpoint of this mode.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Start a new stage with model weights from a checkpoint (for example pretrain to SFT).",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    for name in ("block_size", "batch_size", "max_steps", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_autocast_dtype(value: str, device: torch.device) -> torch.dtype | None:
    if value == "float32":
        return None
    if value == "bfloat16":
        if device.type != "cuda":
            raise ValueError("bfloat16 training currently requires CUDA")
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def _infer_separator_id(tokenizer: Tokenizer) -> int:
    for token in ("<|endoftext|>", "<|end_of_text|>"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return token_id
    raise ValueError("tokenizer has no known end-of-text token; pass --separator-id")


def _new_run_config(args: argparse.Namespace, tokenizer: Tokenizer) -> dict:
    dataset = args.dataset
    dataset_config = args.dataset_config
    if args.mode == "pretrain":
        dataset = dataset or DEFAULT_PRETRAIN_DATASET
    else:
        dataset = dataset or DEFAULT_SFT_DATASET
        if dataset == DEFAULT_SFT_DATASET and dataset_config is None:
            dataset_config = "all"

    return {
        "mode": args.mode,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "sft_format": args.sft_format,
        "tokenizer": args.tokenizer,
        "separator_id": args.separator_id
        if args.separator_id is not None
        else _infer_separator_id(tokenizer),
        "min_quality_score": args.min_quality_score,
        "max_examples": args.max_examples,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
    }


def _block_stream(run_config: dict, tokenizer: Tokenizer) -> Iterator[Block]:
    if run_config["mode"] == "pretrain":
        texts = stream_pretrain_documents(
            run_config["dataset"],
            min_quality_score=run_config["min_quality_score"],
            max_documents=run_config["max_examples"],
        )
        documents = (tokenizer.encode(text) for text in texts)
        for input_ids, target_ids in pack_tokens(
            documents, run_config["block_size"], run_config["separator_id"]
        ):
            yield input_ids, target_ids, None
        return

    row_to_messages = xlam_to_messages if run_config["sft_format"] == "xlam" else None
    kwargs = {}
    if row_to_messages is not None:
        kwargs["row_to_messages"] = row_to_messages
    examples = stream_sft_examples(
        run_config["dataset"],
        tokenizer,
        config_name=run_config["dataset_config"],
        max_examples=run_config["max_examples"],
        **kwargs,
    )
    for input_ids, target_ids, loss_mask in pack_sft_tokens(
        examples, run_config["block_size"], run_config["separator_id"]
    ):
        if torch.any(loss_mask):
            yield input_ids, target_ids, loss_mask


def _next_batch(blocks: Iterator[Block], batch_size: int) -> tuple[Block, int] | None:
    items = []
    for _ in range(batch_size):
        try:
            items.append(next(blocks))
        except StopIteration:
            break
    if not items:
        return None

    input_ids = torch.stack([item[0] for item in items])
    target_ids = torch.stack([item[1] for item in items])
    masks = [item[2] for item in items]
    loss_mask = None if masks[0] is None else torch.stack(masks)  # mode is uniform
    return (input_ids, target_ids, loss_mask), len(items)


def _skip_blocks(blocks: Iterator[Block], count: int) -> None:
    for skipped in range(count):
        try:
            next(blocks)
        except StopIteration as exc:
            raise RuntimeError(
                f"dataset ended while restoring position {count}; stopped at {skipped}"
            ) from exc


def run_training(args: argparse.Namespace) -> Path:
    _validate_args(args)
    device = _resolve_device(args.device)
    autocast_dtype = _resolve_autocast_dtype(args.precision, device)
    torch.manual_seed(args.seed)

    resume_payload = None
    if args.resume:
        resume_path = (
            latest_checkpoint(args.checkpoint_dir, args.mode)
            if args.resume == "latest"
            else Path(args.resume)
        )
        resume_payload = load_checkpoint(resume_path)
        if resume_payload["mode"] != args.mode:
            raise ValueError(
                f"checkpoint mode is {resume_payload['mode']}, not requested {args.mode}"
            )

    init_payload = load_checkpoint(args.init_from) if args.init_from else None
    source_payload = resume_payload or init_payload
    tokenizer_repo = (
        resume_payload["run_config"]["tokenizer"] if resume_payload else args.tokenizer
    )
    tokenizer = Tokenizer(tokenizer_repo)

    if resume_payload:
        run_config = resume_payload["run_config"]
    else:
        run_config = _new_run_config(args, tokenizer)

    if source_payload:
        model_config = ModelConfig.from_dict(source_payload["model_config"])
        if tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError(
                "tokenizer vocabulary does not match the checkpoint model vocabulary"
            )
        if run_config["block_size"] > model_config.max_seq_len:
            raise ValueError("block size exceeds the checkpoint model's maximum sequence length")
    else:
        model_config = model_config_from_preset(
            args.preset, tokenizer.vocab_size, run_config["block_size"]
        )

    model = model_config.build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    step = 0
    blocks_consumed = 0
    if resume_payload:
        restore_checkpoint(resume_payload, model, optimizer, device)
        step = resume_payload["step"]
        blocks_consumed = resume_payload["blocks_consumed"]
    elif init_payload:
        model.load_state_dict(init_payload["model_state_dict"])

    blocks = _block_stream(run_config, tokenizer)
    _skip_blocks(blocks, blocks_consumed)
    last_checkpoint = None
    last_saved_step = -1

    while step < args.max_steps:
        result = _next_batch(blocks, run_config["batch_size"])
        if result is None:
            break
        (input_ids, target_ids, loss_mask), consumed = result
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)

        loss = train_step(
            model,
            optimizer,
            input_ids,
            target_ids,
            loss_mask,
            autocast_dtype=autocast_dtype,
        )
        step += 1
        blocks_consumed += consumed
        if step == 1 or step % args.log_every == 0:
            print(f"step={step} loss={loss:.6f} blocks={blocks_consumed}", flush=True)
        if step % args.save_every == 0:
            last_checkpoint = save_checkpoint(
                args.checkpoint_dir,
                model,
                optimizer,
                step=step,
                blocks_consumed=blocks_consumed,
                mode=args.mode,
                model_config=model_config.to_dict(),
                run_config=run_config,
            )
            last_saved_step = step
            print(f"checkpoint={last_checkpoint}", flush=True)

    if step == 0:
        raise RuntimeError("dataset produced no complete training blocks")
    if step != last_saved_step:
        last_checkpoint = save_checkpoint(
            args.checkpoint_dir,
            model,
            optimizer,
            step=step,
            blocks_consumed=blocks_consumed,
            mode=args.mode,
            model_config=model_config.to_dict(),
            run_config=run_config,
        )
        print(f"checkpoint={last_checkpoint}", flush=True)
    return last_checkpoint


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
