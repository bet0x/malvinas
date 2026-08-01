import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import torch

from malvinas.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    save_model,
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
from malvinas.train import (
    WarmupCosineScheduler,
    accumulate_expert_counts,
    backward_loss,
    build_optimizer,
    compute_training_loss,
    optimizer_step,
    rescale_gradients,
    update_expert_bias,
)

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
    parser.add_argument(
        "--tokens-per-update",
        type=int,
        help="Global token batch. Must be divisible by batch-size * block-size.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--lr-decay-steps", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--embedding-weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm; use 0 to disable clipping.",
    )
    parser.add_argument(
        "--model-name",
        help="Output model name (default: malvinas-{preset}, with -sft for SFT).",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Override the default models/{model_name}/checkpoints directory.",
    )
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
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
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
    if args.tokens_per_update is not None:
        micro_batch_tokens = args.batch_size * args.block_size
        if args.tokens_per_update < micro_batch_tokens:
            raise ValueError("--tokens-per-update cannot be smaller than one micro-batch")
        if args.tokens_per_update % micro_batch_tokens:
            raise ValueError(
                "--tokens-per-update must be divisible by batch-size * block-size"
            )
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.lr_decay_steps is not None and args.lr_decay_steps <= 0:
        raise ValueError("--lr-decay-steps must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be between zero and one")
    if args.weight_decay < 0 or args.embedding_weight_decay < 0:
        raise ValueError("weight decay values must be non-negative")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("--beta1 and --beta2 must be in [0, 1)")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be non-negative")
    if args.model_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.model_name):
        raise ValueError(
            "--model-name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or hyphens"
        )


def _default_model_name(args: argparse.Namespace) -> str:
    suffix = "-sft" if args.mode == "sft" else ""
    return f"malvinas-{args.preset}{suffix}"


def _stage_model_name(source_model_name: str, mode: str) -> str:
    suffix = f"-{mode}"
    if source_model_name.endswith(suffix):
        return source_model_name
    return f"{source_model_name}{suffix}"


def _output_paths(args: argparse.Namespace, model_name: str) -> tuple[Path, Path]:
    model_dir = args.models_dir / model_name
    checkpoint_dir = args.checkpoint_dir or model_dir / "checkpoints"
    return model_dir, checkpoint_dir


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
    if value == "float16":
        if device.type != "cuda":
            raise ValueError("float16 training requires CUDA")
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def _infer_separator_id(tokenizer: Tokenizer) -> int:
    for token in ("<|endoftext|>", "<|end_of_text|>"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return token_id
    raise ValueError("tokenizer has no known end-of-text token; pass --separator-id")


def _new_run_config(
    args: argparse.Namespace, tokenizer: Tokenizer, model_name: str
) -> dict:
    dataset = args.dataset
    dataset_config = args.dataset_config
    if args.mode == "pretrain":
        dataset = dataset or DEFAULT_PRETRAIN_DATASET
    else:
        dataset = dataset or DEFAULT_SFT_DATASET
        if dataset == DEFAULT_SFT_DATASET and dataset_config is None:
            dataset_config = "all"

    return {
        "model_name": model_name,
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
        "training": _training_config(args),
    }


def _training_config(args: argparse.Namespace) -> dict:
    micro_batch_tokens = args.batch_size * args.block_size
    tokens_per_update = args.tokens_per_update or micro_batch_tokens
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else min(100, args.max_steps // 100)
    )
    decay_steps = args.lr_decay_steps or args.max_steps
    if decay_steps <= warmup_steps:
        raise ValueError("LR decay steps must be greater than warmup steps")
    return {
        "tokens_per_update": tokens_per_update,
        "learning_rate": args.learning_rate,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_steps": warmup_steps,
        "lr_decay_steps": decay_steps,
        "weight_decay": args.weight_decay,
        "embedding_weight_decay": args.embedding_weight_decay,
        "betas": (args.beta1, args.beta2),
        "max_grad_norm": args.max_grad_norm,
        "precision": args.precision,
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
    torch.manual_seed(args.seed)

    model_name = args.model_name or _default_model_name(args)
    _, checkpoint_dir = _output_paths(args, model_name)
    resume_payload = None
    if args.resume:
        resume_path = (
            latest_checkpoint(checkpoint_dir, args.mode)
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
    source_model_name = (
        source_payload["run_config"].get("model_name") if source_payload else None
    )
    if (
        args.model_name
        and resume_payload
        and source_model_name
        and args.model_name != source_model_name
    ):
        raise ValueError(
            f"--model-name must remain {source_model_name!r} when resuming this run"
        )
    if not args.model_name and resume_payload and source_model_name:
        model_name = source_model_name
    elif not args.model_name and init_payload and source_model_name:
        model_name = _stage_model_name(source_model_name, args.mode)
    model_dir, checkpoint_dir = _output_paths(args, model_name)
    tokenizer_repo = (
        resume_payload["run_config"]["tokenizer"] if resume_payload else args.tokenizer
    )
    tokenizer = Tokenizer(tokenizer_repo)

    if resume_payload:
        run_config = dict(resume_payload["run_config"])
        run_config.setdefault("model_name", model_name)
    else:
        run_config = _new_run_config(args, tokenizer, model_name)
    training_config = run_config.get("training")
    if training_config is None:
        training_config = _training_config(args)
        run_config["training"] = training_config
    autocast_dtype = _resolve_autocast_dtype(training_config["precision"], device)

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
    optimizer = build_optimizer(
        model,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        embedding_weight_decay=training_config["embedding_weight_decay"],
        betas=tuple(training_config["betas"]),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_lr=training_config["learning_rate"],
        min_lr=training_config["learning_rate"] * training_config["min_lr_ratio"],
        warmup_steps=training_config["warmup_steps"],
        decay_steps=training_config["lr_decay_steps"],
    )
    grad_scaler = torch.amp.GradScaler(
        device.type, enabled=autocast_dtype == torch.float16
    )
    step = 0
    blocks_consumed = 0
    if resume_payload:
        restore_checkpoint(
            resume_payload,
            model,
            optimizer,
            device,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
        )
        step = resume_payload["step"]
        blocks_consumed = resume_payload["blocks_consumed"]
        if "scheduler_state_dict" not in resume_payload:
            scheduler.set_step(step)
    elif init_payload:
        model.load_state_dict(init_payload["model_state_dict"])

    blocks = _block_stream(run_config, tokenizer)
    _skip_blocks(blocks, blocks_consumed)
    micro_batch_tokens = run_config["batch_size"] * run_config["block_size"]
    tokens_per_update = training_config["tokens_per_update"]
    if tokens_per_update % micro_batch_tokens:
        raise ValueError(
            "checkpoint tokens_per_update is not divisible by its micro-batch size"
        )
    accumulation_steps = tokens_per_update // micro_batch_tokens
    last_checkpoint = None
    last_saved_step = -1

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_loss_tokens = 0
        expert_counts = None
        micro_steps = 0
        update_blocks = 0
        for _ in range(accumulation_steps):
            result = _next_batch(blocks, run_config["batch_size"])
            if result is None:
                break
            (input_ids, target_ids, loss_mask), consumed = result
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)

            loss = compute_training_loss(
                model,
                input_ids,
                target_ids,
                loss_mask,
                autocast_dtype=autocast_dtype,
            )
            loss_tokens = (
                int(loss_mask.sum().item())
                if loss_mask is not None
                else target_ids.numel()
            )
            if loss_tokens:
                backward_loss(
                    loss,
                    divisor=tokens_per_update / loss_tokens,
                    grad_scaler=grad_scaler,
                )
            expert_counts = accumulate_expert_counts(model, expert_counts)
            accumulated_loss += loss.item() * loss_tokens
            accumulated_loss_tokens += loss_tokens
            micro_steps += 1
            update_blocks += consumed
            blocks_consumed += consumed

        if micro_steps == 0:
            break
        if accumulated_loss_tokens == 0:
            continue
        if accumulated_loss_tokens != tokens_per_update:
            rescale_gradients(
                model.parameters(),
                tokens_per_update / accumulated_loss_tokens,
            )
        grad_norm = optimizer_step(
            model,
            optimizer,
            max_grad_norm=training_config["max_grad_norm"] or None,
            grad_scaler=grad_scaler,
        )
        update_expert_bias(model, expert_counts)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        step += 1
        mean_loss = accumulated_loss / accumulated_loss_tokens
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step} loss={mean_loss:.6f} lr={learning_rate:.3e} "
                f"grad_norm={grad_norm:.4f} "
                f"tokens={update_blocks * run_config['block_size']} "
                f"blocks={blocks_consumed}",
                flush=True,
            )
        if step % args.save_every == 0:
            last_checkpoint = save_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                step=step,
                blocks_consumed=blocks_consumed,
                mode=args.mode,
                model_config=model_config.to_dict(),
                run_config=run_config,
                scheduler=scheduler,
                grad_scaler=grad_scaler,
            )
            last_saved_step = step
            print(f"checkpoint={last_checkpoint}", flush=True)

    if step == 0:
        raise RuntimeError("dataset produced no complete training blocks")
    if step != last_saved_step:
        last_checkpoint = save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            step=step,
            blocks_consumed=blocks_consumed,
            mode=args.mode,
            model_config=model_config.to_dict(),
            run_config=run_config,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
        )
        print(f"checkpoint={last_checkpoint}", flush=True)
    model_path = save_model(
        model_dir,
        model,
        step=step,
        mode=args.mode,
        model_name=model_name,
        model_config=model_config.to_dict(),
        run_config=run_config,
    )
    print(f"model={model_path}", flush=True)
    return last_checkpoint


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
