import argparse
import itertools
import json
import math
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from malvinas.checkpoint import (
    checkpoint_filename,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_checkpoint,
    save_checkpoint,
    save_model,
)
from malvinas.config import PRESET_NAMES, ModelConfig, model_config_from_preset
from malvinas.data import (
    pack_document_tokens,
    pack_sft_document_tokens,
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
    expert_load_statistics,
    optimizer_step,
    rescale_gradients,
    update_expert_bias,
)

DEFAULT_PRETRAIN_DATASET = "allenai/dolma3_mix-150B-1025"
DEFAULT_SFT_DATASET = "HuggingFaceTB/smoltalk"

Block = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


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
        "--moe-kernel",
        choices=("auto", "eager_mm", "grouped_mm", "grouped_mm_fast"),
        default="auto",
    )
    parser.add_argument(
        "--document-attention-backend",
        choices=("auto", "flex", "sdpa"),
        default="auto",
        help="Packed-document global attention backend; auto selects FlexAttention on CUDA.",
    )
    parser.add_argument(
        "--tokens-per-update",
        type=int,
        help=(
            "Global token batch. Must be divisible by batch-size * block-size "
            "* number of distributed workers."
        ),
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
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Periodic checkpoints to retain; use 0 to keep all.",
    )
    parser.add_argument(
        "--milestone-every",
        type=int,
        default=0,
        help="Also preserve a named checkpoint every N steps; use 0 to disable.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validation-dataset")
    parser.add_argument("--validation-dataset-config")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefetch the next batch on a separate CUDA stream.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Capture a PyTorch trace for the first N optimizer steps.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_args(args: argparse.Namespace, world_size: int = 1) -> None:
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    for name in ("block_size", "batch_size", "max_steps", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints must be non-negative")
    if args.milestone_every < 0:
        raise ValueError("--milestone-every must be non-negative")
    if args.validate_every < 0 or args.validation_steps <= 0:
        raise ValueError("validation intervals must be non-negative and steps positive")
    if args.validation_dataset and args.validate_every == 0:
        raise ValueError("--validation-dataset requires --validate-every")
    if args.validate_every and not args.validation_dataset:
        raise ValueError("--validate-every requires --validation-dataset")
    if args.profile_steps < 0:
        raise ValueError("--profile-steps must be non-negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.tokens_per_update is not None:
        micro_batch_tokens = args.batch_size * args.block_size * world_size
        if args.tokens_per_update < micro_batch_tokens:
            raise ValueError("--tokens-per-update cannot be smaller than one micro-batch")
        if args.tokens_per_update % micro_batch_tokens:
            raise ValueError(
                "--tokens-per-update must be divisible by batch-size * block-size "
                "* distributed world size"
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


def _initialize_distributed(device_value: str) -> tuple[DistributedContext, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(), _resolve_device(device_value)
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA and one process per GPU")
    if device_value not in ("auto", "cuda"):
        raise ValueError("under torchrun, --device must be 'auto' or 'cuda'")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return (
        DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size),
        torch.device("cuda", local_rank),
    )


def _distributed_barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()


def _all_ranks_have_batch(
    result: tuple[Block, int] | None,
    context: DistributedContext,
    device: torch.device,
) -> bool:
    if not context.enabled:
        return result is not None
    local_count = 0 if result is None else result[1]
    minimum = torch.tensor(local_count, device=device, dtype=torch.int32)
    maximum = minimum.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    return bool(minimum.item()) and minimum.item() == maximum.item()


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
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    model_name: str,
    world_size: int = 1,
) -> dict:
    dataset = args.dataset
    dataset_config = args.dataset_config
    if args.mode == "pretrain":
        dataset = dataset or DEFAULT_PRETRAIN_DATASET
    else:
        dataset = dataset or DEFAULT_SFT_DATASET
        if dataset == DEFAULT_SFT_DATASET and dataset_config is None:
            dataset_config = "all"

    config = {
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
        "distributed_world_size": world_size,
        "training": _training_config(args, world_size),
    }
    if args.validation_dataset:
        config["validation"] = {
            "dataset": args.validation_dataset,
            "dataset_config": args.validation_dataset_config,
            "split": args.validation_split,
            "every": args.validate_every,
            "steps": args.validation_steps,
        }
    return config


def _training_config(args: argparse.Namespace, world_size: int = 1) -> dict:
    micro_batch_tokens = args.batch_size * args.block_size * world_size
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
            split=run_config.get("split", "train"),
            config_name=run_config["dataset_config"],
        )
        documents = (tokenizer.encode(text) for text in texts)
        for block in pack_document_tokens(
            documents, run_config["block_size"], run_config["separator_id"]
        ):
            yield (
                block.input_ids,
                block.target_ids,
                block.loss_mask,
                block.position_ids,
                block.document_ids,
            )
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
        split=run_config.get("split", "train"),
        **kwargs,
    )
    for block in pack_sft_document_tokens(
        examples, run_config["block_size"], run_config["separator_id"]
    ):
        if torch.any(block.loss_mask):
            yield (
                block.input_ids,
                block.target_ids,
                block.loss_mask,
                block.position_ids,
                block.document_ids,
            )


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
    positions = [
        item[3] if len(item) == 5 else torch.arange(item[0].numel())
        for item in items
    ]
    documents = [
        item[4] if len(item) == 5 else torch.zeros_like(item[0])
        for item in items
    ]
    return (
        input_ids,
        target_ids,
        loss_mask,
        torch.stack(positions),
        torch.stack(documents),
    ), len(items)


def _move_batch(batch: Block, device: torch.device) -> Block:
    non_blocking = device.type == "cuda"

    def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if non_blocking and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=non_blocking)

    return tuple(move(tensor) for tensor in batch)  # type: ignore[return-value]


def _validation_run_config(run_config: dict) -> dict:
    validation = run_config["validation"]
    config = dict(run_config)
    config.update(
        dataset=validation["dataset"],
        dataset_config=validation["dataset_config"],
        split=validation["split"],
        max_examples=None,
    )
    return config


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    blocks: Iterator[Block],
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    try:
        for _ in range(max_batches):
            result = _next_batch(blocks, batch_size)
            if result is None:
                break
            batch, _ = result
            input_ids, target_ids, loss_mask, position_ids, document_ids = _move_batch(
                batch, device
            )
            loss = compute_training_loss(
                model,
                input_ids,
                target_ids,
                loss_mask,
                autocast_dtype=autocast_dtype,
                position_ids=position_ids,
                document_ids=document_ids,
            )
            loss_tokens = int(loss_mask.sum().item()) if loss_mask is not None else target_ids.numel()
            total_loss += loss.item() * loss_tokens
            total_tokens += loss_tokens
            batches += 1
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise RuntimeError("validation dataset produced no loss-bearing blocks")
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "tokens": total_tokens,
        "batches": batches,
    }


def _append_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


class _BatchPrefetcher:
    def __init__(
        self,
        blocks: Iterator[Block],
        batch_size: int,
        device: torch.device,
        enabled: bool,
    ) -> None:
        self.blocks = blocks
        self.batch_size = batch_size
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device)
            if enabled and device.type == "cuda"
            else None
        )
        self.pending: tuple[Block, int] | None = None
        if self.stream is not None:
            self._preload()

    def _preload(self) -> None:
        result = _next_batch(self.blocks, self.batch_size)
        if result is None:
            self.pending = None
            return
        batch, consumed = result
        with torch.cuda.stream(self.stream):
            self.pending = _move_batch(batch, self.device), consumed

    def next(self) -> tuple[Block, int] | None:
        if self.stream is None:
            result = _next_batch(self.blocks, self.batch_size)
            if result is None:
                return None
            batch, consumed = result
            return _move_batch(batch, self.device), consumed

        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        result = self.pending
        if result is None:
            return None
        batch, consumed = result
        for tensor in batch:
            if tensor is not None:
                tensor.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch, consumed


def _skip_blocks(blocks: Iterator[Block], count: int) -> None:
    for skipped in range(count):
        try:
            next(blocks)
        except StopIteration as exc:
            raise RuntimeError(
                f"dataset ended while restoring position {count}; stopped at {skipped}"
            ) from exc


def run_training(args: argparse.Namespace) -> Path:
    distributed, device = _initialize_distributed(args.device)
    _validate_args(args, distributed.world_size)
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
        if args.validation_dataset:
            run_config["validation"] = {
                "dataset": args.validation_dataset,
                "dataset_config": args.validation_dataset_config,
                "split": args.validation_split,
                "every": args.validate_every,
                "steps": args.validation_steps,
            }
    else:
        run_config = _new_run_config(
            args, tokenizer, model_name, distributed.world_size
        )
    saved_world_size = run_config.get("distributed_world_size", 1)
    if saved_world_size != distributed.world_size:
        raise ValueError(
            "a run must be resumed with the same distributed world size "
            f"(checkpoint={saved_world_size}, current={distributed.world_size})"
        )
    training_config = run_config.get("training")
    if training_config is None:
        training_config = _training_config(args, distributed.world_size)
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
        model_config = replace(
            model_config,
            moe_kernel=args.moe_kernel,
            document_attention_backend=args.document_attention_backend,
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

    training_model = torch.compile(model) if args.compile_model else model
    if distributed.enabled:
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            find_unused_parameters=model.mtp_head is not None,
        )

    blocks = _block_stream(run_config, tokenizer)
    if distributed.enabled:
        blocks = itertools.islice(
            blocks, distributed.rank, None, distributed.world_size
        )
    _skip_blocks(blocks, blocks_consumed)
    batch_loader = _BatchPrefetcher(
        blocks,
        run_config["batch_size"],
        device,
        args.prefetch,
    )
    local_micro_batch_tokens = run_config["batch_size"] * run_config["block_size"]
    global_micro_batch_tokens = local_micro_batch_tokens * distributed.world_size
    tokens_per_update = training_config["tokens_per_update"]
    if tokens_per_update % global_micro_batch_tokens:
        raise ValueError(
            "checkpoint tokens_per_update is not divisible by its global micro-batch size"
        )
    accumulation_steps = tokens_per_update // global_micro_batch_tokens
    local_tokens_per_update = tokens_per_update // distributed.world_size
    last_checkpoint = None
    last_saved_step = -1
    metrics_path = model_dir / "metrics.jsonl"
    best_validation_loss = float(
        (resume_payload or {}).get("extra_state", {}).get(
            "best_validation_loss", float("inf")
        )
    )
    profile = None
    profile_start_step = step
    run_start_step = step
    run_started = time.perf_counter()
    profile_path = model_dir / "profile.json"
    if args.profile_steps and distributed.is_main:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profile.start()

    while step < args.max_steps:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_loss_tokens = 0
        expert_counts = None
        micro_steps = 0
        update_blocks = 0
        data_wait_seconds = 0.0
        forward_backward_seconds = 0.0
        compute_events = []
        for _ in range(accumulation_steps):
            data_wait_started = time.perf_counter()
            result = batch_loader.next()
            data_wait_seconds += time.perf_counter() - data_wait_started
            if not _all_ranks_have_batch(result, distributed, device):
                break
            assert result is not None
            (
                input_ids,
                target_ids,
                loss_mask,
                position_ids,
                document_ids,
            ), consumed = result

            compute_started = time.perf_counter()
            compute_start_event = compute_end_event = None
            if device.type == "cuda":
                compute_start_event = torch.cuda.Event(enable_timing=True)
                compute_end_event = torch.cuda.Event(enable_timing=True)
                compute_start_event.record()
            loss = compute_training_loss(
                training_model,
                input_ids,
                target_ids,
                loss_mask,
                autocast_dtype=autocast_dtype,
                position_ids=position_ids,
                document_ids=document_ids,
            )
            loss_tokens = (
                int(loss_mask.sum().item())
                if loss_mask is not None
                else target_ids.numel()
            )
            backward_loss(
                loss,
                divisor=(
                    local_tokens_per_update / loss_tokens
                    if loss_tokens
                    else 1.0
                ),
                grad_scaler=grad_scaler,
            )
            if compute_end_event is not None:
                compute_end_event.record()
                compute_events.append((compute_start_event, compute_end_event))
            else:
                forward_backward_seconds += time.perf_counter() - compute_started
            expert_counts = accumulate_expert_counts(model, expert_counts)
            accumulated_loss += loss.item() * loss_tokens
            accumulated_loss_tokens += loss_tokens
            micro_steps += 1
            update_blocks += consumed
            blocks_consumed += consumed

        if micro_steps == 0:
            break
        aggregate = torch.tensor(
            [accumulated_loss, accumulated_loss_tokens, update_blocks],
            dtype=torch.float64,
            device=device,
        )
        if distributed.enabled:
            torch.distributed.all_reduce(aggregate)
            if expert_counts is not None:
                for counts in expert_counts.values():
                    torch.distributed.all_reduce(counts)
        global_loss_sum, global_loss_tokens, global_update_blocks = aggregate.tolist()
        if global_loss_tokens == 0:
            continue
        if global_loss_tokens != tokens_per_update:
            rescale_gradients(
                model.parameters(),
                tokens_per_update / global_loss_tokens,
            )
        optimizer_started = time.perf_counter()
        optimizer_start_event = optimizer_end_event = None
        if device.type == "cuda":
            optimizer_start_event = torch.cuda.Event(enable_timing=True)
            optimizer_end_event = torch.cuda.Event(enable_timing=True)
            optimizer_start_event.record()
        grad_norm = optimizer_step(
            model,
            optimizer,
            max_grad_norm=training_config["max_grad_norm"] or None,
            grad_scaler=grad_scaler,
        )
        update_expert_bias(model, expert_counts)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if optimizer_end_event is not None:
            optimizer_end_event.record()
            optimizer_seconds = 0.0
        else:
            optimizer_seconds = time.perf_counter() - optimizer_started
        step += 1
        mean_loss = global_loss_sum / global_loss_tokens
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            forward_backward_seconds = sum(
                start.elapsed_time(end) for start, end in compute_events
            ) / 1000.0
            assert optimizer_start_event is not None and optimizer_end_event is not None
            optimizer_seconds = optimizer_start_event.elapsed_time(
                optimizer_end_event
            ) / 1000.0
        update_seconds = time.perf_counter() - update_started
        if distributed.enabled:
            elapsed = torch.tensor(update_seconds, device=device)
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            update_seconds = elapsed.item()
        processed_tokens = int(global_update_blocks) * run_config["block_size"]
        completed_steps = step - run_start_step
        elapsed_run_seconds = time.perf_counter() - run_started
        eta_seconds = (
            (args.max_steps - step) * elapsed_run_seconds / completed_steps
            if completed_steps
            else 0.0
        )
        record = {
            "kind": "train",
            "step": step,
            "loss": mean_loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "tokens": processed_tokens,
            "tokens_per_second": processed_tokens / update_seconds,
            "update_seconds": update_seconds,
            "data_wait_seconds": data_wait_seconds,
            "forward_backward_seconds": forward_backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "eta_seconds": eta_seconds,
            "blocks_consumed": blocks_consumed,
            "world_size": distributed.world_size,
            **expert_load_statistics(expert_counts),
        }
        if device.type == "cuda":
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(device), device=device
            )
            if distributed.enabled:
                torch.distributed.all_reduce(
                    peak_memory, op=torch.distributed.ReduceOp.MAX
                )
            record["cuda_peak_memory_bytes"] = int(peak_memory.item())
        if distributed.is_main:
            _append_metric(metrics_path, record)
        if distributed.is_main and (step == 1 or step % args.log_every == 0):
            print(
                f"step={step} loss={mean_loss:.6f} lr={learning_rate:.3e} "
                f"grad_norm={grad_norm:.4f} "
                f"tokens={processed_tokens} tok/s={record['tokens_per_second']:.0f} "
                f"blocks={blocks_consumed}",
                flush=True,
            )
        validation_config = run_config.get("validation")
        if validation_config and step % validation_config["every"] == 0:
            _distributed_barrier(distributed)
            if distributed.is_main:
                validation_metrics = _evaluate(
                    model,
                    _block_stream(_validation_run_config(run_config), tokenizer),
                    batch_size=run_config["batch_size"],
                    max_batches=validation_config["steps"],
                    device=device,
                    autocast_dtype=autocast_dtype,
                )
                validation_loss = float(validation_metrics["loss"])
                _append_metric(
                    metrics_path,
                    {"kind": "validation", "step": step, **validation_metrics},
                )
                print(f"validation_step={step} loss={validation_loss:.6f}", flush=True)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_path = save_checkpoint(
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
                        filename="best.pt",
                        extra_state={"best_validation_loss": best_validation_loss},
                    )
                    print(f"best_checkpoint={best_path}", flush=True)
            if distributed.enabled:
                best = torch.tensor(
                    best_validation_loss, device=device, dtype=torch.float64
                )
                torch.distributed.broadcast(best, src=0)
                best_validation_loss = best.item()
            _distributed_barrier(distributed)
        if step % args.save_every == 0:
            if distributed.is_main:
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
                    extra_state={"best_validation_loss": best_validation_loss},
                )
                prune_checkpoints(checkpoint_dir, args.mode, args.keep_checkpoints)
                last_saved_step = step
                print(f"checkpoint={last_checkpoint}", flush=True)
            _distributed_barrier(distributed)
        if args.milestone_every and step % args.milestone_every == 0:
            if distributed.is_main:
                milestone_path = save_checkpoint(
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
                    filename=f"{args.mode}-milestone-{step:08d}.pt",
                    extra_state={"best_validation_loss": best_validation_loss},
                )
                print(f"milestone_checkpoint={milestone_path}", flush=True)
            _distributed_barrier(distributed)
        if profile is not None:
            profile.step()
            if step - profile_start_step >= args.profile_steps:
                profile.stop()
                profile.export_chrome_trace(str(profile_path))
                print(f"profile={profile_path}", flush=True)
                profile = None

    if step == 0:
        raise RuntimeError("dataset produced no complete training blocks")
    if profile is not None:
        profile.stop()
        profile.export_chrome_trace(str(profile_path))
        print(f"profile={profile_path}", flush=True)
    if distributed.is_main and step != last_saved_step:
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
            extra_state={"best_validation_loss": best_validation_loss},
        )
        prune_checkpoints(checkpoint_dir, args.mode, args.keep_checkpoints)
        print(f"checkpoint={last_checkpoint}", flush=True)
    _distributed_barrier(distributed)
    if distributed.is_main:
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
    _distributed_barrier(distributed)
    if last_checkpoint is None:
        last_checkpoint = checkpoint_dir / checkpoint_filename(args.mode, step)
    if distributed.enabled:
        torch.distributed.destroy_process_group()
    return last_checkpoint


def main() -> None:
    try:
        run_training(build_parser().parse_args())
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
