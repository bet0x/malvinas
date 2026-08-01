import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from malvinas.moe import MoEFeedForward

EXPERT_BIAS_UPDATE_RATE = 0.01
IGNORE_INDEX = -100


class WarmupCosineScheduler:
    """Warm up linearly, then decay to a minimum LR with a cosine curve."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        decay_steps: int,
    ) -> None:
        if not 0.0 <= min_lr <= max_lr:
            raise ValueError("min_lr must be between zero and max_lr")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if decay_steps <= warmup_steps:
            raise ValueError("decay_steps must be greater than warmup_steps")
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.step_num = 0
        self._apply_lr()

    def learning_rate(self, step: int | None = None) -> float:
        step = self.step_num if step is None else step
        if self.warmup_steps and step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        if step >= self.decay_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (
            self.decay_steps - self.warmup_steps
        )
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coefficient * (self.max_lr - self.min_lr)

    def _apply_lr(self) -> None:
        learning_rate = self.learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self) -> None:
        self.set_step(self.step_num + 1)

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("scheduler step must be non-negative")
        self.step_num = step
        self._apply_lr()

    def state_dict(self) -> dict:
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.max_lr = state_dict["max_lr"]
        self.min_lr = state_dict["min_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.decay_steps = state_dict["decay_steps"]
        self.step_num = state_dict["step_num"]
        self._apply_lr()


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    embedding_weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Create AdamW groups for matrices, embeddings, and scalar parameters."""
    embedding_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    groups: dict[str, list[nn.Parameter]] = {
        "matrix": [],
        "embedding": [],
        "scalar": [],
    }
    seen: set[int] = set()
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if not parameter.requires_grad or parameter_id in seen:
            continue
        seen.add(parameter_id)
        if parameter_id in embedding_parameters:
            groups["embedding"].append(parameter)
        elif parameter.ndim >= 2:
            groups["matrix"].append(parameter)
        else:
            groups["scalar"].append(parameter)

    parameter_groups = [
        {
            "params": groups["matrix"],
            "weight_decay": weight_decay,
            "group_name": "matrix",
        },
        {
            "params": groups["embedding"],
            "weight_decay": embedding_weight_decay,
            "group_name": "embedding",
        },
        {
            "params": groups["scalar"],
            "weight_decay": 0.0,
            "group_name": "scalar",
        },
    ]
    return torch.optim.AdamW(
        [group for group in parameter_groups if group["params"]],
        lr=learning_rate,
        betas=betas,
    )


def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute next-token cross-entropy, optionally excluding masked tokens."""
    B, T, V = logits.shape
    targets = target_ids
    if loss_mask is not None:
        targets = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    flat_logits = logits.reshape(B * T, V)
    flat_targets = targets.reshape(B * T)
    if not torch.any(flat_targets != IGNORE_INDEX):
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_INDEX)


def compute_training_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    mtp_target_ids: torch.Tensor | None = None,
    mtp_weight: float = 0.1,
    autocast_dtype: torch.dtype | None = None,
    position_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    device_type = input_ids.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        sequence_kwargs = {}
        if position_ids is not None:
            sequence_kwargs["position_ids"] = position_ids
        if document_ids is not None:
            sequence_kwargs["document_ids"] = document_ids
        if mtp_target_ids is not None:
            logits, mtp_logits = model.forward_with_mtp(
                input_ids,
                target_ids,
                **sequence_kwargs,
            )
            loss = compute_loss(logits, target_ids, loss_mask) + mtp_weight * compute_loss(
                mtp_logits, mtp_target_ids, loss_mask
            )
        else:
            logits = model(input_ids, **sequence_kwargs)
            loss = compute_loss(logits, target_ids, loss_mask)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss: {loss.item()}")
    return loss


def backward_loss(
    loss: torch.Tensor,
    *,
    divisor: float = 1.0,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> None:
    if divisor <= 0:
        raise ValueError("loss divisor must be positive")
    scaled_loss = loss / divisor
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()


def rescale_gradients(parameters: Iterable[nn.Parameter], scale: float) -> None:
    if scale == 1.0:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> float:
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.unscale_(optimizer)

    clip_limit = max_grad_norm if max_grad_norm is not None else float("inf")
    try:
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), clip_limit, error_if_nonfinite=True
        )
    except RuntimeError as exc:
        raise FloatingPointError("non-finite gradients detected") from exc
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite gradient norm: {grad_norm.item()}")

    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    return float(grad_norm)


@torch.no_grad()
def accumulate_expert_counts(
    model: nn.Module,
    totals: dict[MoEFeedForward, torch.Tensor] | None = None,
) -> dict[MoEFeedForward, torch.Tensor]:
    totals = {} if totals is None else totals
    for module in model.modules():
        if (
            not isinstance(module, MoEFeedForward)
            or module.last_selected_experts is None
        ):
            continue
        counts = torch.bincount(
            module.last_selected_experts,
            minlength=module.num_experts,
        ).float()
        if module in totals:
            totals[module].add_(counts)
        else:
            totals[module] = counts
    return totals


@torch.no_grad()
def update_expert_bias(
    model: nn.Module,
    expert_counts: dict[MoEFeedForward, torch.Tensor] | None = None,
) -> None:
    for module in model.modules():
        if isinstance(module, MoEFeedForward):
            counts = None if expert_counts is None else expert_counts.get(module)
            if expert_counts is not None and counts is None:
                continue
            module.update_expert_bias(EXPERT_BIAS_UPDATE_RATE, counts=counts)


def expert_load_statistics(
    expert_counts: dict[MoEFeedForward, torch.Tensor] | None,
) -> dict[str, float]:
    """Summarize router balance across all experts used in an update."""
    if not expert_counts:
        return {}
    counts = torch.cat([value.detach().float().cpu() for value in expert_counts.values()])
    mean = counts.mean()
    if mean == 0:
        return {}
    return {
        "expert_load_cv": float(counts.std(unbiased=False) / mean),
        "expert_load_min": float(counts.min()),
        "expert_load_max": float(counts.max()),
    }


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    mtp_target_ids: torch.Tensor | None = None,
    mtp_weight: float = 0.1,
    autocast_dtype: torch.dtype | None = None,
    max_grad_norm: float | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one complete optimizer update for callers that do not accumulate."""
    optimizer.zero_grad(set_to_none=True)
    loss = compute_training_loss(
        model,
        input_ids,
        target_ids,
        loss_mask,
        mtp_target_ids,
        mtp_weight,
        autocast_dtype,
    )
    backward_loss(loss, grad_scaler=grad_scaler)
    optimizer_step(
        model,
        optimizer,
        max_grad_norm=max_grad_norm,
        grad_scaler=grad_scaler,
    )
    update_expert_bias(model)
    return loss.item()
