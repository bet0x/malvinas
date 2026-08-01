import os
from pathlib import Path

import torch
from torch import nn

CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, CHECKPOINT_VERSION)
MODEL_VERSION = 1


def checkpoint_filename(mode: str, step: int) -> str:
    return f"{mode}-step-{step:08d}.pt"


def _atomic_save(payload: dict, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    blocks_consumed: int,
    mode: str,
    model_config: dict,
    run_config: dict,
    scheduler=None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> Path:
    """Atomically save everything required to resume a training run."""
    destination = Path(checkpoint_dir) / checkpoint_filename(mode, step)
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "blocks_consumed": blocks_consumed,
        "mode": mode,
        "model_config": model_config,
        "run_config": run_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if grad_scaler is not None:
        payload["grad_scaler_state_dict"] = grad_scaler.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return _atomic_save(payload, destination)


def save_model(
    model_dir: str | Path,
    model: nn.Module,
    *,
    step: int,
    mode: str,
    model_name: str,
    model_config: dict,
    run_config: dict,
) -> Path:
    """Save the final inference artifact without optimizer or RNG state."""
    payload = {
        "version": MODEL_VERSION,
        "model_name": model_name,
        "step": step,
        "mode": mode,
        "model_config": model_config,
        "run_config": run_config,
        "model_state_dict": model.state_dict(),
    }
    return _atomic_save(payload, Path(model_dir) / "model.pt")


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "version",
        "step",
        "blocks_consumed",
        "mode",
        "model_config",
        "run_config",
        "model_state_dict",
        "optimizer_state_dict",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"invalid checkpoint, missing: {', '.join(sorted(missing))}")
    if payload["version"] not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint version {payload['version']} "
            f"(expected one of {SUPPORTED_CHECKPOINT_VERSIONS})"
        )
    return payload


def _load_optimizer_state(
    payload: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    saved_state = payload["optimizer_state_dict"]
    try:
        optimizer.load_state_dict(saved_state)
        return
    except ValueError:
        if payload["version"] != 1:
            raise

    old_parameter_ids = [
        parameter_id
        for group in saved_state["param_groups"]
        for parameter_id in group["params"]
    ]
    model_parameters = list(model.parameters())
    if len(old_parameter_ids) != len(model_parameters):
        raise ValueError("cannot migrate the version 1 optimizer parameter layout")

    old_state_by_parameter = {
        id(parameter): saved_state["state"].get(parameter_id, {})
        for parameter, parameter_id in zip(model_parameters, old_parameter_ids)
    }
    migrated = optimizer.state_dict()
    migrated["state"] = {}
    for live_group, serialized_group in zip(
        optimizer.param_groups, migrated["param_groups"]
    ):
        for parameter, parameter_id in zip(
            live_group["params"], serialized_group["params"]
        ):
            state = old_state_by_parameter.get(id(parameter))
            if state:
                migrated["state"][parameter_id] = state
    optimizer.load_state_dict(migrated)


def restore_checkpoint(
    payload: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> None:
    model.load_state_dict(payload["model_state_dict"])
    _load_optimizer_state(payload, model, optimizer)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if grad_scaler is not None and "grad_scaler_state_dict" in payload:
        grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def latest_checkpoint(checkpoint_dir: str | Path, mode: str | None = None) -> Path:
    pattern = f"{mode}-step-*.pt" if mode else "*-step-*.pt"
    candidates = sorted(Path(checkpoint_dir).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    return candidates[-1]
