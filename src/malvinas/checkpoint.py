import os
from pathlib import Path

import torch
from torch import nn

CHECKPOINT_VERSION = 1


def checkpoint_filename(mode: str, step: int) -> str:
    return f"{mode}-step-{step:08d}.pt"


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
) -> Path:
    """Atomically save everything required to resume a training run."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / checkpoint_filename(mode, step)
    temporary = destination.with_suffix(".pt.tmp")
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
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


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
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload['version']} "
            f"(expected {CHECKPOINT_VERSION})"
        )
    return payload


def restore_checkpoint(
    payload: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
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
