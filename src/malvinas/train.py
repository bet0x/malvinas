import torch
import torch.nn as nn
from torch.nn import functional as F

from malvinas.moe import MoEFeedForward

EXPERT_BIAS_UPDATE_RATE = 0.01
IGNORE_INDEX = -100


def compute_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Next-token cross-entropy. With `loss_mask` (True = include), positions
    where it's False are excluded entirely (SFT: mask out the prompt/user
    turns, keep only the assistant's own tokens)."""
    B, T, V = logits.shape
    targets = target_ids
    if loss_mask is not None:
        targets = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=IGNORE_INDEX)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    mtp_target_ids: torch.Tensor | None = None,
    mtp_weight: float = 0.1,
) -> float:
    """One training step: forward, loss (optionally SFT-masked and/or with
    an MTP auxiliary term, plan 09), backward, optimizer step, then nudge
    each MoE layer's expert bias (plan 03, no gradient)."""
    if mtp_target_ids is not None:
        logits, mtp_logits = model.forward_with_mtp(input_ids, target_ids)
        loss = compute_loss(logits, target_ids, loss_mask) + mtp_weight * compute_loss(
            mtp_logits, mtp_target_ids, loss_mask
        )
    else:
        logits = model(input_ids)
        loss = compute_loss(logits, target_ids, loss_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for module in model.modules():
        if isinstance(module, MoEFeedForward):
            module.update_expert_bias(EXPERT_BIAS_UPDATE_RATE)

    return loss.item()
