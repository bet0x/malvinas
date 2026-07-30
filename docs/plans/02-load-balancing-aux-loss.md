# 2. Load-balancing auxiliary loss

**Difficulty:** easy
**Source:** Switch Transformer / DeepSeekMoE (standard MoE technique, not novel to DeepSeek but the baseline they build on)

## Problem

A top-k MoE router with no balancing pressure has a real collapse risk —
with only cross-entropy loss, nothing discourages the router from always
picking the same 1-2 experts, and a "many experts, top-k" setup degenerates
into an expensive top-k-fixed.

## What it does

Add a small auxiliary loss term that penalizes uneven expert usage across a
batch. Classic formulation (Switch Transformer / DeepSeekMoE):

```
f_i = fraction of tokens routed to expert i (from the top-k selection)
P_i = mean router probability assigned to expert i (softmax over router_logits)
aux_loss = num_local_experts * sum_i(f_i * P_i)
loss = criterion(...) + aux_loss_coeff * aux_loss
```

`aux_loss` is minimized when routing is uniform across experts; `aux_loss_coeff`
is typically small (0.01–0.1) so it nudges balance without dominating the main
objective.

## How it plugs into the architecture

Inside `MoEFeedForward.forward()` (`src/malvinas/moe.py`), around where
`router_logits`/`selected_experts` are computed:

- Compute `router_probs = F.softmax(router_logits, dim=-1)` (a separate
  quantity from the top-k selection itself, needed only for this loss).
- Compute `f_i` from `selected_experts` via a one-hot count per expert,
  averaged over all routed tokens.
- Return `aux_loss` alongside the module's output (needs a second return
  value or an instance attribute the training loop reads after the call —
  same pattern already used for `expert_bias`/`last_selected_experts`).
- In the training loop, add `aux_loss_coeff * aux_loss` to the main loss
  before `.backward()`.

## Why this order

Cheapest possible win: no architecture change, no new parameters, just an
extra loss term computed from tensors already in scope.

## Status: not implemented — plan 03 was chosen instead

Plan 03's auxiliary-loss-free dynamic bias (`MoEFeedForward.expert_bias` +
`update_expert_bias`, no gradient, no extra loss term) is what's actually
wired in and tested. This plan stays documented as the fallback if 03 ever
proves harder to tune well in practice — see `docs/architecture.md`.
