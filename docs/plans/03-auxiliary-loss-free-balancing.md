# 3. Auxiliary-loss-free load balancing

**Difficulty:** easy–medium
**Source:** DeepSeek-V3 Technical Report (arXiv:2412.19437)

## Problem

Plan 02's aux loss works, but it's a tradeoff: too weak and experts still
collapse, too strong and it fights the main cross-entropy objective and hurts
model quality. DeepSeek-V3 dropped the aux loss entirely in favor of a
bookkeeping-only mechanism.

## What it does

Give each expert a learnable-but-not-gradient-trained **bias** term added to
its router logit before top-k selection: `router_logits[..., i] + bias_i`.
After every batch (no backward pass involved), adjust each `bias_i` by a fixed
step: decrease it slightly if expert `i` was over-loaded that batch, increase
it if under-loaded. This steers routing toward balance via a simple control
loop, without ever touching the training loss.

## How it plugs into the architecture

- A per-expert `expert_bias` buffer — plain tensor, no gradient, not a
  learnable parameter.
- Before top-k selection, add the bias to the router logits for routing
  purposes; the actual gate *value* (how much an expert's output gets
  scaled by) uses the original, unbiased logit — the bias only steers which
  experts get picked, not how strongly.
- Outside the training step (no gradient involved): count how many tokens
  went to each expert, and nudge the bias down for over-loaded experts, up
  for under-loaded ones, by a fixed step.

## Why this order

Slightly more moving parts than plan 02 (a control-loop update step outside
the autograd graph), but no new loss term to tune against the main
objective — arguably *simpler* to reason about.

## Status: implemented — chosen over plan 02

`src/malvinas/moe.py`: `MoEFeedForward.expert_bias` (buffer, added to router
logits before top-k only, not to the gate value — matches DeepSeek-V3's own
distinction) and `update_expert_bias(update_rate)` (the bookkeeping nudge,
`@torch.no_grad()`). Tested: a large bias forces its expert to always be
selected regardless of the router's learned weights, and the update method
moves an over-loaded expert's bias down and under-loaded ones up by exactly
`update_rate`. Wired into `train.py`'s `train_step`, called after every
optimizer step.
