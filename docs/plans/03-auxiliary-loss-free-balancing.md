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

## How it plugs into the current script

- Add `expert_bias = [torch.zeros(num_local_experts, device=device) for _ in range(n_layers)]` — plain tensors, `requires_grad=False`, not added to `all_model_parameters`.
- In `forward()`, right before `torch.topk(router_logits, ...)`, add
  `router_logits = router_logits + expert_bias[i]`.
- After `optimizer.step()` in the training loop, for each layer compute how
  many tokens went to each expert this batch (same `f_i` count as plan 02) and
  nudge: `expert_bias[i][e] -= update_rate if f_i[e] > target else +update_rate`.

## Why this order

Slightly more moving parts than plan 02 (a control-loop update step outside
the autograd graph), but no new loss term to tune against the main objective —
arguably *simpler* to reason about once the counting logic from plan 02 exists.
Natural follow-up once you've built the expert-usage counting from plan 02;
can replace it outright, or the two can be compared side by side.
