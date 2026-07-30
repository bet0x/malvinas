# 2. Load-balancing auxiliary loss

**Difficulty:** easy
**Source:** Switch Transformer / DeepSeekMoE (standard MoE technique, not novel to DeepSeek but the baseline they build on)

## Problem

`train_moe.py` routes each token to `num_experts_per_tok` of `num_local_experts`
via `torch.topk(router_logits, ...)`, but the only loss is `criterion`
(cross-entropy). Nothing discourages the router from always picking the same 1-2
experts. With 4 experts and no pressure to balance, collapse is a real risk —
the "4 experts, top-2" setup degenerates into an expensive top-1 or top-2-fixed.

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

## How it plugs into the current script

Inside the `forward()` MoE block (around `router_logits = moe_routers[i](x_norm)`):

- Compute `router_probs = F.softmax(router_logits, dim=-1)` (currently only
  `torch.topk` + `sigmoid` on the top-k logits is computed — softmax over *all*
  experts is a separate, additional quantity needed just for the aux loss).
- Compute `f_i` from `selected_experts` via a one-hot count per expert, averaged
  over `B*T*num_experts_per_tok`.
- Accumulate `aux_loss` per layer, sum across `n_layers`, return it alongside
  `logits` from `forward()` (currently returns only `logits` — needs a second
  return value).
- In the training loop, add `aux_loss_coeff * aux_loss` to `loss` before
  `.backward()`.

## Why this order

Cheapest possible win: no architecture change, no new parameters, just an
extra loss term computed from tensors `forward()` already has in scope.
Directly addresses the biggest real gap identified in this trainer.
