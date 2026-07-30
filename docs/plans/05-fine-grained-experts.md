# 5. Fine-grained experts + shared-expert isolation

**Difficulty:** medium
**Source:** DeepSeekMoE paper (arXiv:2401.06066), carried into DeepSeek-V2/V3

## Problem

A small number of coarse experts limits how finely the model can split up
"kinds of tokens." The always-on shared expert is already DeepSeekMoE-shaped
— it's the routed side that isn't fine-grained by default.

## What it does

DeepSeekMoE's recipe: split each expert into `m` smaller experts (divide
`expert_dim` by `m`, multiply `num_local_experts` by `m`), and route to
proportionally more of them (`num_experts_per_tok` scaled by `m` too), keeping
total active compute roughly constant. More, smaller experts → more
combinations of "which experts fire together" → finer specialization per
token, at the same active-parameter budget.

## How it plugs into the architecture

Purely a hyperparameter reshape, no new code paths:

```python
m = 4  # granularity multiplier
num_experts = base_num_experts * m
top_k = base_top_k * m
expert_dim = base_expert_dim // m  # keep total active compute roughly constant
```

## Why this order

Zero new mechanisms — it's a resize of what's already built. Fine-grained
routing makes load balancing *more* important (more experts to keep fed
evenly), so it pairs naturally with plan 03.

## Status: implemented — as a constructor choice, no new code

`src/malvinas/moe.py`'s `MoEFeedForward(num_experts, top_k, expert_dim, ...)`
already generalizes to any expert count/size — nothing in `forward()`
assumes a specific number of experts. Going fine-grained is just picking
different constructor arguments when building the model.
