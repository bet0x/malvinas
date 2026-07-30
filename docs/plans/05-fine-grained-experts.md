# 5. Fine-grained experts + shared-expert isolation

**Difficulty:** medium
**Source:** DeepSeekMoE paper (arXiv:2401.06066), carried into DeepSeek-V2/V3

## Problem

Current config: `num_local_experts=4`, `num_experts_per_tok=2`, each expert's
hidden dim is `expert_dim = d_model * 2 = 256`. Only 4 coarse experts to
specialize across limits how finely the model can split up "kinds of tokens."
The always-on `shared_expert_*` already exists in the script — that part is
already DeepSeekMoE-shaped — but the routed side isn't fine-grained.

## What it does

DeepSeekMoE's recipe: split each expert into `m` smaller experts (divide
`expert_dim` by `m`, multiply `num_local_experts` by `m`), and route to
proportionally more of them (`num_experts_per_tok` scaled by `m` too), keeping
total active compute roughly constant. More, smaller experts → more
combinations of "which experts fire together" → finer specialization per
token, at the same active-parameter budget.

## How it plugs into the current script

Purely a hyperparameter reshape, no new code paths — everything is already
parameterized by `num_local_experts` and `expert_dim`:

```python
m = 4  # granularity multiplier
num_local_experts = 4 * m       # e.g. 16
num_experts_per_tok = 2 * m     # e.g. 8
intermediate_size_expert = (d_model * 2) // m  # keep expert_dim smaller
```

The `moe_expert_gate_up_proj` / `moe_expert_down_proj` tensor shapes, the
router `nn.Linear(d_model, num_local_experts)`, and the `torch.bmm` routing
logic in `forward()` all already generalize — nothing in the forward pass
assumes 4 experts specifically.

## Why this order

Zero new mechanisms — it's a resize of what's already built. Ordered after
the load-balancing plans (02/03) because fine-grained routing makes balance
*more* important (more experts to keep fed evenly), so it's more useful once
that exists.
