# 7. Grouped Query Attention (GQA)

**Difficulty:** medium
**Source:** widely adopted (Llama2-70B onward, Llama4, DeepSeek-V2/V3 uses MLA instead — see plan 10); mentioned here because it's the standard alternative to plan 06 for the same KV-cache-shrinking goal

## Problem

`forward()` gives every one of the `n_heads` query heads its own K/V head —
full MHA. That's the most expensive KV-cache option; GQA is the usual
middle ground between full MHA (best quality, biggest cache) and MQA (one
shared K/V head for all queries, smallest cache, more quality loss).

## What it does

Split `n_heads` query heads into `n_kv_groups` groups (e.g. 4 query heads,
2 KV groups → 2 query heads share each K/V head). Project K/V at the smaller
`n_kv_groups` width, then repeat each K/V head across its group before the
attention dot product.

## How it plugs into the current script

- Add `n_kv_heads` (must divide `n_heads`) alongside `n_heads` in the
  hyperparameter block.
- Change `mha_qkv_linears[i]` to project to `d_model + 2 * n_kv_heads * d_k`
  instead of `3 * d_model`, and split accordingly (`q` at full width, `k`/`v`
  at `n_kv_heads * d_k`).
- Before the `q @ k.transpose(-2, -1)` step, repeat `k`/`v` along the head
  dimension: `k.repeat_interleave(n_heads // n_kv_heads, dim=1)` (after the
  `.permute(0, 2, 1, 3)` that puts heads on dim 1).
- RoPE application needs `k`'s rotation computed at `n_kv_heads` width before
  the repeat, not after.

## Why this order

More moving parts than plan 06 (key-value reuse) — new hyperparameter, shape
bookkeeping in two places (projection width, repeat-before-attention), and
RoPE has to happen before the repeat, not after. Same underlying goal as
plan 06; the two are alternatives, not stacked (pick key-reuse OR GQA per
layer, not both).
