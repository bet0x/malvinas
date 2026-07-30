# 7. Grouped Query Attention (GQA)

**Difficulty:** medium
**Source:** widely adopted (Llama2-70B onward, Llama4, DeepSeek-V2/V3 uses MLA instead — see plan 10); mentioned here because it's the standard alternative to plan 06 for the same KV-cache-shrinking goal

## Problem

Giving every query head its own K/V head (full MHA) is the most expensive
KV-cache option; GQA is the usual middle ground between full MHA (best
quality, biggest cache) and MQA (one shared K/V head for all queries,
smallest cache, more quality loss).

## What it does

Split `n_heads` query heads into `n_kv_groups` groups (e.g. 4 query heads,
2 KV groups → 2 query heads share each K/V head). Project K/V at the
smaller `n_kv_groups` width, then repeat each K/V head across its group
before the attention dot product.

## How it plugs into the architecture

- Add `n_kv_heads` (must divide `n_heads`) alongside `n_heads`.
- Shrink the QKV projection to `d_model + 2 * n_kv_heads * d_k` instead of
  `3 * d_model`, and split accordingly (`q` at full width, `k`/`v` at
  `n_kv_heads * d_k`).
- Before the `q @ k.transpose(-2, -1)` step, repeat `k`/`v` along the head
  dimension to match `n_heads` (after permuting heads onto their own axis).
- RoPE needs `k`'s rotation computed at `n_kv_heads` width *before* the
  repeat, not after.

## Why this order

More moving parts than plan 06 (key-value reuse) — new hyperparameter,
shape bookkeeping in two places (projection width, repeat-before-attention),
and RoPE has to happen before the repeat, not after.

## Status: not implemented — plan 06 was chosen instead

Same underlying goal as plan 06 (`Attention(reuse_key_as_value=True)` in
`src/malvinas/attention.py`); the two are alternatives, not stacked. GQA
stays documented here in case key-value reuse ever proves too aggressive a
cut for quality and a middle ground is needed.
