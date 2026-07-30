# 4. QK-Norm

**Difficulty:** easy–medium
**Source:** used across most 2024-2026 frontier models (DeepSeek-V3, Llama4, Qwen2, Gemma2) — general training-stability technique, not DeepSeek/Kimi-specific

## Problem

`forward()` computes attention scores directly from raw `q`/`k` projections
(after RoPE): `attn_scores = (q @ k.transpose(-2, -1)) * (d_k ** -0.5)`. As
`d_model`/`n_heads` grow, unnormalized Q/K magnitudes can drift during
training and destabilize the softmax (a known failure mode in larger
transformers).

## What it does

Apply RMSNorm (or plain L2 norm) to `q` and `k` independently, per head,
right after the QKV projection and before RoPE is applied. Keeps the dot
product's scale bounded regardless of how large the learned projections grow.

## How it plugs into the current script

- Add two more RMSNorm weight lists, `qk_norm_weight_q` / `qk_norm_weight_k`
  (shape `(d_k,)`, one pair per layer, initialized like
  `rmsnorm_weights_input`).
- In `forward()`, right after `q, k, v = qkv.chunk(3, dim=-1)` and before the
  RoPE `view_as_complex` block, apply the same RMSNorm formula already used
  elsewhere in the script (`x_float.pow(2).mean(-1, keepdim=True)` pattern) to
  `q` and `k` along their last dim (`d_k`).

## Why this order

Pure addition, reuses the RMSNorm math already written three times in the
script — no new architecture, no new training-loop logic. At `d_model=128`
this toy model won't visibly benefit, but it's the correct habit to have in
place before scaling `d_model`/`n_layers` up.
