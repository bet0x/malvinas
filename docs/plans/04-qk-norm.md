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

## How it plugs into the architecture

Add an `RMSNorm(d_k)` instance for `q` and one for `k`, applied per head
right after the QKV projection and before RoPE — reusing the existing
`RMSNorm` module rather than a new normalization formula.

## Why this order

Pure addition, reuses the same `RMSNorm` module used everywhere else in the
model — no new architecture, no new training-loop logic. At small scale
this won't visibly benefit, but it's the correct habit to have in place
before scaling `d_model`/`n_layers` up.

## Status: implemented

`src/malvinas/attention.py`: `Attention.q_norm` / `Attention.k_norm`
(`RMSNorm(d_k)` each), applied in `forward()` right after the QKV chunk and
before `apply_rotary_emb`. Tested in `tests/test_attention.py` via a
zeroed-`q_norm.weight` case: forces every attention score to 0, so the
causal-masked softmax becomes uniform and the output collapses to an exact
causal mean of `v` — a precise, checkable consequence of the norm actually
running where it should.
