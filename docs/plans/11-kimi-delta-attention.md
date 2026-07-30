# 11. Kimi Delta Attention (linear-attention hybrid)

**Difficulty:** hard
**Source:** Kimi Linear (Moonshot AI), arXiv:2510.26692; production use in Kimi K3

## What it does

Kimi Delta Attention (KDA) is a **linear** attention mechanism (state updated
with a gated delta rule, O(1) per-token instead of attending over the full
growing context) used for most layers, with regular full softmax attention
kept at a periodic interval (Kimi Linear uses a 3:1 KDA-to-full-attention
ratio) to preserve global information flow. Claimed up to 6x faster decoding
and 75% less KV-cache/memory at long context versus full attention everywhere
— but it's a genuinely different attention *mechanism*, not a variant of the
softmax-attention block this script already has.

## How it plugs into the current script

Nothing here is a small edit — it's a new component:

- Implement the gated delta rule as a recurrent state update: for each layer
  designated "KDA", replace the `q @ k.transpose(...)` softmax block with a
  linear recurrence over a fixed-size state matrix, updated per token via a
  data-dependent gate and a delta-rule correction term (see the paper for the
  exact recurrence — this is the novel part, not something to improvise from
  memory).
- Keep RoPE-based full attention (the block already in `forward()`) on every
  4th layer (3:1 ratio) for global mixing.
- Training-wise, the recurrence can be unrolled in parallel via a chunked
  scan (needed for GPU efficiency) — a non-trivial implementation detail on
  its own, independent of getting the math right.

## Why this order

This is not a tweak to the existing attention block, it's a second attention
implementation living alongside the first, plus a parallel-scan formulation
to make training tractable on GPU. Meaningfully harder than MLA (plan 10),
which stays within the softmax-attention framework this script already has.
