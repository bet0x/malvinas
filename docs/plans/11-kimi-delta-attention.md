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
— but it's a genuinely different attention *mechanism*, not a variant of a
softmax-attention block.

## How it plugs into the architecture

Nothing here is a small edit — it's a new component:

- The gated delta rule as a recurrent state update: for each layer
  designated "KDA", replace the softmax attention block with a linear
  recurrence over a fixed-size state matrix, updated per token via a
  data-dependent gate and a delta-rule correction term.
- Keep RoPE-based full attention on every 4th layer (3:1 ratio) for global
  mixing.
- The reference paper's recurrence can be unrolled in parallel via a
  chunked scan for GPU efficiency — a non-trivial implementation detail on
  its own, independent of getting the math right (see status below).

## Why this order

Not a tweak to an existing attention block — a second attention mechanism
living alongside the first. Meaningfully different from MLA (plan 10),
which stays within the softmax-attention framework.

## Status: implemented (sequential form, not the chunked kernel)

`src/malvinas/kda.py`, `KimiDeltaAttention` — coded directly against eq. 1
of the actual Kimi Linear tech report (the arXiv HTML mirror didn't have
the equations; downloaded and read the PDF instead). This implements the
**sequential** recurrence, looped per token — semantically the same
function as the paper's hardware-optimized "chunkwise" form, just not that
optimized kernel. Tested against an exact closed-form check at `t=1`
derived from eq. 1 with a zero initial state. Two explicit, documented
simplifications of the surrounding neural parameterization (not the
recurrence itself): plain linear q/k/v projections instead of
ShortConv+Swish, and sigmoid standing in for the paper's unspecified decay
function `f(.)`.
