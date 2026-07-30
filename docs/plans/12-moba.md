# 12. Mixture of Block Attention (MoBA)

**Difficulty:** hardest
**Source:** [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) — "Mixture of Block Attention for Long-Context LLMs"

## What it does

Applies the MoE idea to attention itself instead of the FFN: split the
context into fixed-size blocks, and for each query, run a lightweight router
that picks which *blocks* of the context are worth attending to (instead of
attending to every position). Full attention and block-sparse attention
become the same mechanism at different router settings — MoBA can fall back
to full attention when needed.

## Why it's the hardest item on this list

It combines two things this script already has separately — top-k routing
(the MoE FFN's router) and attention (the softmax block) — but *inside* the
attention mechanism itself, at the block level, not the token level. That
means:

- A second router (block-level, not token-level like the MoE FFN router).
- Attention scores are only computed for selected blocks — needs
  block-sparse indexing/gathering before the `q @ k.transpose(...)` step,
  not a dense computation with parts masked out.
- Causal masking interacts with block selection (a block can't be selected if
  it's entirely in the future) — an extra correctness constraint on top of
  the existing `causal_mask`.
- At `block_size=64` (this script's whole sequence length), there may not be
  enough context length for block-sparse routing to make sense at all — MoBA
  is designed for long-context regimes (the paper targets far longer
  sequences than this toy setup uses). Realistically only worth attempting
  after `block_size` has already been scaled up substantially.

## Recommendation

Don't scope an implementation plan for this one yet — read the MoBA repo's
reference implementation directly when you're ready, and treat it as a
research spike rather than a straightforward port like the rest of this list.
