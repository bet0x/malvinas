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

## Why it's hard

It combines two things elsewhere kept separate — top-k routing (the MoE
FFN's router) and attention (the softmax block) — but *inside* the attention
mechanism itself, at the block level, not the token level:

- A second router (block-level, not token-level like the MoE FFN router).
- Causal masking interacts with block selection (a block can't be selected
  if it's entirely in the future) — an extra correctness constraint on top
  of the usual causal mask.
- MoBA is designed for long-context regimes; it's most meaningful once
  context length is actually large (see `docs/plans/00` §6/§8), not at a
  short sequence length.

## Status: implemented, adapted from the real reference code

`src/malvinas/moba.py`, `MoBAAttention` — adapted directly from
`github.com/MoonshotAI/MoBA`'s `moba/moba_naive.py` (the authors' own
"clean version ... for educational purposes" reference implementation),
not reconstructed from the paper. Changed from the original's
variable-length (`cu_seqlens`) batching to this project's padded
`(B, T, n_heads, d_head)` tensors; kept the routing logic itself faithful:
a per-chunk mean-key gate, forced inclusion of the query's own chunk
(regardless of its relevance score), exclusion of future chunks, and top-k
selection over the rest. Tested specifically for the detail most likely to
break silently without the source in hand: a query's own chunk stays
attended even when every block is made to tie on relevance score.
