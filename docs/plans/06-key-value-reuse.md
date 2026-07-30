# 6. Key-as-value reuse in global attention

**Difficulty:** easy
**Source:** Gemma 4 Technical Report (arXiv:2607.02770), Google DeepMind

## What it is

Gemma 4 reuses the same projection for both K and V in its global attention
layers: `values = keys` (stated explicitly in the paper, applied to all
variants except the two smallest, E2B/E4B). Skips computing a separate V
projection for those layers — one less matrix multiply, less to store in the
KV cache, with apparently negligible quality cost since it shipped in a
production model family.

## How it plugs into the architecture

A combined QKV projection to `3 * d_model`, split via `chunk(3, ...)`, is
the baseline. Applying this trick means: shrink that projection to
`2 * d_model`, split into just `q, k`, and set `v = k` directly — after RoPE
has been applied to `k` (RoPE never touches `v` either way, so there's no
ordering conflict). Everything downstream (attention scores, the weighted
sum over `v`) is unchanged.

## Why this order

About the smallest possible architecture change in this list — literally
deleting a projection and an assignment. No new hyperparameters, no
training-loop changes.

## Status: implemented

`src/malvinas/attention.py`: `Attention(reuse_key_as_value=True)` shrinks
the QKV projection to `2 * d_model` and sets `v = k` post-RoPE, exactly as
scoped above. Tested for the projection actually shrinking (weight shape
check) and for correctness at `seq_len=1` (causal attention weight is
exactly 1.0 on the sole position, so the output must equal
`out_proj(k)` — verified by recomputing `k` independently from the module's
own sublayers and comparing).
