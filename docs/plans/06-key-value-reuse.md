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

## How it plugs into the current script

`forward()` currently does one combined `qkv = mha_qkv_linears[i](x_norm)`
projection to `3 * d_model`, then `q, k, v = qkv.chunk(3, dim=-1)`. To try
this trick on (say) every other layer:

- Shrink that layer's projection to `2 * d_model` (`mha_qkv_linears[i] =
  nn.Linear(d_model, 2 * d_model, bias=False)`) and split into just `q, k =
  qkv.chunk(2, dim=-1)`.
- Set `v = k` directly (post-RoPE, since RoPE is applied to `q`/`k` only,
  never to `v`, in the existing code — no ordering conflict).
- Everything downstream (`attn_scores`, `attention_weights @ v`) is unchanged.

## Why this order

About the smallest possible architecture change in this list — literally
deleting a projection and an assignment. No new hyperparameters, no training-
loop changes. Good "next easy thing" once the MoE-side plans (02-05) are in.
