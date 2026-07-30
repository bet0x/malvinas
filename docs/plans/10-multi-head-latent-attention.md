# 10. Multi-head Latent Attention (MLA)

**Difficulty:** hard
**Source:** DeepSeek-V2/V3 Technical Reports (arXiv:2405.04434, arXiv:2412.19437)

## What it does

Instead of caching full per-head K/V tensors (MHA) or grouped ones (GQA, plan
07), MLA compresses K and V for a token into a single **low-rank latent
vector** (much smaller than `d_model`), caches *that*, then decompresses it
back up to per-head K/V on the fly via learned up-projection matrices whenever
attention is computed. DeepSeek-V2/V3 use this to shrink the KV cache far more
aggressively than GQA while keeping quality closer to full MHA — it's *why*
DeepSeek didn't need GQA at all.

## How it plugs into the current script

This genuinely rewrites the attention block, not a parameter tweak on top of
it:

- Add a down-projection `nn.Linear(d_model, kv_latent_dim, bias=False)`
  (`kv_latent_dim` << `d_model`, e.g. 4x smaller) applied to `x_norm` once per
  layer instead of the current `k`/`v` halves of `mha_qkv_linears[i]`.
- Add per-head up-projections `nn.Linear(kv_latent_dim, d_k, bias=False)` (one
  each for K and V) that expand the latent vector back to full K/V right
  before the `q @ k.transpose(...)` step.
- RoPE gets tricky here: DeepSeek's actual solution splits each head into a
  "content" part (compressed, no RoPE) and a small separate "rope" part
  (uncompressed, RoPE applied) — decoupled RoPE. Reproducing that faithfully
  is most of the implementation effort in this plan.
- Q can optionally get the same down/up-projection treatment (DeepSeek does
  this too, mainly to save activation memory during training, not inference
  cache).

## Why this order

The decoupled-RoPE workaround is the real complexity spike — it's not
optional (plain RoPE doesn't compose cleanly with a compressed latent K), and
getting it wrong silently degrades quality rather than erroring out loudly.
Ordered above GQA/key-reuse (same goal, smaller KV cache) because it's the
"do it properly, DeepSeek-style" version of the same problem those solve more
cheaply.
