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

## How it plugs into the architecture

This genuinely rewrites the attention block, not a parameter tweak on top
of an existing one:

- A down-projection to a KV latent vector (much smaller than `d_model`),
  replacing separate `k`/`v` projections.
- Per-head up-projections that expand the latent vector back to full K/V
  right before the attention dot product.
- RoPE gets tricky here: DeepSeek's actual solution splits each head into a
  "content" part (compressed, no RoPE) and a small separate "rope" part
  (uncompressed, RoPE applied) — decoupled RoPE, with the rope part's key
  *shared across all heads* (no per-head weights, unlike the content key).
  Reproducing that faithfully is most of the implementation effort here.
- Q gets the same down/up-projection treatment (DeepSeek does this too,
  mainly to save activation memory during training, not inference cache).

## Why this order

The decoupled-RoPE detail is the real complexity spike — it's not optional
(plain RoPE doesn't compose cleanly with a compressed latent K), and getting
it wrong silently degrades quality rather than erroring out loudly.

## Status: implemented (as an alternative, not the default)

`src/malvinas/mla.py`, `MLAAttention` — coded directly against DeepSeek-V2's
own eq. 9-19 (arXiv:2405.04434), fetched and transcribed before writing any
code, not reconstructed from memory. Tested: causal masking, that the KV
latent is genuinely narrower than the full per-head KV width, and
specifically the detail called out above (the rope key's weight matrix has
no per-head dimension — checked structurally, not just by shape of the
output). Not wired into `TransformerBlock`/`MalvinasModel` as the active
default; available alongside `Attention` as the "do it properly,
DeepSeek-style" alternative to plans 06/07's cheaper KV-cache cuts.
