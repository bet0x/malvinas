# 8. Local/global sliding-window attention + dual RoPE

**Difficulty:** medium
**Source:** Gemma 4 Technical Report (arXiv:2607.02770), Google DeepMind

## What it is

Gemma 4 alternates attention layer types at a fixed ratio — 5:1 (local
sliding-window : global full-context) for the 12B/26B-A4B/31B variants, 4:1
for the smallest E2B. Local layers use plain RoPE at `theta=10000`; global
layers use **partial-rotary RoPE** (`pp-RoPE`, only a fraction `p=0.25` of
each head's dimensions get rotated) at `theta=1000000`. The high theta on
global layers is what lets them generalize to long context; the low theta +
short window on local layers keeps most of the compute cheap.

## How it plugs into the current script

- Pick a ratio, e.g. with `n_layers=4`: layers 0-2 local, layer 3 global (or
  scale the ratio to whatever `n_layers` you're running).
- **Local layers**: reuse `causal_mask` as-is but additionally zero out
  attention beyond a fixed window `w` behind each position — i.e. mask
  `j < i - w` in addition to the existing `j > i` causal mask.
- **Global layers**: keep the current full causal mask, but swap `rope_theta`
  for a second, much larger value (e.g. `1e6`) when precomputing that layer's
  `inv_freq`, and only rotate the first `p * d_k` dimensions of `q`/`k`
  (leave the rest untouched) instead of the full `d_k` the script rotates
  today.
- Needs two `inv_freq` tensors (local/global) instead of the single shared
  one `forward()` currently precomputes once outside the layer loop.

## Why this order

Two independent mechanisms bundled (windowed masking + a second RoPE
configuration), each simple on its own but touching the same code path
(`forward()`'s per-layer attention block) — more surface area than plans
06/07, but no new parameters or training-loop changes. Ordered after GQA
since it's a similar "attention efficiency" family, one step more involved.
