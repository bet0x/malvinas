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

## How it plugs into the architecture

- Pick a ratio (e.g. 5:1: layers 0-4 local, layer 5 global, repeating).
- **Local layers**: the standard causal mask, plus zeroing out attention
  beyond a fixed window `w` behind each position.
- **Global layers**: the full causal mask, but with `rope_theta` swapped for
  a second, much larger value (e.g. `1e6`), and only the first `p * d_k`
  dimensions of `q`/`k` rotated (partial-rotary), leaving the rest untouched.
- Needs two separate RoPE frequency tables (local/global), not one shared
  table for every layer.

## Why this order

Two independent mechanisms bundled (windowed masking + a second RoPE
configuration), each simple on its own but touching the same attention
code path — more surface area than plans 06/07, but no new parameters or
training-loop changes.

## Status: implemented

- `src/malvinas/attention.py`: `Attention(window_size=...)` — additionally
  masks positions further than `window_size` behind the query, on top of
  the existing causal mask.
- `src/malvinas/rope.py`: `precompute_freqs_cis(..., rotary_pct=...)` and
  `apply_rotary_emb` both support partial-rotary RoPE — only the first
  `dim * rotary_pct` dims of a head get rotated, the rest pass through.
- `src/malvinas/model.py`: `MalvinasModel(local_global_ratio=...,
  local_window_size=..., global_rope_theta=..., global_rotary_pct=...)`
  alternates layer types at the configured ratio, giving each its own
  RoPE table and attention config. Tested for the 5:1 wiring itself
  (`tests/test_local_global_layers.py`) and for both mechanisms
  independently (`tests/test_attention.py`, `tests/test_rope.py`).
