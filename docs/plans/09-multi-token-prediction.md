# 9. Multi-Token Prediction (MTP)

**Difficulty:** medium-hard
**Source:** DeepSeek-V3 Technical Report (arXiv:2412.19437); Gemma 4 (arXiv:2607.02770) ships a variant too — a small autoregressive "drafter" head that cross-attends to the main model's KV cache instead of sharing the trunk

## What it does

Train the model to predict not just the next token but `k` future tokens at
each position, using extra lightweight prediction heads. DeepSeek-V3's
version: each extra head has its own small transformer block, takes the main
trunk's output plus the embedding of the *already-known* next tokens, and is
trained with its own cross-entropy loss, summed into the total loss (down-
weighted). Densifies the training signal per token and doubles as a built-in
speculative-decoding draft model at inference — Gemma 4's variant separates
this into a standalone small drafter that only needs the main model's KV
cache, not a full parallel trunk.

## How it plugs into the current script

- Add `mtp_depth = 1` (predict `t+1` in addition to the base `t+1`... i.e.
  `t+2` beyond the normal target) and one small extra head reusing the same
  `d_model` width: an `nn.Linear(d_model, vocab_size, bias=False)` plus (DeepSeek
  style) a tiny transformer block, or (Gemma-4 style) just a cross-attention
  layer over the final `x` from `forward()`.
- Needs a second target tensor shifted by 2 instead of 1
  (`train_y2 = full_data_sequence[i+2 : i+block_size+2]`, built alongside the
  existing `train_x`/`train_y` construction).
- In the training loop: `loss = criterion(...) + mtp_weight *
  criterion(mtp_logits.view(...), yb2.view(...))`.

## Why this order

Requires touching data prep (second shifted target), the forward pass (a new
head), and the training loop (a second loss term) simultaneously — three
places instead of one, which is why it lands above the single-loss-term
plans (02/03) and the pure-attention tweaks (06/07/08) despite not being
conceptually exotic.
