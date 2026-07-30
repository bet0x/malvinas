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

## How it plugs into the architecture

- One extra head predicting `t+2` (DeepSeek-style: its own small transformer
  block, taking the main trunk's hidden state plus the embedding of the
  already-known `t+1` token; Gemma-4-style would instead be a standalone
  cross-attention drafter over the main model's KV cache).
- A second target tensor, shifted by 2 instead of 1.
- A second loss term: `loss = main_loss + mtp_weight * mtp_loss`.

## Why this order

Requires touching data prep (second shifted target), the forward pass (a
new head), and the training loop (a second loss term) simultaneously —
three places instead of one, despite not being conceptually exotic.

## Status: implemented (DeepSeek-style)

`src/malvinas/mtp.py`, `MTPHead` — combine projection + norm + one more
`TransformerBlock`, predicting `t+2` from the trunk's hidden state and the
ground-truth `t+1` embedding. Wired into `MalvinasModel.forward_with_mtp`
and `train.py`'s `train_step(mtp_target_ids=...)`, which adds the weighted
MTP loss on top of the main next-token loss. Also covers an interaction
that would otherwise be a silent bug: `resize_token_embeddings` re-ties
`mtp_head`'s embedding/output-head references to the new (grown) ones.
