# 1. Real BPE tokenizer (via Gigatoken)

**Difficulty:** easy
**Source:** tokenizer swap is standard practice; [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) is the fast *encoder* being proposed for it

## Correction before scoping this

Gigatoken is not a new/better tokenization *scheme* — it doesn't define a new
vocabulary or a new BPE algorithm. It's a Rust implementation that runs
**existing** tokenizers (GPT-2, Llama 3/4, Qwen, DeepSeek V3, Kimi K2, etc.) at
much higher throughput (claims ~GB/s vs ~30-70 MB/s for HF `tokenizers` /
tiktoken on the same vocab — verified from the repo's own README benchmark
table, 3.8k stars, active as of 2026-07-30). It ships a compatibility mode
(`gt.Tokenizer(hf_tokenizer).as_hf()`) that's a drop-in replacement, plus a
faster native API that reads files directly in Rust. So: "better than all" is
true for *encoding speed*, not for token quality — it doesn't change what a
model learns, only how fast you can preprocess text into token IDs.

## Problem

`train_moe.py` currently tokenizes at the **character level** (`chars =
sorted(list(set(corpus_raw)))`), confirmed to be a simplification with no BPE
anywhere in the original notebook it was extracted from. Character-level
tokenization is extremely wasteful for a language model: every character is a
separate prediction step, sequences get long for very little semantic content
per token, and `vocab_size` (currently 36) carries no subword structure.

## What it does

Swap the char-level tokenizer for a real pretrained BPE vocabulary (e.g.
GPT-2's, or any HF tokenizer), encoded via Gigatoken in compatibility mode.
Immediate benefits: `block_size=64` characters becomes ~64 *subword* tokens
(covers far more text per training sequence), and if the corpus is scaled up
(see the earlier discussion — char-level barely benefits from a bigger
corpus), preprocessing a large text file stays fast instead of becoming the
bottleneck.

## How it plugs into the current script

- `pip install gigatoken`, pick a small pretrained vocab (GPT-2's ~50k is the
  simplest starting point).
- Replace the `chars`/`char_to_int`/`int_to_char` block and
  `encoded_corpus = [char_to_int[ch] for ch in corpus_raw]` with:
  ```python
  import gigatoken as gt
  from transformers import AutoTokenizer
  hf_tok = AutoTokenizer.from_pretrained("gpt2")
  tokenizer = gt.Tokenizer(hf_tok).as_hf()
  encoded_corpus = tokenizer.encode(corpus_raw)
  vocab_size = hf_tok.vocab_size
  ```
- Generation-side decoding (`int_to_char.get(...)`) becomes `hf_tok.decode(...)`.
- Everything downstream (`token_embedding_table`, `output_linear_layer`,
  block/batch construction) already keys off `vocab_size` — no other changes
  needed, aside from `vocab_size` jumping from 36 to ~50k, which meaningfully
  grows the embedding + output layer (see the "vocab_size barely matters at
  36" caveat from earlier — that stops being true here).

## Why this order

Foundational and orthogonal to the MoE/attention plans below — but genuinely
easy: it's a preprocessing swap, not a model-architecture change. Doing it
first means every later experiment (aux loss, GQA, MLA, ...) is measured on
realistic subword sequences instead of single characters.
