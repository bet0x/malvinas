# 1. Real BPE tokenizer

**Difficulty:** easy
**Source:** tokenizer swap is standard practice

## Problem

Character-level tokenization (the starting point for this project's
earliest toy experiments) is extremely wasteful for a language model: every
character is a separate prediction step, sequences get long for very little
semantic content per token, and a tiny character vocab carries no subword
structure.

## What it does

Swap the char-level tokenizer for a real pretrained BPE vocabulary, ideally
one already sized and trained for a small model (not a needlessly huge
128-200K vocab meant for frontier-scale multilingual/code coverage — see
`docs/plans/00` §3 for the sizing tradeoff). Immediate benefit: a
`block_size` of N now covers far more text than N characters would.

## Status: implemented

`src/malvinas/tokenizer.py` wraps the real **SmolLM2 tokenizer** (49,152
vocab, `HuggingFaceTB/SmolLM2-135M`) via the `tokenizers` package —
`Tokenizer.encode`/`.decode`/`.vocab_size`, plus `.add_special_tokens` for
tool-call markers (plan 00 §9). Tested with a real download-and-roundtrip
integration test.

## Why this order

Foundational and orthogonal to the MoE/attention plans below — a
preprocessing swap, not a model-architecture change. Doing it first (or at
least before any real training) means every later experiment is measured on
realistic subword sequences instead of single characters.
