# Malvinas

A Mixture-of-Experts language model framework, built from scratch — RoPE
attention + top-k routed experts with a shared expert, plus the training
pipeline (pretraining data, tokenizer, scaling, context extension, SFT,
tool use) needed to actually train one.

`docs/plans/` has the full roadmap: target size, tokenizer, datasets,
scaling laws, and the architecture techniques (borrowed from DeepSeek,
Moonshot/Kimi, and Gemma 4) implemented on the way to a real 0.5-1B MoE
with 128K context, tool use, and conversational fine-tuning. `docs/architecture.md`
diagrams the target shape. Start at [`docs/plans/README.md`](docs/plans/README.md).

## Status

All 12 architecture plans are implemented, test-first, in `src/malvinas/`
(59 tests, `uv run pytest`). What's here:

| Module | What it is |
|---|---|
| `norm.py` | `RMSNorm` |
| `rope.py` | RoPE (`precompute_freqs_cis`/`apply_rotary_emb`), partial-rotary support |
| `yarn.py` | YaRN context-extension RoPE rescaling (plan 08) |
| `attention.py` | Causal MHA: QK-Norm, optional local windowing, optional key-as-value reuse (plan 06) |
| `moe.py` | Top-k routed experts + shared expert, DeepSeek-V3-style auxiliary-loss-free load balancing (plan 03) |
| `block.py` | Pre-norm attention + pre-norm MoE FFN transformer block |
| `model.py` | Full model: tied embedding/output head, local/global layer alternation (plan 08), `resize_token_embeddings` |
| `mtp.py` | Multi-Token Prediction head (plan 09) |
| `mla.py` | Multi-head Latent Attention (plan 10) — alternative to `attention.py`'s key-reuse, not additive |
| `kda.py` | Kimi Delta Attention (plan 11) — linear-attention alternative |
| `moba.py` | Mixture of Block Attention (plan 12) — block-sparse attention alternative |
| `tokenizer.py` | Wraps the real SmolLM2 tokenizer (plan 01) |
| `data.py` | Streaming + packing for pretraining (Dolma 3 Mix) and SFT (SmolTalk, xLAM, Dolci-Think) |
| `train.py` | `train_step`/`compute_loss` — pretrain, SFT (loss masking), and MTP in one training loop |

`mla.py`/`kda.py`/`moba.py` are alternative attention modules, not wired
into the default `TransformerBlock`/`MalvinasModel` — same relationship as
plan 06 vs plan 07 (GQA, also not implemented; key-reuse was chosen).

Running an actual training job (pretrain → YaRN extension → SFT → optional
DPO) is intentionally not done yet — that happens last, on real GPU

## Development

```bash
uv sync            # creates .venv, installs dependencies
uv run pytest       # 59 tests, all green
```

## Author

Alberto Ferrer — [albertof@barrahome.org](mailto:albertof@barrahome.org) — [barrahome.org](https://barrahome.org/)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
