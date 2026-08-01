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

All 12 architecture plans are implemented, test-first, in `src/malvinas/`.
The project also includes a runnable, resumable training CLI. What's here:

| Module | What it is |
|---|---|
| `norm.py` | `RMSNorm` |
| `rope.py` | RoPE (`precompute_freqs_cis`/`apply_rotary_emb`), partial-rotary support |
| `yarn.py` | YaRN context-extension RoPE rescaling (plan 08) |
| `attention.py` | Causal MHA: QK-Norm, bounded local windows, document masks, FlexAttention/SDPA, and KV cache |
| `moe.py` | Top-k routed experts + shared expert, auxiliary-loss-free balancing, and selectable grouped kernels |
| `block.py` | Pre-norm attention + pre-norm MoE FFN transformer block |
| `model.py` | Full model: tied embedding/output head, local/global layer alternation (plan 08), `resize_token_embeddings` |
| `mtp.py` | Multi-Token Prediction head (plan 09) |
| `mla.py` | Multi-head Latent Attention (plan 10) — alternative to `attention.py`'s key-reuse, not additive |
| `kda.py` | Kimi Delta Attention (plan 11) — linear-attention alternative |
| `moba.py` | Mixture of Block Attention (plan 12) — block-sparse attention alternative |
| `tokenizer.py` | Wraps the real SmolLM2 tokenizer (plan 01) |
| `data.py` | Streaming document-aware packing for pretraining (Dolma 3 Mix) and SFT (SmolTalk, xLAM, Dolci-Think) |
| `train.py` | `train_step`/`compute_loss` — pretrain, SFT (loss masking), and MTP in one training loop |
| `config.py` | Selectable `tiny`, `0.5b`, `1b-deep`, and `1b-wide` model presets |
| `checkpoint.py` | Durable atomic checkpoints with model, optimizer, progress, RNG state, and retention |
| `cli.py` | `malvinas-train` CLI with DDP, validation, metrics, profiling, prefetch, resume, and stage initialization |
| `generate.py` | Cached text generation with temperature, top-k/top-p, EOS, and deterministic seeding |
| `evaluate.py` | Reproducible streaming evaluation from a saved model artifact |
| `benchmark_moe.py` | Reproducible expert-kernel throughput and numerical comparison |

`mla.py`/`kda.py`/`moba.py` are alternative attention modules, not wired
into the default `TransformerBlock`/`MalvinasModel` — same relationship as
plan 06 vs plan 07 (GQA, also not implemented; key-reuse was chosen).

## Training

Start a small pretraining run:

```bash
uv run malvinas-train \
  --mode pretrain \
  --preset tiny \
  --model-name malvinas-tiny \
  --max-examples 1000 \
  --block-size 256 \
  --batch-size 2 \
  --tokens-per-update 2048 \
  --max-steps 100 \
  --warmup-steps 10 \
  --save-every 25 \
  --keep-checkpoints 3
```

`--tokens-per-update` defines the effective batch and uses gradient
accumulation when it is larger than `batch-size * block-size`. `--max-steps`
counts optimizer updates, not micro-batches. The default AdamW setup applies
weight decay to matrix parameters, keeps scalar parameters un-decayed, and
supports separate embedding decay, betas, cosine LR decay, and gradient
clipping through CLI options. `--precision auto` prefers BF16 on supported
CUDA hardware and otherwise uses scaled FP16 on CUDA or FP32 on CPU.

Continue the same run, restoring model, optimizer, scheduler, GradScaler,
step, random state, and the number of consumed data blocks:

```bash
uv run malvinas-train \
  --mode pretrain \
  --resume latest \
  --max-steps 200
```

Start a new SFT stage from pretrained weights. This intentionally starts a
fresh optimizer and step counter:

```bash
uv run malvinas-train \
  --mode sft \
  --init-from models/malvinas-tiny/checkpoints/pretrain-step-00000200.pt \
  --max-steps 100
```

By default, training outputs are grouped by model and stage:

```text
models/
  malvinas-tiny/
    model.pt
    checkpoints/
      pretrain-step-00000100.pt
  malvinas-tiny-sft/
    model.pt
    checkpoints/
      sft-step-00000100.pt
```

`model.pt` contains the final weights and model metadata for inference. Each
checkpoint also includes optimizer, scheduler, mixed-precision scaler,
progress, and random state so training can resume exactly. Use `--model-name`
to choose the directory name, `--models-dir` to change the `models/` root, or
`--checkpoint-dir` as an explicit override.

Periodic checkpoints are rotated with `--keep-checkpoints` (default: 3).
`--milestone-every` writes separately named checkpoints that retention does not
remove. Periodic validation uses `--validation-dataset`, `--validate-every`,
and `--validation-steps`; its best resumable checkpoint is `best.pt`. Training
and validation metrics are appended to `metrics.jsonl`, and `--profile-steps`
exports a Chrome trace to `profile.json`.

The final layout is:

```text
models/{model_name}/
  model.pt
  metrics.jsonl
  profile.json                 # only when profiling is enabled
  checkpoints/
    best.pt                    # only when validation is enabled
    pretrain-step-00000100.pt
    pretrain-milestone-00001000.pt
```

Run distributed data-parallel training with one process per GPU. The effective
`--tokens-per-update` is global across all ranks:

```bash
torchrun --standalone --nproc-per-node=8 -m malvinas.cli \
  --mode pretrain \
  --preset 0.5b \
  --model-name malvinas-05b \
  --tokens-per-update 1048576 \
  --batch-size 1 \
  --block-size 8192
```

Before a paid GPU run, compare the reference and grouped expert kernels using
the intended dimensions and precision:

```bash
uv run malvinas-benchmark-moe \
  --device cuda \
  --dtype bfloat16 \
  --sequence-length 8192
```

Generate from the final `model.pt`, or evaluate it against a held-out stream:

```bash
uv run malvinas-generate \
  --model models/malvinas-05b/model.pt \
  --prompt "La soberania argentina"

uv run malvinas-evaluate \
  --model models/malvinas-05b/model.pt \
  --dataset allenai/dolma3_mix-150B-1025 \
  --split train \
  --max-batches 100
```

Use `--preset 0.5b`, `--preset 1b-deep`, or `--preset 1b-wide` for the
larger target configurations. Those presets require appropriate GPU memory;
`tiny` is intended for pipeline validation. `--moe-kernel auto` chooses
`grouped_mm` on CUDA and the eager reference on CPU. Document-aware global
attention similarly uses FlexAttention on CUDA and SDPA as the portable
fallback. Run `uv run malvinas-train --help` for all dataset, precision,
validation, performance, and checkpoint options.

## Development

```bash
uv sync            # creates .venv, installs dependencies
uv run pytest
```

## Author

Alberto Ferrer — [albertof@barrahome.org](mailto:albertof@barrahome.org) — [barrahome.org](https://barrahome.org/)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
