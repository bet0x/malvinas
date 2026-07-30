# Malvinas

A Mixture-of-Experts language model framework, built from scratch — RoPE
attention + top-k routed experts with a shared expert, plus the training
pipeline (pretraining data, tokenizer, scaling, context extension, SFT,
tool use) needed to actually train one.

`docs/plans/` has the full roadmap: target size, tokenizer, datasets,
scaling laws, and the architecture techniques (borrowed from DeepSeek,
Moonshot/Kimi, and Gemma 4) being implemented on the way to a real 0.5-1B
MoE with 128K context, tool use, and conversational fine-tuning. Start at
[`docs/plans/README.md`](docs/plans/README.md).

## Author

Alberto Ferrer — [albertof@barrahome.org](mailto:albertof@barrahome.org) — [barrahome.org](https://barrahome.org/)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
