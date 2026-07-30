# Malvinas — architecture and training plans

Techniques borrowed from DeepSeek (V2/V3, DeepSeekMoE), Moonshot/Kimi (Kimi
Linear, MoBA), and Gemma 4 (Google DeepMind), plus the training pipeline
needed to actually train the model. Ordered easy → hard. Each file is a
standalone spec: what it is, why it helps, how it plugs into the
architecture — and, for the ones now built, a "Status: implemented" section
pointing at the real module/tests in `src/malvinas/`.

0. [Training process, data, and scaling](00-training-process-and-data.md) — target size, tokenizer, data volume/sources, 4096→128K context (YaRN), pretrain → context-extend → SFT+tool-use path. Read this first; everything else assumes it.
1. [Real BPE tokenizer](01-tokenizer.md) — implemented
2. [Load-balancing auxiliary loss](02-load-balancing-aux-loss.md) — not implemented (plan 03 chosen instead)
3. [Auxiliary-loss-free load balancing (DeepSeek-V3)](03-auxiliary-loss-free-balancing.md) — implemented
4. [QK-Norm](04-qk-norm.md) — implemented
5. [Fine-grained experts + shared-expert isolation (DeepSeekMoE)](05-fine-grained-experts.md) — implemented (constructor config, no new code needed)
6. [Key-as-value reuse in global attention (Gemma 4)](06-key-value-reuse.md) — implemented
7. [Grouped Query Attention (GQA)](07-grouped-query-attention.md) — not implemented (plan 06 chosen instead)
8. [Local/global sliding-window attention + dual RoPE (Gemma 4)](08-local-global-sliding-window.md) — implemented
9. [Multi-Token Prediction (DeepSeek-V3 / Gemma 4)](09-multi-token-prediction.md) — implemented
10. [Multi-head Latent Attention (DeepSeek-V2/V3)](10-multi-head-latent-attention.md) — implemented (alternative module, not wired in as default)
11. [Kimi Delta Attention / linear-attention hybrid (Moonshot)](11-kimi-delta-attention.md) — implemented (alternative module, not wired in as default)
12. [Mixture of Block Attention (MoBA, Moonshot)](12-moba.md) — implemented (alternative module, not wired in as default)

See `docs/architecture.md` for the target shape these compose into, and
