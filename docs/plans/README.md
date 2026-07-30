# MoE trainer — improvement plans

Ideas to graft onto `train_moe.py`, borrowed from DeepSeek (V2/V3, DeepSeekMoE),
Moonshot/Kimi (Kimi Linear, MoBA), and Gemma 4 (Google DeepMind). Ordered
easy → hard. Each file is a standalone spec: what it is, why it helps, how it
plugs into the current script's variable names. None are implemented yet —
pick one, we scope it into an actual change.

0. [Training process, data, and scaling](00-training-process-and-data.md) — target size, tokenizer, data volume/sources, 4096→128K context (YaRN), pretrain → context-extend → SFT+tool-use path. Read this first; everything else assumes it.
1. [Real BPE tokenizer (via Gigatoken)](01-tokenizer-bpe-gigatoken.md)
2. [Load-balancing auxiliary loss](02-load-balancing-aux-loss.md)
3. [Auxiliary-loss-free load balancing (DeepSeek-V3)](03-auxiliary-loss-free-balancing.md)
4. [QK-Norm](04-qk-norm.md)
5. [Fine-grained experts + shared-expert isolation (DeepSeekMoE)](05-fine-grained-experts.md)
6. [Key-as-value reuse in global attention (Gemma 4)](06-key-value-reuse.md)
7. [Grouped Query Attention (GQA)](07-grouped-query-attention.md)
8. [Local/global sliding-window attention + dual RoPE (Gemma 4)](08-local-global-sliding-window.md)
9. [Multi-Token Prediction (DeepSeek-V3 / Gemma 4)](09-multi-token-prediction.md)
10. [Multi-head Latent Attention (DeepSeek-V2/V3)](10-multi-head-latent-attention.md)
11. [Kimi Delta Attention / linear-attention hybrid (Moonshot)](11-kimi-delta-attention.md)
12. [Mixture of Block Attention (MoBA, Moonshot)](12-moba.md)
