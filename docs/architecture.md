# Malvinas — architecture (proposed)

Reflects the techniques favored in `docs/plans/` so far, not the full menu of
alternatives (e.g. GQA and key-value-reuse both solve the same problem —
this picks one). Nothing here is executed yet; this is the target shape.

## Full stack

```mermaid
flowchart TB
    TOK["Token IDs\n(SmolLM2 tokenizer, ~49K vocab)"] --> EMB["Token Embedding\n(d_model, tied with output head)"]
    EMB --> BLK1["Transformer Block × N\n(see block detail below)"]
    BLK1 --> FNORM["Final RMSNorm"]
    FNORM --> HEAD["Output head\n(tied weights → vocab logits)"]

    subgraph ratio["local:global ratio ≈ 5:1 (plan 08, Gemma 4)"]
    direction TB
    L1["Local layers: windowed attention,\nRoPE theta=10k"]
    G1["Global layers (1-in-5): full attention,\npartial-RoPE theta=1M, K reused as V (plan 06)"]
    end
    BLK1 -.-> ratio
```

## One Transformer block

```mermaid
flowchart TB
    X["x (B, T, d_model)"] --> N1["RMSNorm + QK-Norm (plan 04)"]
    N1 --> ATT["Attention\nlocal: windowed / global: full + key-as-value (plan 06)\nRoPE (dual config, plan 08)"]
    ATT --> ADD1["+ residual"]
    X --> ADD1

    ADD1 --> N2["RMSNorm"]
    N2 --> ROUTER["Router (top-k of 32 fine-grained experts, plan 05)"]
    ROUTER -->|top-4 selected| EXPERTS["Routed Experts (small, many)\nSiLU-gated MLP each"]
    N2 --> SHARED["Shared Expert\n(always active)"]
    EXPERTS --> COMBINE["weighted sum\n(scatter-add by routing weight)"]
    SHARED --> COMBINE
    COMBINE --> ADD2["+ residual"]
    ADD1 --> ADD2
    ADD2 --> OUT["block output"]

    ROUTER -.->|expert-load bias, no grad| BAL["Auxiliary-loss-free\nload balancing (plan 03)"]
    BAL -.-> ROUTER
```

## Notes

- **Attention**: 5-in-6 layers are local/windowed (cheap, short RoPE theta);
  the rest are global/full-attention with key-as-value reuse instead of a
  separate V projection, and a partial-rotary, high-theta RoPE for long-range
  reach (plan 08 + 06, both Gemma 4).
- **MoE FFN**: many small experts (fine-grained, plan 05) instead of few big
  ones, top-k routed, plus one always-on shared expert (DeepSeekMoE
  pattern) — router is kept balanced via a dynamic per-expert bias updated
  outside the gradient graph (plan 03, DeepSeek-V3), not an auxiliary loss
  term (plan 02 is the fallback if 03 turns out harder to get right in
  practice).
- **Not pictured**: GQA (plan 07) and MLA (plan 10) are alternatives to the
  key-value-reuse approach shown here, not additions to it — pick one path,
  not all three. Same for MTP (plan 09), Kimi Delta Attention (plan 11), and
  MoBA (plan 12) — later-stage options, not part of this first target shape.
