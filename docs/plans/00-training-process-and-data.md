# 0. Training process, data, and scaling — the plan everything else sits on

This is the foundational doc: target model size, tokenizer, how much data that
size actually needs, where the data comes from, and the pretrain → SFT path
to a 4096-context conversational model. Plans 01-12 are architecture
techniques to graft on; this doc decides what we're training *for* before any
of them matter.

## 1. Hardware target: single high-VRAM GPU (B200-class)

Confirmed specs: an NVIDIA B200 has **192GB HBM3e** at 8.0 TB/s — per NVIDIA's
own framing, that's enough to run a 70B-parameter model at FP16 with headroom
for KV cache. Against that, a 0.5-1B *total*-parameter MoE is small — even
with the usual mixed-precision AdamW rule of thumb (~16 bytes/param: bf16
weights + bf16 grads + fp32 optimizer moments + fp32 master weights), the
1.24B-total config below is only ~20GB for model+optimizer state. **VRAM is
not the constraint here** — a single B200 swallows this model whole with
massive headroom left for batch size and activations, even at `block_size=4096`
with full (non-windowed) attention.

That changes what plan 08 (local/global attention) and the other efficiency
plans (05 key-value reuse, 06 GQA) are *for* in this project: not "fit in
memory" (already fits easily), but **wall-clock training speed** — a single
GPU, even a B200, has finite FLOPs/s, and full O(n²) attention on every layer
at 4096 tokens across tens of billions of training tokens is the thing that
actually costs time. Spend the generous VRAM headroom on **larger batch
size** (better GPU utilization, more stable gradients) rather than on a
bigger model — there's no reason to go past ~1B total params just because
memory allows it; the data budget in section 4 is the real limiter on how big
a model is worth training on one GPU.

## 2. Target size: pick one of these two MoE configs

Using this script's own parameter-count structure (attention `4·d²` per
layer, each expert `3·d·expert_dim`, one always-on shared expert, tied
embedding/output):

| Target | d_model | n_layers | experts | top_k | expert_dim | vocab | **Total** | **Active** | active % |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B | 512 | 16 | 32 | 4 | 512 | 32,000 | **448M** | **96M** | 21% |
| 1B (deep) | 512 | 32 | 32 | 4 | 512 | 32,000 | **880M** | **176M** | 20% |
| 1B (wide) | 768 | 20 | 32 | 4 | 768 | 32,000 | **1.24B** | **249M** | 20% |

All three land near a ~20% active ratio (top-4-of-32), similar sparsity to
production MoEs like DeepSeekMoE/OLMoE. Widening (`d_model`) vs deepening
(`n_layers`) trades the usual things (depth = more sequential reasoning
steps, width = more per-token capacity) — no strong reason to prefer one for
this project; start with the 0.5B row, it's the cheaper one to iterate on.

## 3. Tokenizer: not GPT-2's vocab

The reason to skip literal GPT-2 (OpenAI's 2019 vocab, trained mostly on old
web text) is **compression quality and vocab-size-vs-parameter-budget**:

- A bigger, more modern vocab (Llama 3: 128k, Qwen2.5: ~152k, GPT-4o's
  `o200k_base`: 200k) compresses English/code/multilingual text into fewer
  tokens per sentence — genuinely better. But at these target sizes (0.5-1B
  *total*), a 128-200k vocab eats a disproportionate slice of the parameter
  budget in the embedding + output layers (even tied, `vocab_size · d_model`
  — at `d_model=512` a 152k vocab alone is 78M params, ~17% of the whole 0.5B
  budget for a component that does zero reasoning).

**Recommendation: reuse the SmolLM2 tokenizer** (`HuggingFaceTB/SmolLM2-135M`)
— 49,152 vocab, GPT-2-style BPE but retrained on a modern mix (70%
FineWeb-Edu, 15% Cosmopedia-v2, 8% OpenWebMath, 5% StarCoderData, 2%
StackOverflow). It's sized specifically for small models, trained on
broadly the same *kind* of data (quality-filtered web + math + code) we're
about to pretrain on (section 5) even though the exact corpus differs
(Dolma 3 there, not FineWeb-Edu) — close enough in composition that
compression stays reasonable. Implemented in `src/malvinas/tokenizer.py`
(plan 01).

## 4. How much data: scaling laws, applied to *active* parameters

Chinchilla's classic result (~20 tokens/parameter) is for dense,
compute-optimal training. It does not transfer directly to MoE. A relevant
finding (arXiv:2508.18672, "Optimal Sparsity of MoE LMs for Reasoning
Tasks"): at 10M active parameters, an MoE pretrained at the literal
Chinchilla-optimal token count (200M tokens = 20×) **underperforms** an
equivalent dense model — but with 20× more data than that (4B tokens, i.e.
~400 tokens per active parameter), the MoE **outperforms** dense. Real
production ratios go further still: OLMoE (1.3B active) trained on 5T tokens
≈ 3,800 tokens/active-param; DeepSeek-V3 (37B active) on 14.8T tokens ≈ 400
tokens/active-param.

Applied to our two configs:

| Target | Active | Chinchilla floor (20×) | MoE realistic min (400×) | Frontier-style (3,800×) |
|---|---|---|---|---|
| 0.5B | 96M | 1.9B tokens | **38B tokens** | 365B tokens |
| 1B (deep) | 176M | 3.5B tokens | **70B tokens** | 668B tokens |

The **Chinchilla floor is a known-bad target for MoE** per the finding above
— don't train to that number expecting it to work. The frontier-style column
is what SmolLM2 itself does (135M model on 2T tokens, 1.7B model on 11T
tokens) and is out of reach for a hobbyist run. **Target the "MoE realistic
min" column (tens of billions of tokens)** — enough to actually show the MoE
earning its complexity over a dense model of the same active size, without
requiring a datacenter.

This also settles the earlier question about TinyStories: it's ~2-3M short
stories, on the order of a few hundred million tokens total — nowhere near
the 38-70B needed here. TinyStories stays useful as the fast sanity-check
corpus for the *current* 2.24M-param toy script (does the code run, does loss
go down), but it is not the pretraining corpus for the 0.5-1B target.

## 5. Pretraining data: Dolma 3 Mix (150B), not FineWeb-Edu

**Updated recommendation: [`allenai/dolma3_mix-150B-1025`](https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025)**
— a better fit than FineWeb-Edu for this project, for a concrete reason: Ai2
built this specific 150B-token sample (104M documents) *for small-scale
model experiments* — it's explicitly the mix used for their "1Bx5C and
7Bx1B" configs, i.e. our exact regime, not something we'd have to guess a
subset size for ourselves. Composition is also richer than plain edu-web
text: 76.9% Common Crawl web, 12.6% academic PDFs (science), 7.06% GitHub
code (Stack-Edu), 2.60% math web pages (FineMath 3+), 0.82% arXiv, plus
Wikipedia — the code/math/academic slice matters later for the
function-calling and reasoning SFT stages (sections 9-10), since a pretrain
that's seen zero code/math makes those much harder to teach in SFT alone.
At 150B tokens it comfortably covers the ~38-70B "MoE realistic min" target
from section 4 with headroom for the YaRN context-extension pass (section 8)
too, without touching the full pool.

The full pool this was sampled from —
[`allenai/dolma3_mix-6T`](https://huggingface.co/datasets/allenai/dolma3_mix-6T)
(6T tokens, built for Olmo-3-1125-32B) — is available if 150B ever turns out
to be insufficient, but isn't needed for this target; using it would mean
going back to streaming/subsampling it ourselves, the exact complexity the
150B sample already avoids.

**One real caveat, stated plainly on the 6T dataset's own page**: this is
Common Crawl-derived web text, so it inherently contains some adult/NSFW
material by nature of broad web crawling — Ai2 ships quality-control
metadata specifically to filter this (`dolma2_qc` scores, `madlad` content
rules, dedup info). **Apply those filters when loading the data, don't skip
them** — they're there to be used, not optional metadata to ignore. This
applies to the 150B sample too, since it's drawn from the same pool.

Both are ODC-BY-1.0 licensed (same as the SFT-side datasets in sections 9-10)
— consistent licensing across the whole pipeline if we stay in the Ai2
ecosystem for pretraining data, separate from the SmolLM2 tokenizer/SmolTalk
(HuggingFace) side. Mixing organizations across tokenizer/pretrain/SFT is
fine technically — nothing here requires them to match — but worth knowing
this is a deliberate blend of two labs' data, not one coherent released recipe.

## 6. Context length: 4096, and why that forces plan 08's hand

The target context length is `block_size=4096` (before the section 8 YaRN
extension to 128K), not the tiny context of early toy experiments. Two
direct consequences:

- **Attention cost**: full causal attention is O(block_size²) per layer.
  At 4096 that's real (16.7M score entries per head per layer, times
  `n_layers`), and it's paid on *every* layer for *every* token during
  pretraining. This is exactly the problem [plan 08](08-local-global-sliding-window.md)
  (Gemma 4's local/global sliding-window split) solves — running most layers
  windowed and only a 1-in-5 minority at full 4096 context is the difference
  between this being affordable and not. Treat plan 08 as a prerequisite for
  4096-length training, not an optional later optimization.
- **Data packing**: Dolma 3 Mix documents vary in length; standard practice is
  to concatenate documents with an end-of-text separator token and slice into
  fixed `block_size` chunks ("packing"), rather than padding every short
  document up to 4096. Needs the packing step added wherever `train_x`/
  `train_y` are currently built.

## 7. SFT stage: SmolTalk

Pretraining alone (next-token prediction on web text) does not produce a
model that follows instructions or holds a conversation — it produces a model
that continues text. **Recommendation:
[`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)**
(or its smaller `smol-smoltalk` variant) — built specifically for fine-tuning
small models into conversational assistants, same lineage as the tokenizer
and pretraining corpus above.

Mechanically, SFT on this script's training loop needs:

- A chat template (turn markers for user/assistant, e.g. `<|user|>...
  <|assistant|>...`) added as special tokens or plain text markers the
  tokenizer already encodes.
- **Loss masking**: compute `criterion` only on assistant-turn tokens, not on
  the user's prompt tokens — set masked positions' targets to an ignored
  index (`nn.CrossEntropyLoss(ignore_index=-100)`, targets set to `-100`
  wherever the token belongs to a user turn).
- Same `block_size=4096`, same model — SFT reuses the pretrained weights as
  initialization, it is not a separate architecture.

## 8. 128K context: RoPE extension (YaRN), not a tokenizer problem

Reaching 128K is **not** a tokenizer change — the tokenizer already produces
however many tokens a 128K-token document needs, same as it does at 4096.
The real levers:

- **RoPE has to be extended, not just widened.** A model trained at
  `block_size=4096` with `rope_theta=10000` (local layers) hasn't seen
  position indices past 4096 during training — feeding it position 100,000
  produces frequencies the model never learned to interpret. Naively raising
  `block_size` doesn't fix this by itself.
- **[YaRN](https://arxiv.org/abs/2309.00071)** is the concrete, verified-efficient method: it
  interpolates RoPE's frequency bands so a model pretrained at 4K generalizes
  to 128K after a short dedicated fine-tuning pass — measured at **~384
  A100-GPU-hours** to take a 4K model to 128K, ~16x cheaper than NTK-aware
  scaling for a similar target (~6,400 GPU-hours for ~100K) and better at the
  extreme end of the range. This becomes a **third stage**, after pretraining
  and before (or interleaved with) SFT: continue training on long documents
  with YaRN-rescaled RoPE, at a much smaller token budget than the original
  pretrain.
- **Attention cost compounds hard at 128K.** O(n²) at 128K is ~1,000x the
  score-matrix cost at 4096 (128K² vs 4096²). Plan 08's local/global 5:1 split
  is exactly how Gemma 4 reaches its 256K context in production — the *global*
  layers (1-in-5) are the ones that pay the O(n²) cost and the ones YaRN needs
  to target; local/windowed layers stay cheap regardless of total context. If
  128K is a firm target, this also makes plan 11 (Kimi Delta Attention) worth
  a second look — it's built specifically for million-token-class context
  where even a 1-in-5 full-attention layer gets expensive, not just a "harder
  version" of plan 08.
- **Needs actual long-document training data** for the extension pass — most
  most Dolma 3 Mix documents are far shorter than 128K tokens. Concatenating related
  documents/chapters or sourcing long-form text (books, long code repos) is
  necessary; the model can't learn to use 128K of context from a training set
  where no example is longer than a few thousand tokens.

## 9. Function calling / tools: an SFT-data-and-format problem, tokenizer plays a small supporting role

This is the piece that's genuinely *not* mainly a tokenizer question — the
tokenizer's job here is small (reserve a handful of special tokens to mark
tool-call boundaries), while the real work is a **third training-data
mixture** and a **consistent conversation format**.

- **Reference format** (Llama 3.1, verified real tokens): `<|python_tag|>`
  marks a tool invocation in the model's output; `<|eom_id|>` ends a message
  when the model expects to call a tool next (vs. `<|eot_id|>` for a normal
  turn end). Equivalent to adding 2-3 new special tokens to the SmolLM2
  tokenizer and resizing the tied embedding/output head by that many rows
  (copy existing weights, randomly init the new rows) — a small, mechanical
  change, not a new tokenizer. Implemented: `Tokenizer.add_special_tokens`
  and `MalvinasModel.resize_token_embeddings`.
- **The actual content is training data**: conversations where the system
  turn lists available tools (JSON-schema-style function signatures), the
  model emits a structured tool call instead of a direct answer, a tool
  response gets inserted into the context, and the model produces a final
  answer using that result. Real datasets to draw from:
  [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
  (60k examples; Salesforce also released `xLAM-1b-fc-r` — a **1B-scale**
  function-calling model benchmarked on the Berkeley Function-Calling
  Leaderboard, good evidence this is achievable at our target size),
  [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1),
  and [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
  (52k, deliberately balanced with no-call/multi-call/no-tools-offered
  examples so the model doesn't learn to call a tool on every turn).
- Mix this into the SFT stage (section 7) rather than treating it as a fourth
  separate stage — same loss-masking mechanism (mask everything except the
  model's own turns, including its tool-call tokens), just a richer data
  mixture.

## 9b. Alternative ordering: SFT before *and* after YaRN extension

A reasonable variant of the pipeline below: **Pretrain@4096 → SFT@4096
(general chat only) → YaRN-extend to 128K → SFT@128K again (SmolTalk +
function-calling)**. This buys an early, cheap checkpoint to sanity-check
conversational quality *before* spending the extension pass — useful if
you'd rather catch a broken pretrain early than discover it after context
extension.

One correction to how this gets framed, though: the second SFT pass here
isn't an optional "if I want to extend, then also re-tune" step — **it's
required, not optional**, as soon as you extend context after the first SFT.
YaRN's extension pass is continued pretraining on raw long-form text (no
chat formatting, no assistant/user turns) — running plain next-token
prediction on raw text after an instruction-tuned checkpoint measurably
erodes the chat behavior that first SFT pass just taught it (the standard
"alignment tax" / regression-from-continued-pretraining effect). So if you
extend after SFT, the first SFT's checkpoint is a validation artifact, not
the final one — the model *needs* the second SFT pass to recover and
recalibrate, it's load-bearing, not a bonus.

Given that, the cleanest version of this variant: keep function-calling data
out of the *first* SFT pass entirely (general conversation only, at 4096, as
a pure validation checkpoint) and put all of SmolTalk + the function-calling
mixture in the second, final SFT pass at 128K — no reason to teach tool use
twice.

## 10. Optional: reasoning traces for "thinking mode" (Dolci-Think-SFT)

[`allenai/Dolci-Think-SFT-32B`](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B)
(Ai2) — real, well-documented dataset: 2.25M examples / 36GB, chat-formatted
with explicit `<think>...</think>` reasoning traces (math, code, instruction-
following, safety, multilingual), sourced partly from DeepSeek R1 distillation,
ODC-BY-1.0 licensed. This is the "thinking mode" ingredient (same feature
Gemma 4 ships, section on Gemma 4 tricks elsewhere in these plans).

Two caveats before mixing it in:

- The "32B" names the model it was built for (Olmo-3-32B-Think; a 7B variant
  also exists) — **there is no validated small-model precedent for this
  dataset**, unlike SmolTalk (proven down to 135M) or xLAM (proven at 1B).
  Small models plausibly struggle to make good use of long chain-of-thought
  the way a 32B model does — real risk, not just theoretical, that this
  produces verbose/incoherent reasoning rather than useful reasoning at our
  scale.
- No function-calling content — doesn't replace section 9's tool-use data.

**Recommendation**: treat as an *optional, experimental* addition to the
final SFT mixture, not a core ingredient — take a curated subset (math/code,
where explicit reasoning helps most) rather than the full 2.25M/36GB, which
is oversized and too generalist (safety/jailbreak/multilingual content) for
what a ~0.5B model can absorb. Final SFT mixture: **SmolTalk (section 7) +
function-calling (section 9) + a small curated Dolci-Think subset
(this section, experimental)**.

## 11. Optional 4th stage: DPO (Dolci-Instruct-DPO)

[`allenai/Dolci-Instruct-DPO`](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO)
— same Ai2/Olmo-3 pipeline family as section 10 (that dataset's own page
mentions the model line "progresses through DPO and RLVR stages," this is
that stage). 260K chosen/rejected preference pairs (125K via a "Delta
Learning" heuristic, 125K from a GPT-judge pipeline, 10K multiturn) across
code/science/translation/QA, ODC-BY licensed, built for Olmo 3 Instruct 7B
(same sub-1B precedent gap as section 10 — reused up to 32B, not validated
smaller).

DPO is a post-SFT preference-tuning step, not a place to teach new
capabilities — it doesn't replace SFT's role (that's still where
conversation, tool-use, and reasoning get taught). What it does: given two
responses to the same prompt, nudge the model toward the preferred one
directly, without needing a full RL/reward-model setup. Typically polishes
things SFT alone leaves rough — verbosity, tone, response quality — rather
than adding behavior.

Optional 4th stage, after SFT is solid: **Pretrain → YaRN-extend to 128K →
SFT (SmolTalk + tools + reasoning) → DPO (Dolci-Instruct-DPO)**.

## Summary: the pipeline

1. **Pretrain** the 0.5B config (section 2) on the Dolma 3 Mix 150B dataset
   (section 5), tokenized with the SmolLM2 tokenizer (section 3),
   at `block_size=4096` with plan 08's local/global attention (section 6).
   This is a next-token-prediction run using `train.py`'s `train_step`, just
   at real scale — a serious single-GPU job (one B200-class card, per
   section 1), not a laptop run.
2. **Extend context to 128K via YaRN** (section 8) — a short dedicated
   fine-tuning pass on long-form documents, rescaling RoPE rather than
   training long-context from scratch.
3. **SFT** the resulting checkpoint on SmolTalk + a function-calling mixture
   + optionally a curated reasoning subset (sections 7, 9, 10), same 128K
   context, with assistant-only loss masking (including the model's own
   tool-call tokens), to get a model that can hold a conversation *and* call
   tools instead of just continuing text.
4. **(Optional) DPO** on Dolci-Instruct-DPO (section 11) to polish response
   quality/preference beyond what SFT alone gives.

Plans 02-07, 09-12 (load balancing, QK-norm, fine-grained experts, GQA, key-
value reuse, MTP, MLA, Kimi Delta Attention, MoBA) are architecture
improvements layered into step 1's model — independent of this doc, but
sized/justified against the 0.5-1B target established here, not the 2.24M
toy config.
