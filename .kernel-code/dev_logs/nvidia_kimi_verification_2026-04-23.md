# Kimi K2 as Critic Model — NVIDIA NIM Verification

**Date:** 2026-04-23
**Owner:** verify-kimi (Task #2)
**Claim under test:** "kimi-k2-instruct as the pivot/critic option for harder reasoning."
**Role:** Critic reads failing kernel attempts + profile data (compute_util, bandwidth_util, bottleneck_type) and diagnoses what to change. Needs strong reasoning more than raw code generation. Must be on NVIDIA NIM free tier (40 RPM, OpenAI-compatible).

---

## Section 1 — Kimi K2's reasoning specialization + architecture

The NIM catalog currently exposes several Moonshot variants: `moonshotai/kimi-k2-instruct` (July 2025 launch), `moonshotai/kimi-k2-thinking`, and `moonshotai/kimi-k2.5` (multimodal). Only the non-thinking `kimi-k2-instruct` is what the team proposed — an important distinction.

**Architecture (both variants share the base):**
- MoE, 1T total / 32B active parameters
- 384 experts, 8 selected per token, 61 layers
- MLA attention, 160K vocab
- `kimi-k2-instruct`: 128K context (per NIM reference)
- `kimi-k2-thinking`: 256K context, native INT4 QAT, designed for 200–300 sequential tool calls

**Training specialization:**
- `kimi-k2-instruct` is a **general-purpose agentic MoE** — its published strengths are tool use (94.7% tool-calling success), coding (SWE-bench), and agentic workflows. It is *not* a chain-of-thought reasoning specialist in the DeepSeek-R1 / o1 sense. It does not emit thinking tokens by default.
- `kimi-k2-thinking` is the reasoning-specialized sibling: it emits CoT, sustains long-horizon agentic reasoning, and holds the current public SOTA on HLE-with-tools (44.9%) among open-weight models.

**Implication for the claim as written:** "kimi-k2-instruct" (the exact model name in the claim) is the *weaker* of the two Moonshot options for a reasoning-heavy critic. If the team wants Kimi-family reasoning, the correct pick is `kimi-k2-thinking`, not `kimi-k2-instruct`.

Sources:
- [Kimi-K2-Instruct on Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- [Kimi-K2-Thinking on Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- [kimi-k2-instruct NIM reference](https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k2-instruct)
- [Kimi K2 technical deep dive (IntuitionLabs)](https://intuitionlabs.ai/articles/kimi-k2-technical-deep-dive)

---

## Section 2 — Reasoning benchmark comparison

Numbers are best-available published figures; "no tools" / "w/ tools" is flagged where the delta matters. Candidates that are **not** on NIM are noted and excluded from the verdict.

| Model                          | On NIM | GPQA        | AIME 2025       | HLE (no tools) | HLE (w/ tools) | MMLU-Pro | LiveCodeBench |
|--------------------------------|--------|-------------|-----------------|----------------|----------------|----------|---------------|
| moonshotai/kimi-k2-instruct    | Yes    | ~75.1       | ~49.5           | n/r            | n/r            | 81.1     | 53.7          |
| moonshotai/kimi-k2-thinking    | Yes    | **84.5**    | 94.5 / 99.1 (py)| **23.9**       | **44.9**       | 84.6     | **83.1**      |
| deepseek-ai/deepseek-v3.2 (think) | Yes | 82.4        | 93.1            | **25.1**       | n/r            | 85.0     | 83.3          |
| openai/gpt-oss-120b (high)     | Yes    | 80.9 (w/tools) | 97.9 (w/tools) | n/r           | n/r            | **90.0** | high          |
| minimaxai/minimax-m2 (230B)    | Yes    | ~77–85 (M2.5 cited)| 86.3 (M2.5)| n/r          | n/r            | ~78–82   | ~75           |
| thudm/glm-4.5 / glm-5          | Yes    | 79.1 (4.5)  | n/r             | n/r            | n/r            | 84.6 (4.5)| n/r          |

**Observations:**
1. `kimi-k2-instruct` (the literal claim) is the **lowest-scoring reasoning option** among the NIM candidates — it underperforms `kimi-k2-thinking`, `deepseek-v3.2 (thinking)`, and `gpt-oss-120b` on every reasoning axis.
2. Among "pure reasoning" picks, `kimi-k2-thinking` has the strongest HLE-with-tools number (44.9% — public SOTA for open weights), with `deepseek-v3.2` essentially tied on HLE-no-tools (25.1 vs 23.9).
3. `gpt-oss-120b` wins MMLU-Pro (90.0) and AIME-with-tools (97.9) by a clear margin but has a much smaller active param count (5.1B active) and weaker coding/agentic track record.
4. GLM and MiniMax are mid-pack; no published numbers put them ahead of K2-thinking or DeepSeek-V3.2-thinking on the reasoning axis.

Sources:
- [DeepSeek V3.2 technical report (arxiv)](https://arxiv.org/pdf/2512.02556)
- [DeepSeek V3.2 (Reasoning) vs Kimi K2 Thinking — Artificial Analysis](https://artificialanalysis.ai/models/comparisons/deepseek-v3-2-reasoning-vs-kimi-k2-thinking)
- [GPT-OSS benchmarks vs open-source competitors — Clarifai](https://www.clarifai.com/blog/openai-gpt-oss-benchmarks-how-it-compares-to-glm-4.5-qwen3-deepseek-and-kimi-k2)
- [Kimi K2.5 vs DeepSeek V3.2 — docsbot.ai](https://docsbot.ai/models/compare/deepseek-v3-2/kimi-k2-5)

---

## Section 3 — Coding-critique / diagnostic evidence

No public benchmark specifically measures "diagnose a broken kernel from profile data." The closest signal is SWE-bench Verified (bug-localization-heavy) and SciCode (reasoning over scientific/technical code).

| Model                     | SWE-bench Verified | SciCode (w/ think) | Notes                                                 |
|---------------------------|--------------------|--------------------|-------------------------------------------------------|
| kimi-k2-instruct          | 65.8               | ~39                | strong at single-patch generation; less critique depth|
| kimi-k2-thinking          | **71.3**           | ~49                | holds narrative across 200+ debug steps (Long2Short)  |
| deepseek-v3.2 (thinking)  | 67.8               | ~45                | plans before acting, low hallucinated tool calls      |
| gpt-oss-120b              | ~62                | n/r                | math/competition skew, weaker at sprawling codebases  |
| glm-4.5                   | ~60                | n/r                | mid-pack                                              |
| minimax-m2                | ~58                | n/r                | mid-pack                                              |

The qualitative analyses consistently describe the same split:
- **DeepSeek V3.2 (thinking):** pre-computes a reasoning step, then acts — good at precise diagnosis of a single fault, produces clean *why this is wrong* explanations before a patch.
- **Kimi K2 Thinking:** maintains a running narrative across many steps — better when the critic must cross-reference prior attempts and profile history (our case has a "prior_attempts" list in the critic prompt, so this matters).

For our specific critic role — ingest (kernel source + ref + profile + prior attempts) and produce a diagnosis — both `kimi-k2-thinking` and `deepseek-v3.2 (thinking)` are plausible. `kimi-k2-instruct` (non-thinking) is not: it does not expose a thinking trace, and the thinking variants clearly outperform it on every diagnostic-adjacent metric.

Sources:
- [DeepSeek V3.2 vs Kimi K2 Thinking (CanopyWave)](https://canopywave.com/blog/deepseek-v32-vs-kimi-k2-thinking)
- [DeepSeek V3.1 vs Kimi K2 for coding — Novita](https://medium.com/@marketing_novita.ai/deepseek-v3-1-vs-kimi-k2-which-model-should-you-use-for-coding-b817f763c7ad)
- [Kimi K2 vs DeepSeek — Clarifai](https://www.clarifai.com/blog/kimi-k2-vs-deepseek-v3/r1)

---

## Section 4 — Long-context performance

The critic prompt in our pipeline can balloon: full kernel source (hundreds of lines), reference source, profile JSON, plus 2–5 prior_attempts. Realistic ceiling ~20–40K tokens.

- `kimi-k2-instruct`: 128K nominal. Independent LongCodeEdit evaluation reports effective context of **<32K** on complex code-editing — a significant gap from the marketing number.
- `kimi-k2-thinking`: 256K nominal, INT4-QAT-trained end-to-end for long multi-step use; designed for 200–300 sequential tool calls without degradation — the only candidate with published long-horizon stability claims.
- `deepseek-v3.2`: 128K nominal with DSA (DeepSeek Sparse Attention). The V3.2 paper focuses on long-context efficiency gains; no independent needle-in-haystack number was surfaced in my search window, but the architecture is explicitly built for long contexts.
- `gpt-oss-120b`: 128K nominal; weakest of the four on published long-context tasks.

At the realistic 20–40K critic-prompt size, all four should perform fine. At the tail (repeated critic calls with accumulating history), `kimi-k2-thinking` and `deepseek-v3.2` are the safest picks.

Sources:
- [Evaluating Long Context Reasoning (nrehiew blog)](https://nrehiew.github.io/blog/long_context/)
- [Kimi K2 long-context effective window — SmythOS](https://smythos.com/developers/ai-models/kimi-k2-is-here-is-this-the-open-source-ai-agent-weve-been-waiting-for/)

---

## Section 5 — NIM-specific observations

- **Availability (confirmed):** `moonshotai/kimi-k2-instruct`, `moonshotai/kimi-k2-thinking`, `moonshotai/kimi-k2.5`, `openai/gpt-oss-120b` all have dedicated NIM reference pages and live endpoints on `https://integrate.api.nvidia.com/v1/chat/completions`. DeepSeek-V3/V3.2 and GLM-family are in the broader catalog.
- **Free tier:** 1,000 credits on signup, **40 RPM** global rate limit, OpenAI-compatible API. Larger models (DeepSeek-R1 671B, GLM-5 744B) consume more credits per request, so a K2-thinking vs DeepSeek-V3.2 critic at comparable call volume will burn credits at similar rates — the free tier credit count, not model choice, will be the practical constraint.
- **Context on NIM:** NIM docs advertise 128K for `kimi-k2-instruct`. The 256K Kimi-K2-Thinking context in Moonshot's own docs may or may not be fully exposed through the NIM proxy — this should be verified with a quick curl before committing.
- **Throughput / latency:** no independent NIM-vs-Moonshot-first-party benchmarks surfaced in my search window. Reasoning models (thinking variants) can be slow because they emit long thinking traces before the final answer. At 40 RPM this is manageable; at higher volumes it would matter.
- **Known quirk:** community third-party re-evaluations have flagged that Moonshot's self-reported HLE numbers are overstated (re-run ~29% vs reported 44.9%). The relative ranking vs DeepSeek/GPT-OSS still holds, but treat the absolute HLE lead as weaker than the marketing number suggests.

Sources:
- [NVIDIA NIM free-tier guide 2026](https://awesomeagents.ai/tools/free-ai-inference-providers-2026/)
- [NVIDIA NIM API Explained — decodethefuture.org](https://decodethefuture.org/en/nvidia-nim-api-explained/)
- [Build with Kimi K2.5 via NVIDIA endpoints](https://developer.nvidia.com/blog/build-with-kimi-k2-5-multimodal-vlm-using-nvidia-gpu-accelerated-endpoints/)
- [HLE score inflation analysis — MindStudio](https://www.mindstudio.ai/blog/humanities-last-exam-benchmark-score-inflation-explained)

---

## Verdict

**PARTIALLY VERIFIED — use `moonshotai/kimi-k2-thinking`, not `moonshotai/kimi-k2-instruct`.**

**Why partial:**
- The *spirit* of the claim (Kimi-family as the reasoning-heavy critic pivot) is defensible: Kimi-K2-Thinking leads the open-weight NIM catalog on HLE-with-tools, matches DeepSeek-V3.2-thinking on GPQA/AIME/MMLU-Pro while beating it on long-horizon agentic coherence (SWE-bench Verified 71.3 vs 67.8, SciCode 49 vs 45), and its 256K context with long-run stability is the best fit for a critic that may see growing prior-attempt history.
- The *letter* of the claim is wrong: `moonshotai/kimi-k2-instruct` is the **non-thinking** variant. It lacks reasoning traces, scores lower on every reasoning benchmark than its thinking sibling, and underperforms DeepSeek-V3.2-thinking on GPQA/AIME/MMLU-Pro. Picking it as the "harder reasoning" critic would be a downgrade, not an upgrade, vs the DeepSeek default.

**Concrete recommendation:**
1. **Primary critic: `moonshotai/kimi-k2-thinking`** — best open-weight reasoner on NIM, narrative-coherence wins for iterative kernel debugging, long context for accumulating prior_attempts.
2. **Close second / fallback: `deepseek-ai/deepseek-v3.2` (thinking mode)** — nearly tied on raw reasoning numbers, stronger "plan-before-act" discipline (fewer hallucinated tool calls), and if it's already the generator, running the critic on the same provider simplifies pipeline tooling.
3. **Reject: `moonshotai/kimi-k2-instruct`** — the literal claim — as the reasoning-critic pick. Keep it only if the team wants a non-thinking Kimi for agentic/tool-calling roles elsewhere.
4. **Reject for critic role: `gpt-oss-120b`** (great at math competitions but weaker at multi-file code diagnosis), `minimax-m2`, `glm-4.5/5` (mid-pack with no reasoning-specialist edge over K2-thinking or DeepSeek-V3.2-thinking).

**One verification TODO before committing:** curl `moonshotai/kimi-k2-thinking` on NIM with a ~40K-token critic-shaped prompt to confirm (a) context window actually honored via the NIM proxy and (b) end-to-end latency is acceptable under the 40 RPM budget. If the thinking trace is too slow, fall back to DeepSeek-V3.2-thinking.
