# Selected Path v1

Status: DW04 decision (owner, 2026-10-13). One primary implementation path for P04–P05.

## Decision

Primary path: **Qwen2.5-Coder 7B Instruct Q4_K_M** (dense `qwen2`), resident memory policy, contexts 2048
and 4096.

Rationale: best balance of coding focus, memory and licensing. Peak RSS 7504 MiB is inside the 12 GB
budget. Decode 2.94 tok/s misses the ≥3 tok/s threshold by 2% and TTFT at 2K is 92 s against ≤60 s; both
are raised by optimization, not by changing the artifact. The artifact is pinned in
[the comparison manifest](comparison-manifest-v1.md); the measurements are in
[the comparison report](benchmarks/comparison-2026-10-13/README.md).

External comparison baseline: upstream llama.cpp `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444`.

Not selected: APEX-I-NanoPlus (exceeds the memory budget without bounded expert loading) and Bonsai PTQ1_0
(ternary codec ~300x too slow in prefill). Both remain scouting evidence.

## Required capabilities

| Concern | Required | State |
|---|---|---|
| Architecture | dense `qwen2`: QKV bias, GQA, half-split RoPE, RMSNorm, SwiGLU | present (`DenseTransformerAdapter`, `TransformerOps`) |
| Tokenizer | gpt2 BPE, `qwen2` pre-tokenizer, ChatML template | present (`BpeTokenizer`, `ChatMlTemplate`) |
| Codecs | Q4_K (169 tensors), Q6_K (29 tensors), F32 (141) | **missing** Q4_K and Q6_K |
| Descriptor | metadata-driven shape/plan, explicit capability check | present (`ModelDefinition`, `ModelPlanner`) |
| Session | KV state, context limit, lifetime | present (`DenseSession`) |
| Memory policy | resident weights, KV and scratch inside 12 GB | present (`GgufWeightStore`, `MappedWeights`) |
| Backend | CPU; Vulkan only after a fixed benefit criterion | CPU present |

The only missing capability is the Q4_K and Q6_K codecs, with their MatVec and their AVX2 kernels.

## Thresholds to reach by optimization

| Metric | Now | Target |
|---|---:|---:|
| TTFT at 2K | 92 s | ≤ 60 s |
| Decode at 2K | 2.94 tok/s | ≥ 3 tok/s |

Memory (7504 MiB) and quality (to be measured against the frozen C# corpus) are the remaining gates.

## Out of scope for this path

MoE routing, expert cache, hybrid/recurrent attention, ternary codecs (BitNet I2_S, Prism PTQ1_0/PQ2_0),
IQ codecs, vision projectors and the 262144-token context.
