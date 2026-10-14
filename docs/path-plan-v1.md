# Selected Path Plan v1

Status: DW05. Restricts the capability matrix, the runtime contracts and the task breakdown to the path
selected in [selected-path-v1.md](selected-path-v1.md). No implementation here.

## Restricted capability matrix

| Concern | Required by the path | State |
|---|---|---|
| Architecture | dense `qwen2` | present |
| Tokenizer | gpt2 BPE, `qwen2` pre-tokenizer, ChatML | present |
| Operator | QKV bias, GQA, half-split RoPE, RMSNorm, SwiGLU | present |
| Codec Q4_K | 169 tensors | **missing** |
| Codec Q6_K | 29 tensors (`output.weight`, part of `ffn_down`) | **missing** |
| Format F32 | 141 norm/bias vectors | present |
| Session / resource | KV, context, lifetime, cancellation | present |
| Memory | resident weights, KV, scratch ≤ 12 GB | present |
| Backend | CPU | present |

Everything else in [the capability matrix](capability-matrix-v1.md) — MoE, hybrid attention, Q3_K, the IQ
types, BF16, ternary codecs, vision — is out of scope for this path.

## Session and resource contracts (narrowed)

- Session: `DenseSession` over `qwen2`, contexts 2048 and 4096. KV is F32, dimensioned from the shape.
- Resource limits: 12 GB runtime ceiling; cancellation latency ≤ 2 s; no sustained swap.
- Memory plan: weights 4.68 GB (compressed, resident) + KV + workspace, computed by `ModelPlanner`.
- No streaming, no expert cache, no SSD KV. Resident policy only, so DW08 is not applicable.
- Compute limits: 8 inference threads by default; the thread count is a configuration input.

## Codec specification task

The Q4_K and Q6_K block layouts are pinned at DW06.1 from the ggml reference at the revision recorded in
[gguf-subset-v1.md](gguf-subset-v1.md) (`ggml-common.h`, `ggml-quants.c`). The oracle is an independent FP64
reconstruction, never the decoder just written. Tolerances are set per operator before optimizing.

## Subtasks (DW06)

Each subtask has a dependency, an oracle and a verification criterion. Execute one at a time.

| Id | Subtask | Depends on | Oracle | Verification |
|---|---|---|---|---|
| DW06.1 | Q4_K scalar decoder and dot | DW05 | independent FP64 decode of the same block bytes | known blocks, zeros, extremes, invalid sizes; reconstructed weights within tolerance (done) |
| DW06.2 | Q6_K scalar decoder and dot | DW06.1 | independent FP64 decode | corpus of blocks; range checks; MatVec against FP64 (done) |
| DW06.3 | Register Q4_K/Q6_K in the format dispatch | DW06.2 | n/a | `GgufTypes`, `BlockQuantization`, `GgufQuantizedMatrix` accept the types; a Q4_K_M file passes the capability check (done) |
| DW06.4 | Q4_K_M end-to-end parity | DW06.3 | independent forward reference on the same artifact | every tensor type inventoried and decodable with no silent fallback; logits and greedy tokens against the reference (done: real-tensor byte parity + end-to-end run) |
| DW06.5 | Q4_K/Q6_K AVX2 kernels | DW06.4 | the scalar kernels | numerical parity with scalar; benchmark on the same matrices; keep experimental without a measured advantage (done: 6.4x on the 7B artifact, default path) |
| DW06.6 | Prefill optimization to meet TTFT ≤ 60 s at 2K | DW06.5 | the reference forward | batched prefill; TTFT at 2K ≤ 60 s on the target; identical logits (structure done, batched prefill bit-identical to sequential append; prefill 3.65 tok/s measures ~560 s at 2K: criterion **not met**, and the remaining gap is per-call cost and scheduling rather than the codec) |
| DW06.7 | Decode optimization to meet ≥ 3 tok/s at 2K | DW06.5 | the reference forward | decode at 2K ≥ 3 tok/s; identical greedy tokens (kernel restructured, batched step on every projection: 0.65 tok/s against the 3 tok/s threshold, **not met**; the earlier 1.32x claim came from single-run measurements and is withdrawn, see measurement-method.md. The gap is per-call cost and scheduling, not arithmetic: a token makes 197 projections and ~1576 task spawns) |
| DW06.8 | Resource limits and cancellation | DW06.4 | n/a | memory and context bounds enforced; cancellation ≤ 2 s; no sustained swap |

## Ordering rule

DW06.6 and DW06.7 are ordered after the AVX2 kernels (DW06.5) because the measured bottleneck is the
quantized MatVec and the prefill loop, not the codec itself. DW08 (cache/prefetch) is not applicable: the
path is resident. DW09 (Vulkan) starts only after a fixed end-to-end benefit criterion.

## Quality gate

DW06.8 and the delivery verification run the frozen C# corpus. The corpus and the concurrent workload
proposals are frozen by the owner before DW06.8. The 80% C# pass rate and the responsiveness threshold
apply to the delivered path.
