# Reference Comparison 2026-10-13

Status: DW04 (P02). Candidate measurements in their reference runtimes. No Satori runtime is involved.

## Environment

| | |
|---|---|
| Host | Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux |
| Role | Intel comparison host, not the primary Geekom A5 |
| Qwen / APEX runtime | upstream llama.cpp `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444` (0.6.0-dev, ggml 0.26.0) |
| Bonsai runtime | Prism llama.cpp fork `e8fd209c975c80aa7fb36a8423571ff3b1b7e955` (ggml 0.21.0) |
| Threads | 8 |
| KV type | f16 |

## Throughput

`llama-bench -p 2048 -n 128 -r 1 -t 8`, batch 2048. Prefill processes 2048 tokens; decode generates 128.

| Candidate | Prefill tok/s | Decode tok/s | TTFT at 2K (s) |
|---|---:|---:|---:|
| Qwen2.5-Coder 7B Q4_K_M | 22.16 | 2.94 | 92 |
| APEX-I-NanoPlus (35B-A3B) | 39.70 | 4.88 | 52 |
| Bonsai PTQ1_0 (27B) | 1.37 | 0.99 | 1490 |

TTFT is derived as 2048 / prefill rate, not measured directly. The APEX prefill rate exceeds the 7B
because the mixture activates about 3B parameters per token, while Bonsai is bound by its ternary codec.

## Memory

`run-memory.sh` wraps a one-token `llama-bench` run and reports the child peak RSS (KiB), which includes
the touched mapped model pages. Context is minimal, so KV is small; a 2K context adds the model's KV.

| Candidate | File (B) | Peak RSS (MiB) | Load + 1 token (s) |
|---|---:|---:|---:|
| Qwen2.5-Coder 7B Q4_K_M | 4683073536 | 7504 | 16.6 |
| APEX-I-NanoPlus | 12554424672 | 12705 | 14.4 |
| Bonsai PTQ1_0 | 5946648928 | 5955 | 8.1 |

Bonsai peak RSS is close to its file size: the ternary weights are used without permanent expansion.
Qwen peak RSS exceeds its file: the runtime expands part of the weights and holds KV and scratch.

## Verdict against the fixed thresholds

| Criterion | Qwen 7B Q4_K_M | APEX-I-NanoPlus | Bonsai PTQ1_0 |
|---|---|---|---|
| TTFT 2K ≤ 60 s | fail (92) | pass (52) | fail (1490) |
| Decode ≥ 3 tok/s | fail (2.94) | pass (4.88) | fail (0.99) |
| Peak RAM < 12 GB | pass (7.5) | fail (12.7) | pass (6.0) |

No candidate passes every threshold on this host. APEX is the closest on latency but exceeds the memory
budget by about 0.7 GB at minimal context; a 2K and 4K context increases it further.

## Reproduce

```sh
# runtimes built from source; see docs/comparison-manifest-v1.md for the flags
CD=.temp/deepseek/dw02
LD_LIBRARY_PATH=$CD/llama.cpp/build/bin $CD/llama.cpp/build/bin/llama-bench \
    -m $CD/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf -p 2048 -n 128 -r 1 -t 8
bash $CD/run-memory.sh
```

## Limits

- One host (Intel comparison). The primary Geekom A5 is not measured.
- One repetition; the prefill rate on this CPU is sensitive to load, so treat the figures as a first
  order comparison, not a stable benchmark.
- Peak RSS is a peak working set, not a resident-set guarantee.
- TTFT is derived from the prefill rate.
- Memory is at minimal context; the 2K and 4K figures include additional KV.
- No quality measurement in this report; the C# corpus is a separate gate.
