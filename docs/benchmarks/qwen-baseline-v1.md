# Resident CPU Baseline (Coding)

Status: DS11. First measured baseline of the resident dense CPU path on coding artifacts. Report only; no
optimization is claimed. Raw reports: `docs/benchmarks/qwen-2026-10-13/`.

## Environment

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H |
| Logical processors | 22 |
| RAM | 32 GiB |
| OS | Arch Linux, x64 |
| Runtime | .NET 10.0.12 |
| SIMD | AVX2 (scalar fallback available) |
| Corpus | `python tools/download_qwen_models.py` (Q8_0, pinned SHA-256) |

## Method

`dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --qwen-baseline <model.gguf> <context>
[--prefill N] [--decode N] [--quality N]`.

- Startup is separated into descriptor validation, container open and session construction.
- Prefill forwards a prompt of `N` tokens; TTFT is the same measurement to the first token.
- Decode appends greedily and reports per-token latency.
- Peak memory is the process peak working set, which includes the memory-mapped file pages that were
  touched, not only managed allocations.
- Quality is a small greedy coding smoke (prompt completion), not the DS04 acceptance pass rate.

## Results (3 data points)

| Model | Context | Startup ms | Prefill tok/s | TTFT ms | Decode tok/s | Decode ms/tok | Peak WS MiB | Plan total MiB | ≤16 GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| Qwen2.5-Coder 1.5B Q8_0 | 2048 | 160 | 2.34 | 27308 | 2.27 | 443.7 | 2032 | 1913 | yes |
| Qwen2.5-Coder 1.5B Q8_0 | 4096 | 202 | 2.19 | 29179 | 2.24 | 443.9 | 2032 | 2025 | yes |
| Qwen2.5-Coder 7B Q8_0 | 2048 | 147 | 0.55 | 57982 | 0.53 | 1851.3 | 7949 | 7942 | yes |

Decode and prefill use the current scalar/AVX2 quantized MatVec and single-threaded operators. Both models
fit the 16 GiB budget: the 7B plan is ~7.75 GiB of weights plus KV, ~7.9 GiB total, leaving room for OS and
scratch.

Quality smoke (greedy, 24 tokens): the 1.5B and 7B both continue `def add(a, b):` with `return a + b` and
`def factorial(n):` with a correct recursive implementation; a truncated last function makes the
concatenated Python syntactically incomplete, which is a token-limit artifact of the smoke, not an engine
defect.

## Limits

- The numbers are the unoptimized baseline. Prefill is superlinear because each position is forwarded over
  its own causal prefix; a batched prefill is a future task.
- Decode is single-threaded. No prompt/KV caching across calls, no batch decode.
- Peak working set counts touched mapped pages; it is not a resident-set guarantee.
- One hardware configuration. No comparison with other runtimes; no token/s promise.
- No quality threshold is asserted here; DS04 fixed an 80% coding pass rate for acceptance, measured
  separately.

## Reproduce

```sh
python3 tools/download_qwen_models.py
dotnet build benchmarks/Satori.Benchmarks -c Release --no-restore -m:1
dotnet run --project benchmarks/Satori.Benchmarks -c Release --no-build -- \
    --qwen-baseline .temp/qwen-corpus/qwen2.5-coder-1.5b-q8_0.gguf 2048 --prefill 64 --decode 16 --quality 2
dotnet run --project benchmarks/Satori.Benchmarks -c Release --no-build -- \
    --qwen-baseline .temp/qwen-corpus/qwen2.5-coder-7b-q8_0.gguf 2048 --prefill 32 --decode 8 --quality 2
```
