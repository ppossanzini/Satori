# Batched Prefill Measurement 2026-10-14

Status: DW06.6 (P04). The Satori CPU path's prefill, measured against its own previous design and against
the DW04 reference on the same host. This is the evidence behind the DW06.6 verdict: the structure is
correct and a real gain, but the TTFT threshold is **not** met.

## Environment

| | |
|---|---|
| Host | Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux |
| Artifact | Qwen2.5-Coder 7B Instruct Q4_K_M, SHA-256 `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Context / threads | 4096 / 8 (`InferenceOptions.BudgetThreads`) |
| Build | `dotnet build Satori.sln -c Release --no-restore -m:1` |
| Prompt | `tools/dw06_prefill_prompt.py --words 512`, 607 tokens after tokenization |

## End-to-end prefill

`measure.py --label prefill-512w-t8 -- dotnet run --project src/Satori.Console -c Release --no-build -- text <model> <prompt> 0 --context 4096 --threads 8`

| Run | Prompt tokens | Wall (s) | Peak RSS (MiB) | Prefill tok/s |
|---|---:|---:|---:|---:|
| Batched, 8 threads | 607 | 166.35 | 4756 | 3.65 |

Wall time includes process start, model validation and first-touch of the mapped weights, which the
DW04 reference figure excludes. The reference is 22.16 prefill tok/s (llama.cpp, same host, batch 2048), so
the Satori prefill rate is about **6x below the reference** and about **9x below the 60 s TTFT threshold**
when extrapolated to 2048 tokens.

## Isolated projection throughput

`BatchedMatVecThroughputTests`, Q4_K, 3584x3584, 64 positions, synthetic payload, `Stopwatch` around the
call after a warm-up call.

| Path | Time (ms) | GMAC/s | Ratio |
|---|---:|---:|---:|
| Batched, 1 thread | 74.6 | 11.0 | 1.00x |
| AVX2, one call per position | 207.0 | 4.0 | 2.77x slower than batched |
| Scalar, one call per position | 759.2 | 1.1 | 10.18x slower than batched |

Thread scaling of the batched path on the same matrix and positions:

| Threads | Time (ms) | GMAC/s |
|---:|---:|---:|
| 1 | 91.3 | 9.0 |
| 2 | 38.6 | 21.3 |
| 4 | 18.7 | 43.9 |
| 8 | 20.9 | 39.3 |

Scaling stops at 4 threads: the projection at this shape is bound by the shared memory system, not by the
row split, and adding workers past that point costs more in scheduling than it recovers. The reference
kernel reaches roughly 154 GMAC/s on the same host, about 3.5x the best Satori figure.

## Interpretation

The batched design is doing what it was built for: 2.77x the shipped AVX2 path and 10.2x the scalar path on
the same bytes, because a weight block is decoded once per chunk instead of once per position. The
remaining gap to the reference is **inside the dot kernel**, not in the prefill structure:

- the Satori dot reconstructs the block to F32 and then multiplies with FMA, so an AVX2 instruction
  performs 8 multiply-adds;
- the reference quantizes the activations per block (Q8_K) and multiplies integers, so one instruction
  performs 64 multiply-adds and the activation bytes are shared across rows.

Closing that gap is kernel work, not prefill work, and it is where DW06.7 and DW09 now point.

## Reproduce

```sh
cd <repo>
dotnet build Satori.sln -c Release --no-restore -m:1
python3 tools/dw06_prefill_prompt.py --words 512 --out .temp/deepseek/dw06/prompt-512w.txt
P="$(cat .temp/deepseek/dw06/prompt-512w.txt)"
python3 docs/benchmarks/comparison-2026-10-13/measure.py --label prefill-512w-t8 -- \
  dotnet run --project src/Satori.Console -c Release --no-build -- text \
  .temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf "$P" 0 --context 4096 --threads 8
dotnet test tests/Satori.Numerics.Tests -c Release --filter FullyQualifiedName~BatchedMatVecThroughputTests \
  --logger 'console;verbosity=detailed'
```

## Limits

- One host, one artifact, one repetition. The prompt rate on this CPU is sensitive to concurrent load.
- The isolated figures use a synthetic payload with uniform scales, so they bound the kernel's own cost and
  not a real tensor's distribution.
- Peak RSS counts touched mapped pages, so it is a peak working set, not a resident-set guarantee.
- The end-to-end number includes one-time costs the reference figure excludes; the isolated figures and the
  end-to-end number are consistent, but they are not interchangeable.
