# Decode Measurement 2026-10-14

Status: DW06.7 (P04). The Satori CPU path's decode rate against its own previous design and against the DW04
reference on the same host. Two things were built here — a Q8_K activation quantizer with integer weight
products, and a row-tiled batched kernel with its constants hoisted out of the position loop — and one of
them is the reason the decode rate moved. The DW06.7 threshold is **not** met.

## Environment

| | |
|---|---|
| Host | Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux |
| Artifact | Qwen2.5-Coder 7B Instruct Q4_K_M, SHA-256 `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Context / threads | 512 / 8 |
| Command | `text <model> "def fibonacci(n):" 16 --context 512 --threads 8` |

## End-to-end decode

| Revision | Wall (s) | Total (ms) | Per token (ms) | Decode (tok/s) |
|---|---:|---:|---:|---:|
| DW06.6 (per-position F32 kernel) | 29.23 | 28405 | 1673 | 0.60 |
| DW06.7 (batched step) | 22.31 | 21473 | 1233 | 0.81 |

Process start plus a four-token prompt costs 1743 ms (measured with `max-new-tokens 0`), so the per-token
figure is the generation cost alone. The gain is **1.32x**. The threshold is 3 tok/s, so decode is **3.7x
short** of it.

## Where a decode token's time goes

`BatchedMatVecThroughputTests`, synthetic Q4_K payloads, real model shapes, one position, 8 threads.

| Projection | Shape | Batched (ms) | Batched GMAC/s | Per-position GMAC/s | Ratio |
|---|---|---:|---:|---:|---:|
| gate / up | 18944 x 3584 | 7.087 | 9.6 | 1.0 | 9.38x |
| down | 3584 x 18944 | 6.961 | 9.8 | 3.0 | 3.23x |
| q / o | 3584 x 3584 | 2.490 | 5.2 | 4.5 | 1.15x |
| output | 152064 x 3584 | 41.551 | 13.1 | 3.9 | 3.38x |
| k / v | 512 x 3584 | 0.883 | 2.1 | 4.6 | **0.45x** |

One token is 197 of these projections, 7.07 GMAC in total. Timed as a whole they take 1050.5 ms, a ceiling
of 0.95 tok/s before attention and the elementwise operators, which is consistent with the 0.81 tok/s
measured end to end.

Two facts follow from the table and neither was what DW06.6 predicted:

- **The gain is structural, not integral.** At 16 positions the batched F32 kernel reaches 41.9 GMAC/s and
  the batched integer kernel 55.6 GMAC/s, so the integer path adds about 30%. But the step from 4.5 to
  9.6 GMAC/s on gate/up comes from hoisting the block constants and tiling the rows, not from multiplying
  integers. DW06.6's report named the integer dot as the missing piece; that was half right at most.
- **The batched kernel has a fixed per-call cost of about 0.5 ms** that the per-position kernel does not
  have, because it quantizes its activation and builds a tiled sweep. On `k`/`v`, 1.8 MMAC, that cost is
  larger than the whole projection and the batched path is **2.2x slower**. Routing every projection through
  it is a net 1.32x on decode only because gate/up/down and the output projection are much larger.

## Distance to the reference

The reference reaches about 154 GMAC/s on this host. The best single projection here reaches 13.1 GMAC/s at
one position and the token as a whole achieves 6.7 GMAC/s, so decode is roughly **12x below** the reference
before any memory-bandwidth limit is considered. Reaching 3 tok/s needs about 0.33 s per token against the
1.23 s measured: a factor 3.7, which the identified causes cannot supply.

## Verdict

The DW06.7 criterion is not met. What is delivered is a verified 1.32x decode gain and a measured account of
where the remaining factor lives:

1. a per-call fixed cost that makes the batched kernel wrong for small projections, so a size-aware dispatch
   is needed and its threshold must come from more than one host's timing;
2. a rate ceiling of 13 GMAC/s on a single large projection at one position, about 12x below the reference,
   which is kernel work: wider register blocking, weight-side 8-bit multiply-add for the small-K case,
   software prefetch of the next weight block;
3. attention and the elementwise operators account for the gap between the 0.95 tok/s projection ceiling and
   the 0.81 tok/s measured, and have not been optimized at all.

None of these is a prefill structure problem and none is a codec problem. They are kernel and dispatch work
whose size is now known rather than guessed.

## Reproduce

```sh
python3 docs/benchmarks/comparison-2026-10-13/measure.py --label decode -- \
  dotnet run --project src/Satori.Console -c Release --no-build -- text \
  .temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf "def fibonacci(n):" 16 --context 512 --threads 8
dotnet test tests/Satori.Numerics.Tests -c Release \
  --filter "FullyQualifiedName~BatchedMatVecThroughputTests" --logger 'console;verbosity=detailed'
python3 tools/q8k_reference.py
dotnet test tests/Satori.Numerics.Tests -c Release --filter FullyQualifiedName~Q8KCodecTests
```

## Limits

- One host, one artifact, one repetition per end-to-end figure; the isolated figures are single runs after
  one warm-up call.
- Isolated figures use synthetic payloads with uniform scales, so they bound a kernel's own cost and not a
  real tensor's distribution.
- The 0.95 tok/s ceiling is the projection subset only and assumes the matrices are read from memory once.
- The Q8_K path is lossy, so its results differ from the floating-point path in the last bits; the tests
  bound that difference and the Q4_K codec has separate exact tests against the independent reference.
