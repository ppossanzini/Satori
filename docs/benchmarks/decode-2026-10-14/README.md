# Decode Measurement 2026-10-14

Status: DW06.7 (P04). **Corrected 2026-10-14.** The first version of this report used single-run
measurements and reached a wrong diagnosis; the section "Correction" below states what changed and
why. The figures here follow [measurement-method.md](../../measurement-method.md).

## Correction

The first version reported that the batched kernel was 2.2x **slower** than the per-position kernel
on the 512x3584 k/v projection, and concluded that a size-aware dispatch was required. Both came
from one warmed call and one timed call. With five repetitions and the minimum taken:

| Claim in the first version | Measured | Verdict |
|---|---|---|
| batched is 2.2x slower on k/v (512x3584) | batched 0.116 ms, per-position 0.358 ms | **wrong direction**: batched is 3.1x faster |
| one 3584x3584 projection at one position: 5.2 GMAC/s | 23.6 GMAC/s | understated 4.5x |
| a size-aware dispatch is needed | it is not; the fixed cost never exceeded the work | **withdrawn** |
| decode is ~12x below the reference | ~6x below, using the reference's 154 GMAC/s | understated |

The design is simpler than the first report said: no dispatch, one kernel per format.

## Environment

| | |
|---|---|
| Host | Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux |
| Artifact | Qwen2.5-Coder 7B Instruct Q4_K_M, SHA-256 `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Context / threads | 512 / 8 |
| Revision | `0d4581a` (DW06.7) |
| Harness | `tools/ab_decode.py`, 5 rounds alternating the two configurations |

## End-to-end decode

| Config | min ms/token | median | max | tok/s at min | tok/s at median |
|---|---:|---:|---:|---:|---:|
| `0d4581a` | 1479.1 | 1619.6 | 1807.5 | 0.676 | 0.617 |
| working tree, branch-free scale unpacking | 1508.6 | 1612.6 | 1690.9 | 0.663 | 0.620 |

Ratio 0.980 against a within-configuration spread of **1.22x**. The unpacking is **unresolved and
marginally the wrong direction**, so it was reverted. What the working tree keeps is the
measurement fix, not the optimization.

The threshold is 3 tok/s. Decode sits at **0.65 tok/s, 4.6x short**, and this is the first figure in
this project stated with a resolution behind it.

An earlier single-run sequence on the same artifact gave 1390, 2080 and 1172 ms/token for
equivalent code — the 1.5x inconsistency that motivated writing the method document down.

## The gap between the kernel benchmark and the workload

### The real artifact against the synthetic payload

`RealArtifactProjectionsAgainstSynthetic` loads the artifact's own tensors and times them against a
synthetic payload of the same shape and format, minimum of nine repetitions. This is the
experiment that separates "how fast is the kernel" from "how fast is the model".

| Tensor | Format | Shape | Real (us) | Real GMAC/s | Synthetic GMAC/s | Ratio |
|---|---|---|---:|---:|---:|---:|
| blk.0.attn_q | Q4_K | 3584x3584 | 1567.2 | 8.2 | 9.6 | 1.17x |
| blk.0.ffn_gate | Q4_K | 18944x3584 | 5652.4 | 12.0 | 29.2 | 2.43x |
| blk.0.ffn_down | **Q6_K** | 3584x18944 | 20407.8 | **3.3** | 3.6 | 1.07x |
| output | **Q6_K** | 152064x3584 | **159145.8** | **3.4** | 4.1 | 1.21x |

The synthetic payload is 1.07x to 2.43x optimistic, which is the cache effect and is real but small.
**The dominant fact is the format, which nothing in this project had measured:**

- `ffn_down` and the output projection are **Q6_K**, not Q4_K. So are 29 of the artifact's 198
  quantized tensors, about 2.0 GB of the 4.3 GB a token reads.
- Q6_K has no integer kernel, so it runs `QuantizedMatVec.ExecuteBatchFloat`, which decodes each
  block to F32 and multiplies with FMA. That path reaches **3.3-3.4 GMAC/s**, against **8.2-12.0
  GMAC/s** for the integer Q4_K path: **2.5x to 3.6x slower per MAC**.

Arithmetic for one decode token:

| Portion | Time | Share |
|---|---:|---:|
| output projection (Q6_K) | 159 ms | 11% |
| 28 x ffn_down (Q6_K) | 571 ms | 41% |
| everything else (Q4_K) | ~679 ms | 48% |
| **total** | **~1409 ms** | |

**Q6_K tensors are 730 ms of the token, more than half.** Fixing them is the largest single piece
of decode work available and it is a kernel task with a known oracle, because the Q6_K layout
already has an independent reference fixture.

### What the numbers rule out

- The path streams 4.3 GB in 1409 ms, about **3.1 GB/s**. This machine's memory exceeds 50 GB/s,
  so it is **not bandwidth-bound**: a compute path that reaches 12 GMAC/s can be three times faster
  before the memory system matters.
- Raising the thread-pool floor gave 1.11x and 0.94x, i.e. nothing: the pool is not the wall.
- The activation quantization costs 9 us against calls of 117-762 us: 1% to 8%, not the cost.
- The thread count is worth about 7%, measured at 1.070x against a 1.11x spread: below the
  resolution and not established.
- The thread curve does not flatten after the fast cores, so the hybrid core mixture does not
  explain the scaling either.

### What it points at

Two independent facts agree on the same place. Q6_K runs at 3.3 GMAC/s because it has no integer
kernel. And every kernel discards its decoded weights after one use while a token reads 4.3 GB, so
the loop is worth optimizing for streaming rather than for cache.

The next tasks, in order of measured value:

1. **A Q6_K integer kernel**, covering 730 ms of a 1409 ms token. The layout has four codes per
   byte under four scales, which is why it needs its own design rather than a copy of the Q4_K one.
2. **Prefetching and software pipelining** in the row loop, since the rate sits at 6% of the
   memory system's capacity and the access pattern streams.
3. Attention and the elementwise operators, which account for the gap between the projection total
   and the measured token and have not been touched.

## Legacy: the synthetic-only view

`BatchedMatVecThroughputTests`, synthetic Q4_K payloads, real shapes, one position, 8 threads,
minimum of nine repetitions. Read the section above first: these figures are all in-cache Q4_K and
the real artifact is neither.

| Projection | Shape | Batched (ms) | Batched GMAC/s | Per-position GMAC/s | Ratio |
|---|---|---:|---:|---:|---:|
| gate / up | 18944 x 3584 | 2.115 | 32.1 | 5.2 | 6.22x |
| down | 3584 x 18944 | 2.046 | 33.2 | 5.0 | 6.64x |
| q / o | 3584 x 3584 | 0.544 | 23.6 | 5.2 | 4.53x |
| output | 152064 x 3584 | 12.323 | 44.2 | 5.0 | 8.83x |
| k / v | 512 x 3584 | 0.116 | 15.8 | 5.1 | 3.09x |

The batched kernel is faster on **every** shape, including the smallest: the fixed per-call cost
never dominates. One token's 197 projections total 7.07 GMAC in **271.7 ms**, a ceiling of
**3.68 tok/s** — above the threshold.

**That ceiling is not the workload, and the difference is the finding.** The workload runs 5.5x
slower: 1479 ms against 272 ms. The synthetic payload reuses the same matrices, so they stay in the
last-level cache; the real model streams 4.3 GB of weights per token. The synthetic figure bounds
the kernel's own cost and nothing else.

What the gap therefore means:

- 4.3 GB in 1479 ms is **2.9 GB/s** of weight traffic. This machine's memory does well over
  50 GB/s, so the path is **not** bandwidth-bound and the memory system is not the wall.
- Scaling is poor: 1 thread 2816 ms/token, 8 threads 1172-1479 ms/token, about 2.4x for 8x the
  threads. A compute-bound path would scale further; a path dominated by per-call fixed costs
  scales like this.
- The fixed costs are known and countable. A decode token makes 197 projections, and each one
  takes a lock and acquires a pointer on the mapping, rents an activation buffer, quantizes the
  activation row, and **spawns one task per worker**. That is roughly 1576 task spawns and 197
  lock acquisitions per token. The synthetic benchmark hides them twice over: the mapping is warm
  and uncontended, and the payload fits in cache.

The next optimization therefore targets scheduling and per-call cost, not arithmetic: a threshold
below which the row split is not worth a task, and a mapping kept open across a layer step instead
of reacquired per projection. Both are measurable with the harness above, and neither is justified
until it is.

## Reproduce

```sh
git worktree add --detach /tmp/ab/committed-src 0d4581a
(cd /tmp/ab/committed-src && dotnet publish src/Satori.Console -c Release -o /tmp/ab/committed)
dotnet publish src/Satori.Console -c Release -o /tmp/ab/current
python3 tools/ab_decode.py --model <artifact> --a committed=/tmp/ab/committed --b current=/tmp/ab/current
dotnet test tests/Satori.Numerics.Tests -c Release \
  --filter "FullyQualifiedName~BatchedMatVecThroughputTests" --logger 'console;verbosity=detailed'
```

## Limits

- One host, one artifact. The absolute rates are this machine's; the ratios are more portable but
  are not a general claim.
- The within-configuration spread is 1.22x, so any ratio inside that is unresolved. Reaching
  tighter resolution needs either more rounds or a quieter host, and neither was attempted.
- Per-token figures are generation only: the harness subtracts process start and model validation
  by reading the console's own counters.
- The synthetic figures use uniform scales in the payload, so they do not represent a real
  tensor's distribution, and they are in-cache.
- The Q8_K integer path is lossy by construction; its difference from the floating-point path is
  bounded by a test, not eliminated.
