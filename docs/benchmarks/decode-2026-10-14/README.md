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

`BatchedMatVecThroughputTests`, synthetic Q4_K payloads, real shapes, one position, 8 threads,
minimum of nine repetitions:

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
