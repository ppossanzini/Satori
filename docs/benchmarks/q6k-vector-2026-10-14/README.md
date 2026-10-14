# Q6_K Vectorised Product 2026-10-14

Status: measurement of the vectorised Q6_K integer product committed in `a94b9b3`. The product replaced the
scalar codec product on the decode path; this records what the replacement is worth.

## Environment

| | |
|---|---|
| Host | Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux |
| Artifact | Qwen2.5-Coder 7B Instruct Q4_K_M, SHA-256 `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Threads | 8, unpinned |
| Baseline | `9a136cf`, the scalar codec product, built in a detached worktree |
| Measurement | `RealArtifactProjectionsAgainstSynthetic`, minimum of nine repetitions, seven rounds, the two revisions alternated within each round |

## Result

Microseconds per call on the artifact's own Q6_K tensors:

| Round | ffn_down scalar | ffn_down vectorised | output scalar | output vectorised |
|---|---:|---:|---:|---:|
| 1 | 19145.9 | 15102.4 | 145201.4 | 42711.3 |
| 2 | 17339.6 | 14942.0 | 139177.8 | 35783.3 |
| 3 | 17659.3 | 13772.6 | 133911.0 | 37974.8 |
| 4 | 19502.8 | 13902.4 | 139521.5 | 36574.4 |
| 5 | 18020.3 | 14211.4 | 132056.0 | 36547.7 |
| 6 | 18037.4 | 14229.9 | 139417.1 | 38010.1 |
| 7 | 18173.4 | 14772.5 | 135453.0 | 49530.4 |

| Tensor | Shape | Winning rounds | Median ratio | Rate before | Rate after | Ranges |
|---|---|---:|---:|---:|---:|---|
| blk.0.ffn_down | 3584x18944 | 7 of 7 | **1.26x** | 3.9 GMAC/s | 4.9 GMAC/s | disjoint |
| output | 152064x3584 | 7 of 7 | **3.69x** | 4.1 GMAC/s | **15.2 GMAC/s** | disjoint |

## Why the two differ so much

The scalar product's cost per block is dominated by instruction count rather than by the products: it
re-reads sixteen sub-scales, shifts and masks four quants per byte, and issues two multiply-adds per
quant across 256 weights. That fixed cost is paid once per block, so it amortises better where a block
holds more useful work — and it amortises worst on the tensor with the most blocks per unit of output.

`ffn_down` is 3584 rows of 18944 columns: 74 blocks per row and 3.8 rows per thread at eight threads, so
each thread's blocks are few and the per-block overhead is a smaller share. `output` is 152064 rows of 3584
columns: 14 blocks per row and 19008 rows per thread, so a thread pays the per-block cost 266112 times for
shallow rows. The same 1.26x on the shallow tensor became 3.69x on the deep one.

That is also why the earlier reports read the two tensors as similar: at 3.3 and 3.4 GMAC/s they were both
dominated by the scalar overhead, and the overhead was the same for both. Removing it reveals the shapes.

## Effect on decode

Applying the measured ratios to the Q6_K share of a 1409 ms token from
[decode-2026-10-14](../decode-2026-10-14/README.md) — 159 ms for the output projection and 571 ms for the
28 `ffn_down` weights:

| | ms/token | tok/s |
|---|---:|---:|
| before | 1409 | 0.71 |
| after | ~1175 | ~0.85 |

**Projection, not a measurement.** The DW06.7 threshold of 3 tok/s is still unmet, and this is the first
change of the session that moves the number enough to be worth projecting: the Q6_K share falls from 730 ms
to about 500 ms.

The output projection is now the fastest of the two Q6_K tensors per MAC and no longer the largest single
cost of a token. What remains is 28 `ffn_down` calls at 4.9 GMAC/s, which is where the next measurement
should look.

## Reproduce

```sh
git worktree add --detach /tmp/ab/base 9a136cf
(cd /tmp/ab/base && dotnet restore Satori.sln -m:1 && dotnet build Satori.sln -c Release --no-restore -m:1)
ln -sfn "$PWD/.temp" /tmp/ab/base/.temp
bash tools/ab-pinned.sh 7 none kern
```

## Limits

- One host, one artifact, two tensors. The direction is 7 of 7 on both; the magnitude is peculiar to the
  tensor shapes and is not a general claim about Q6_K.
- The baseline is the scalar product as it stood, not the floating-point path that preceded it. The chain
  from the floating-point path is in [q6k-integer-2026-10-14](../q6k-integer-2026-10-14/README.md).
- The per-round spread on the vectorised `output` reaches 1.38x, from a 49530 us outlier in the last round.
  The two ranges do not come close to overlapping, so the ratio is resolved with a wide margin; the median
  is quoted rather than the minimum for that reason.
- The projection above is arithmetic on a previous report's accounting, not an end-to-end timing of a token.
