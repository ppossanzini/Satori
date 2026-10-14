# Thread Budget and Row Claiming 2026-10-14

Status: experiment and change. Two candidate changes to how a projection uses its workers; one was
measured and withdrawn, the other measured and kept. The figures follow
[measurement-method.md](../../measurement-method.md).

## What was being tested

[hardware-dependence-2026-10-14](../hardware-dependence-2026-10-14/README.md) found that the same
binary runs 1.7x slower when confined to the six physical performance cores than when unpinned, and
attributed it to eight worker tasks sharing six cores through simultaneous multithreading. If that
reading were right, two changes should recover the loss: derive the budget from physical cores, and
stop dealing rows out in equal parts.

## Measured, withdrawn: the physical-core budget

`ProcessorTopology` read the core behind each processor this process may use, from `sysfs` filtered
by the affinity mask, and `InferenceOptions.Default` clamped the budget to that count. Verified
first that the mechanism behaved: under `taskset -c 0-11` the budget fell from eight to six, under
`taskset -c 12-19` it stayed eight, under `taskset -c 20-21` it became two.

Then measured against `dfbe274`, which already carried the integer path, three rounds alternating the
two revisions, confined to the performance cores:

| Round | ffn_down 8 workers | ffn_down 6 workers | output 8 workers | output 6 workers |
|---|---:|---:|---:|---:|
| 1 | 23966.7 | 37050.2 | 189958.7 | 170727.0 |
| 2 | 30134.0 | 23343.3 | 183366.6 | 180305.8 |
| 3 | 25276.4 | 22576.8 | 217182.1 | 184176.4 |

`ffn_down` is unresolved: the eight-worker run is faster in round 1 and the six-worker run in rounds
2 and 3, and the two ranges overlap. The output projection favours the smaller budget in all three
rounds at 1.06x to 1.11x, but the within-configuration spread is 1.64x, so the ratio sits inside the
noise.

**Verdict: unresolved, therefore withdrawn.** The change was reverted. `InferenceOptions.Default`
clamps to `Environment.ProcessorCount` as before, and the reason is recorded on the property.

### What the earlier attribution got wrong

The 1.7x was mostly not the workers sharing cores. `taskset` confines the **whole process** — the
thread pool, the JIT and the collector included — to twelve logical processors instead of
twenty-two, and takes the fast cores with it. The earlier report reasoned from the topology to the
cause without measuring the candidates separately; this experiment separates them and the workers
carry little of the cost.

## Measured, kept: rows claimed instead of dealt

`ExecuteBatchQuantized` gave each worker a contiguous range of `rows / workers` rows and then waited
for all of them, so the call lasted as long as its slowest worker. A worker that drew slow cores, or
that was descheduled, held the whole call at the barrier.

The workers now claim chunks of `rows / (workers x 4)` rows from a shared counter, so a worker that
falls behind claims fewer. Chunks are claimed exactly once and the outputs of a row range are
independent, so the split still produces the same result.

Measured against `6314155`, seven rounds alternating the two revisions, unpinned — the configuration
the runtime actually uses:

| Round | ffn_down dealt | ffn_down claimed | output dealt | output claimed |
|---|---:|---:|---:|---:|
| 1 | 26172.3 | 21139.7 | 180841.1 | 160113.6 |
| 2 | 25202.4 | 23262.8 | 193239.2 | 164257.2 |
| 3 | 31221.4 | 22145.8 | 176339.8 | 161692.7 |
| 4 | 24977.0 | 20757.8 | 172597.6 | 163132.7 |
| 5 | 26799.6 | 25507.1 | 181289.8 | 163134.5 |
| 6 | 30284.3 | 21462.2 | 185929.0 | 156699.3 |
| 7 | 28417.4 | 21284.6 | 186536.1 | 162205.2 |

| Tensor | Ratio range | Median | Winning rounds | Ranges |
|---|---:|---:|---:|---|
| blk.0.ffn_down | 1.05x - 1.41x | **1.24x** | 7 of 7 | overlap in one round |
| output | 1.06x - 1.19x | **1.13x** | 7 of 7 | disjoint |

Every paired round favours the claimed split, on both tensors. A sign test over seven rounds gives
p = 0.008, which is what separates this from the withdrawn budget change: that one reversed in one
of three rounds, this one never reverses in seven.

### The spread is itself evidence

| Tensor | Spread, dealt | Spread, claimed |
|---|---:|---:|
| ffn_down | 1.25x | 1.23x |
| output | 1.13x (172598 - 193239 over 7) | 1.05x (156699 - 164258 over 7) |

The mechanism predicts the variance and not only the mean: a barrier that waits for its slowest
worker produces more variable call times than one whose workers can rebalance, so the smaller spread
under the claimed split is the predicted consequence of removing that wait. It is a second,
independent sign that the change acts where it claims to.

## Where the time went

Applying the median ratios to the Q6_K share of a decode token — 730 ms of 1409 ms from
[decode-2026-10-14](../decode-2026-10-14/README.md):

| | ms/token | tok/s |
|---|---:|---:|
| `6314155` | 1409 | 0.71 |
| this change | ~1257 | ~0.80 |

**Projection, not a measurement.** The DW06.7 threshold of 3 tok/s remains unmet.

## Reproduce

```sh
git worktree add --detach /tmp/ab/base 6314155
(cd /tmp/ab/base && dotnet restore Satori.sln -m:1 && dotnet build Satori.sln -c Release --no-restore -m:1)
ln -sfn "$PWD/.temp" /tmp/ab/base/.temp
bash tools/ab-pinned.sh 7 none        # unpinned, alternating the revision order
bash tools/ab-pinned.sh 3 0-11        # confined to the performance cores
```

## Limits

- One host, one artifact, one tensor pair. The direction is 7 of 7; the magnitude is a range whose
  floor is inside the within-configuration spread, so 1.05x is a floor and 1.24x is a median and
  neither is a guarantee.
- The chunk size, four per worker, was chosen by argument and not searched: two would claim twice as
  often, sixteen would reduce imbalance at the same claim rate but on a smaller granularity than a
  projection needs. A sweep is possible with the same harness and was not run.
- `output` is the last tensor the test times, so it carries the heat of the two before it. The
  alternation cancels that between revisions; it does not remove it.
- `ExecuteBatchFloat` still deals rows out in equal parts. It was left alone deliberately: changing
  it without a measurement would be the same mistake the withdrawn budget change made.
