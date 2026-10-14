# Hardware Dependence of the Integer Path 2026-10-14

Status: experiment. Answers whether the ranking between the Q6_K integer path and the
floating-point path is stable across hardware, and whether an install-time probe could decide
between them. No code changed: this is a measurement only.

## Question

If the two paths rank differently on different hardware, should both be selectable and a probe at
installation choose the faster one per host?

## Method

`RealArtifactProjectionsAgainstSynthetic`, minimum of nine repetitions, on the artifact's own
`blk.0.ffn_down` (3584x18944, Q6_K) and `output.weight` (152064x3584, Q6_K). Baseline `dfbe274`
against the working tree that puts Q6_K on the integer kernel, each built once and run from its own
directory. The two revisions are **alternated within each round**, four rounds, on each core class.

This host is hybrid: P cores 0-11 (6 physical, 12 logical, 4.5-4.8 GHz), E cores 12-19 (8 physical,
8 logical, 3.8 GHz), LP-E cores 20-21 (2 physical, 2.5 GHz). `taskset` partitions the run.

## Result

Microseconds per call, minimum of nine repetitions.

### P cores (0-11)

| Round | ffn_down base | ffn_down integer | output base | output integer |
|---|---:|---:|---:|---:|
| 1 | 32912.8 | 28930.8 | 239227.2 | 188556.7 |
| 2 | 37458.6 | 28470.4 | 256465.3 | 176409.7 |
| 3 | 38525.9 | 26096.2 | 244627.0 | 179612.2 |
| 4 | 36499.3 | 27415.4 | 242917.9 | 184418.5 |

Ranges are disjoint on both tensors. The integer path wins **1.26x** on `ffn_down` and **1.36x** on
the output, 4 of 4 rounds each.

### E cores (12-19)

| Round | ffn_down base | ffn_down integer | output base | output integer |
|---|---:|---:|---:|---:|
| 1 | 33873.5 | 35288.9 | 188497.9 | 190884.5 |
| 2 | 48354.3 | 25949.8 | 180679.6 | 190207.9 |
| 3 | 79679.5 | 25566.4 | 181130.5 | 197340.1 |
| 4 | 32103.3 | 25168.3 | 183189.6 | 196568.5 |

`ffn_down` favours the integer path at the minimum, 1.28x, but the ranges **overlap**: the baseline
reaches 32103.3 and the candidate 35288.9, so on this core class the result is unresolved.

`output` reverses. The ranges are disjoint and the **floating-point path is faster by 1.05x**, 4 of
4 rounds. This is the answer to the question: the ranking is not stable across hardware.

### Unpinned, for reference

From the previous report: integer wins 1.31x on `ffn_down` and 1.23x on the output, all ranges
disjoint. The running configuration is unpinned.

## The noise floor, measured by accident

A first attempt wrote the alternation with an unquoted scalar, which zsh does not word-split, so both
"revisions" were the same binary. The runs are useless as a comparison and useful as a noise floor:
**the same binary, back to back, in the same conditions**.

| Core class | Run 1 | Run 2 | Spread |
|---|---:|---:|---:|
| P cores | 23099.3 | 22745.3 | 1.02x |
| E cores | 41076.5 | 25445.8 | **1.61x** |

On E cores one binary varies by 61% on `ffn_down`. The baseline's own series above (33873, 48354,
79679) rises monotonically across rounds, which is the float path doing roughly twice the arithmetic
and heating more. The effect being measured on that core class is smaller than the variation of a
single build within it.

## What this means

1. **The ranking does depend on hardware, and it inverts.** Confirmed on one host, on the output
   tensor, at E cores: the floating-point path is 5% faster there and 36% slower at P cores.
2. **The inversion is small and sits at the resolution limit where it happens.** 5% at E cores,
   where the same binary varies by up to 61%. A probe cannot decide it; the probe would be choosing
   from noise.
3. **The Q4_K path is not in question.** Its advantage comes from `vpmaddubsw`, a byte-pair
   multiply-add, and is an order of magnitude larger than any of this. Nothing here suggests it
   reverses anywhere.
4. **Core class changed both paths far more than the path choice changed either.** `ffn_down` at
   P cores is 1.7x slower than unpinned for the same binary. The cause was first attributed to eight
   worker tasks sharing six physical cores through SMT; see the correction below, where the
   confinement of the whole process turns out to carry most of it.

## Consequence for the design

**No install-time probe.** The property it would sample is smaller than its own noise on the core
class where it flips, and it decays silently as the host, the scheduler and the kernels change.

**An explicit preference, yes.** Both paths exist already: the floating-point path is the no-AVX2
fallback and the numerical oracle for the integer one. Exposing `Auto | Integer | Float` on
`InferenceOptions`, resolved by a deterministic capability ladder (`Integer` only when AVX2 is
present), gives what the question asked for — both reachable — without inventing a measurement.
`Auto` should keep the integer path: it wins unpinned, which is how the runtime is used.

**The larger finding is the thread budget.** On this host the largest single effect measured is
neither path: it is that the same binary runs 1.7x slower confined to the six physical performance
cores than unpinned. A thread budget derived from physical cores, and from the topology rather than
from a protocol constant, is worth more than the path decision and is measurable with the same
harness.

**Correction, same day.** That attribution was too strong and has been tested:
[thread-budget-2026-10-14](../thread-budget-2026-10-14/README.md) shows the 1.7x is mostly the
confinement itself, since `taskset` restricts the whole process — pool, JIT and collector — and not
only the eight workers. A budget clamped to physical cores measured unresolved and was withdrawn. What
did resolve, in 7 of 7 alternating rounds, is claiming rows from a shared counter instead of dealing
them out in equal parts.

## Limits

- One host, two tensors, one format. The reversal is demonstrated, not characterised.
- The E-core baseline series is not thermal-stabilised, so its absolute figures are not comparable
  across rounds; only the paired comparison within a round is used, and the overlapping ranges are
  reported rather than resolved.
- `output` is the last tensor the test times, so its measurement carries the heat of the two before
  it. The alternation cancels this between revisions; it does not remove it.
- LP-E cores were not measured: eight worker tasks on two physical cores is a 4:1 oversubscription
  whose result would say nothing about either path.
