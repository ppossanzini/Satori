# Q6_K Integer Path Measurement 2026-10-14

Status: DW06.7 follow-up. Records whether moving Q6_K off the floating-point path onto the integer
kernel helped. The figures follow [measurement-method.md](../../measurement-method.md).

## What was asked

[decode-2026-10-14](../decode-2026-10-14/README.md) established that Q6_K, not Q4_K, carries the
decode cost: the output projection and the 28 `ffn_down` weights account for 730 ms of a 1409 ms
token. The change being measured puts Q6_K on `QuantizedKAvx2Kernel.ExecuteQuantizedTiled` with the
scalar codec product, in place of `ExecuteBatchFloat`.

## Environment

| | |
|---|---|
| Host | Intel Core Ultra 7 155H, 22 logical processors, 32 GB, Arch Linux |
| Artifact | Qwen2.5-Coder 7B Instruct Q4_K_M, SHA-256 `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| Threads | 8 |
| Baseline | `dfbe274`, built in a detached worktree |
| Candidate | working tree, same parent commit |
| Measurement | `RealArtifactProjectionsAgainstSynthetic`, minimum of nine repetitions, three rounds, the two revisions alternated within each round |

## Result

Real Q6_K tensors, microseconds per call, minimum of nine repetitions:

| Round | ffn_down `dfbe274` | ffn_down candidate | output `dfbe274` | output candidate |
|---|---:|---:|---:|---:|
| 1 | 19690.1 | 18939.2 | 171003.0 | 135982.1 |
| 2 | 21332.9 | 14975.6 | 165366.6 | 132494.5 |
| 3 | 22196.7 | 16065.5 | 153938.3 | 124734.2 |

The two ranges are **disjoint on both tensors**: the slowest candidate run beats the fastest
baseline run in every pairing.

| Tensor | Shape | Baseline min us | Candidate min us | Ratio |
|---|---|---:|---:|---:|
| blk.0.ffn_down | 3584x18944 | 19690.1 | 14975.6 | **1.31x** |
| output | 152064x3584 | 153938.3 | 124734.2 | **1.23x** |

Rate at the minimum: `ffn_down` 3448 -> 4534 GMAC/s, `output` 3187 -> 4113 GMAC/s.

## The measurement that nearly hid this

Run inside the full 223-test suite, the same candidate reported `ffn_down` at 22261.4 us and the
output at 155221.8 us — 19% and 14% above its own controlled figures. Compared against a baseline
number taken from an earlier session, that reads as "no gain", and that is the conclusion this
reporting initially reached.

The two figures are not comparable. A run inside a suite inherits the suite's thermal and
scheduling state; a run on its own does not. Alternating the two revisions inside one session is
what separates the change from that state, and it is the only comparison this report uses.

## Effect on decode

Applying the measured ratios to the 730 ms Q6_K share of the 1409 ms token:

| | ms/token | tok/s |
|---|---:|---:|
| `dfbe274` | 1409 | 0.71 |
| candidate | ~1263 | ~0.79 |

**Projection, not a measurement**: the token figure is the decode accounting from
[decode-2026-10-14](../decode-2026-10-14/README.md) with the measured tensor ratios applied, not a
new end-to-end timing. The DW06.7 threshold of 3 tok/s remains unmet, and this change does not
close it — it removes 1.3x from a share that needs roughly 12x.

## Reproduce

```sh
git worktree add --detach /tmp/ab/base dfbe274
(cd /tmp/ab/base && dotnet restore Satori.sln -m:1 && dotnet build Satori.sln -c Release --no-restore -m:1)
ln -sfn "$PWD/.temp" /tmp/ab/base/.temp
for round in 1 2 3; do
  (cd /tmp/ab/base && dotnet test tests/Satori.Numerics.Tests -c Release --no-build -m:1 \
    --filter 'FullyQualifiedName~RealArtifactProjectionsAgainstSynthetic' \
    --logger 'console;verbosity=detailed' > /tmp/ab/r$round-base.log 2>&1)
  dotnet test tests/Satori.Numerics.Tests -c Release --no-build -m:1 \
    --filter 'FullyQualifiedName~RealArtifactProjectionsAgainstSynthetic' \
    --logger 'console;verbosity=detailed' > /tmp/ab/r$round-current.log 2>&1
done
```

## Limits

- One host, one artifact. The ratios are more portable than the rates but are not a general claim.
- The within-configuration spread reaches 26% on the candidate's `ffn_down`, so a single round
  proves nothing. Three alternated rounds with disjoint ranges do.
- The synthetic figures in the same test use uniform scales and are in-cache; only the real-byte
  rows above are used here.
- The integer path is lossy by construction. Its difference from the floating-point path is bounded
  by `BatchedMatVecTests`, with the bound derived from the payload rather than tuned.
