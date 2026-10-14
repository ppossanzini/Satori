# Measurement Method

Status: binding for every performance figure this project reports. It exists because a set of
single-run measurements produced a wrong diagnosis that survived two subtasks, and the cost of
that was a design decision (a size-aware dispatch) that the data did not support.

## What went wrong

DW06.6 and DW06.7 were measured with one warmed call and one timed call per configuration, on an
Intel Core Ultra 7 155H laptop. Running the same published binary twice on the same artifact:

| Run | Decode, 8 tokens |
|---|---:|
| A | 11120 ms total, 1390 ms/token |
| B | 16642 ms total, 2080 ms/token |

**1.5x apart on identical code.** From that base, the project concluded that the batched kernel
was 2.2x *slower* than the per-position kernel on a 512x3584 projection and that a size-aware
dispatch was therefore required. Measured with repetitions, the batched kernel is 3.1x *faster* on
that projection and no dispatch is needed. The wrong conclusion was not a small error; it was the
justification for a piece of design work that would have made the code worse.

## Rules

1. **Never report a single run.** Every figure is the minimum of at least five repetitions after
   one warm-up. The median, maximum and the spread are reported beside it. The minimum is the
   repetition least disturbed by scheduling and frequency changes; the spread is what tells a
   reader whether a ratio is real.
2. **Compare revisions in one alternated process pair, not in sequence.** `tools/ab_decode.py`
   runs A and B alternately within each round and reverses their order every round, so thermal
   drift and background load are shared instead of landing on one side.
3. **State the resolution before the comparison.** With the spread measured, a ratio inside the
   spread is **unresolved**, not a win. The harness prints a warning above 15% and the warning is
   binding: a change below that is reverted, not kept with a note.
4. **In-cache benchmarks do not answer workload questions.** A synthetic matrix that is reused
   stays in the last-level cache; a real model streams 4.3 GB per token. The two differ by more
   than 5x here. A synthetic figure bounds a kernel's own cost and may never be presented as the
   workload rate.
5. **Every reported figure names what it measured.** Kernel cost, projection subset, one token, or
   end to end including startup. The four are not interchangeable and mixing them produced the
   discrepancy this document exists to prevent.
6. **A change that cannot be shown to help is reverted.** Carrying an unproven optimization
   forward costs maintenance and hides the next measurement's signal.

## The A/B harness

```sh
# publish both revisions side by side
git worktree add --detach /tmp/ab/committed-src <revision>
dotnet publish src/Satori.Console -c Release -o /tmp/ab/committed   # in the worktree
dotnet publish src/Satori.Console -c Release -o /tmp/ab/current     # in the working tree

python3 tools/ab_decode.py \
  --model .temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --a committed=/tmp/ab/committed --b current=/tmp/ab/current \
  --tokens 8 --repeats 5
```

The harness reads the console's own `elapsed_ms` and `generated` counters, so process start and
model validation are excluded and the figure is the per-token generation cost. It uses a Git
worktree so neither revision's working tree is disturbed.

## Applied to the DW06.7 change

The one measurement this method has decided so far:

| Config | min ms/token | median | max | tok/s at min |
|---|---:|---:|---:|---:|
| committed `0d4581a` | 1479.1 | 1619.6 | 1807.5 | 0.676 |
| working tree (branch-free scale unpacking) | 1508.6 | 1612.6 | 1690.9 | 0.663 |

Ratio 0.980 against a within-configuration spread of 1.22x. **Unresolved, and marginally the wrong
direction, so the branch-free unpacking was reverted.** The kernel returned to `0d4581a`, and only
the measurement fix was kept.
