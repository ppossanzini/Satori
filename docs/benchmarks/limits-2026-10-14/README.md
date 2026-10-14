# Resource Limits and Cancellation 2026-10-14

Status: DW06.8. The criterion is *memory and context bounds enforced; cancellation ≤ 2 s; no sustained swap*.
This records what was already enforced, what was added, and what the cancellation delay measures.

## What was already enforced

The context bound predates this task and is enforced in three places rather than one: `DenseSession` refuses
an `Append` past the limit and refuses a `Forward` whose tokens would cross it, and `TextGenerator` stops with
`GenerationStop.ContextFull` before asking. The key/value state is dimensioned from that bound and the model
shape, never from a nominal constant, and the plan's account of it matches the allocation.

Context was therefore not the gap. Two things were.

## Added: the budget is applied, not only computed

`ModelPlanner.Plan` computed `ExceedsBudget` and the console printed it, but nothing refused a session that
could not fit: the flag was advice. `InferenceOptions` now carries `BudgetBytes`, and `DenseSession` checks the
plan **before its first large allocation** and throws with the three accounts named:

```
The session needs 8117944320 bytes and the budget is 4294967296:
weights 4683073536, state 301989888, workspace 3132866560.
```

The refusal is worth more than the warning it replaces: a session that starts and dies later on an allocation
it cannot honour has already spent the memory it was told it could not have.

## Added: cancellation, bounded by design and not by luck

`CancellationToken` now runs the whole way from `TextGenerator` and `DenseSession` into `DenseForward` and into
the projection itself, where the token is observed **between the pieces of work the row split produces** rather
than only between projections.

The distinction is the whole point. A projection of the largest shape in the model is the longest step of a
layer, and during prefill it sees a whole chunk of positions at once: checking only before it means a stop
request arriving inside it waits for the projection to finish. Checking between claims bounds the delay by one
claim — eight rows — whatever the projection's size and whatever the machine. The bound is then a property of
the design rather than a consequence of when the request happened to arrive.

The same contract covers the floating-point path, where the unit is one `RowTile` of sixteen rows, so a host
without AVX2 gets the same bound rather than a worse one.

Two details that would otherwise be silent:

- A worker's cancellation arrives wrapped in the `AggregateException` that `Task.WaitAll` produces. It is
  unwrapped so the caller sees the stop it asked for and not a fault, which is what distinguishes a
  cancellation from a defect everywhere that catches it.
- A request that arrives during the **last** projection is honoured before the call reports success. Without
  that check a cancelled call could return normally, and its caller would have no way to tell.

## Measured

The console handles `SIGINT` as a stop request rather than letting the process be killed, and reports the
interval between the request and the abort. Prompt: 1650 words from
`tools/dw06_prefill_prompt.py`, 2048-token context, 8 threads, real artifact.

| Signal at | `cancel_latency_ms` | Positions committed |
|---|---:|---:|
| 25 s | 36 | 0 |
| 60 s | 5 | 0 |
| 90 s | 5 | **256** |

All three inside prefill, all far under the 2 s the criterion allows. The 90 s row is the most informative of
the three: 256 is exactly `ChunkPositions`, so the first chunk had been committed, the request arrived during
the second, and the session came to rest **on a chunk boundary** with no partial position. That is the commit
rule stated in this report holding on the real model and not only in the unit test that asserts it.

The 25 s figure was taken before the in-projection checkpoints existed, the later two after. They are not a
controlled comparison — the request landed at different points — and are not presented as one. What the three
show is that a request lands and is answered in milliseconds while the prefill is running, and that the
position it leaves behind is a committed one.

## Two things this measurement got wrong first

**`timeout -s INT` did not exercise the handler.** Sending the signal through `timeout` produced exit 130 with
no message: the process was killed, not asked to stop. Sent directly to the process with `kill -INT`, the
handler ran and reported. The first attempt therefore measured the operating system's kill and would have been
recorded as a 0 ms cancellation if only the exit code had been read. The signal has to be delivered to the
process that installed the handler.

**A log read too early looked like a hang.** A 0-byte log 40 s after the signal was read as "the cancellation
was not observed", and it was not: the file was read before the process wrote it. The suspicion was reported
and then withdrawn once the same file showed `cancel_latency_ms: 5`. It is recorded because the wrong reading
was more alarming than the right one and would have sent the next hour into a defect that does not exist.

## Still open in this subtask

- **No sustained swap** is not yet measured. It needs a workload that presses the budget on a machine whose
  resident set does not fit, which is a measurement on a constrained host and not a code change.
- The concurrent workload and the frozen C# corpus that the plan requires before DW06.8's quality gate are the
  owner's to freeze; neither exists yet, so the pass-rate gate is not run.

## What is not claimed

The latency figures are two samples on one host at one context length. They are not a distribution and the
2 s margin is not a measurement of the worst case everywhere; what makes the margin credible is the structural
bound above, and the samples are consistent with it rather than a substitute for it.
