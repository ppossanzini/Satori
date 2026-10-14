# Incremental Session

Status: DS09. The per-context incremental session defined by the DS03 runtime contract, verified against
full recomputation.

## Contract

`IInferenceSession` (Core.Model):

| Member | Meaning |
|---|---|
| `Model` | The immutable definition the session executes. |
| `ContextLength` | Maximum number of positions this session accepts. |
| `Position` | Number of committed positions. |
| `Reset()` | Discards all positions; the session returns to its initial state. |
| `Forward(ReadOnlySpan<int> tokens, Span<float> logits)` | Appends tokens, advances `Position`, writes the logits of the last position. |

`DenseSession` implements it. Construction rejects a definition with capability gaps (`NotSupportedException`)
or without a derived shape (`InvalidOperationException`), so a non-executable model never reaches execution.

## State

The session owns one F32 key and one F32 value per KV head and head dimension for every layer and position:

```
StateBytes = Layers * 2 * HeadCountKv * HeadDim * ContextLength * 4
```

which is exactly `ModelPlanner.Plan(...).StateBytes`. The context length is a caller choice (the plan's
context); the head geometry comes from the model. A state that would exceed Int32 floats is rejected rather
than allocated. The session is not thread-safe: one `Forward` at a time, per the contract.

## One Implementation of the Layer Step

`DenseForward` exposes its per-position step (`Step`) and final normalization (`Final`) internally. The
whole-sequence pass keeps its key/value prefix in a local buffer; a session keeps it in persistent state.
Both call the same `Step`, so the layer arithmetic has a single implementation and incremental execution
and full recomputation agree by construction — the contract's stated invariant. Every operator is
`TransformerOps` and every projection is the shared quantized MatVec.

## Batched Prefill

`Forward` does not call `Append` per token. It walks the prompt in chunks of 256 positions and runs each
chunk through `DenseForward.LayerBatch`, which executes the same operator sequence as `Step` but calls every
projection once for the whole chunk. A projection that sees all positions at once decodes each weight block
once instead of once per position, and that decode is what a prompt pays for: measured on the 7B Q4_K_M
artifact, the batched prefill is 2.77x the per-position AVX2 path and 10.2x the scalar path. The prefill
never projects the output layer per position — only the last prompt position needs logits — so the
vocabulary-sized projection runs once.

`InferenceOptions` fixes the thread budget once at construction (default 8, the protocol's limit) and the
session never reads a process-wide setting. `DenseBatchScratch` holds the chunk's activations, sized from
the shape and released with the session. Prefill and decode agree exactly, not merely closely:
`BatchedPrefillMatchesSequentialAppend` requires bit-identical logits between a 200-position batched prefill
and 200 `Append` calls.

## Errors and Limits

| Condition | Result |
|---|---|
| Token outside the vocabulary | `ArgumentOutOfRangeException` |
| Appending past `ContextLength` (single token) | `InvalidOperationException` |
| Batch that would exceed `ContextLength` | `ArgumentOutOfRangeException`; no partial append |
| Logits buffer of the wrong size | `ArgumentException` |
| Use after `Dispose` | `ObjectDisposedException`; `Dispose` is idempotent |
| Definition with capability gaps / no shape | `NotSupportedException` / `InvalidOperationException` |

Limits are errors, never silent truncation.

## Verification

Offline (`DenseSessionTests`, synthetic weights, no corpus):

- incremental logits equal full recomputation for a prompt and for a prompt plus one appended token, on
  both a qwen2-like (QKV bias) and a qwen3-like (Q/K norm, tied embeddings) shape;
- `Reset` reproduces the initial state and identical logits;
- `StateBytes` equals the plan; the context limit, invalid tokens and the logits-buffer size are enforced;
- `Dispose` is idempotent and rejects further use; capability and shape guards fire.

Model integration (`QwenForwardTests.IncrementalSessionMatchesFullForward`, requires
`python tools/download_qwen_models.py`):

- the session over the reference tokens reaches the same independent-reference argmax as the
  whole-sequence pass, on Qwen2.5-Coder 1.5B and Qwen3 4B;
- appending one token equals recomputing the prefix through `DenseForward`, within `1e-3 + 1e-3 * |ref|`
  (FP32 accumulation, not the float64 reference).

## Test Isolation

`tests/Satori.Numerics.Tests/xunit.runner.json` disables test-collection parallelization. Several tests
assert exact allocation counts with `GC.GetAllocatedBytesForCurrentThread`; running them concurrently with
the model-integration tests (which load multi-gigabyte weights) made those assertions flaky. Serial
execution keeps them meaningful.
