# Model Descriptor

DS05 implementation of the DS03 contracts: a validated, immutable description of a model produced before
generation.

```sh
dotnet run --project src/Satori.Console -c Release -- describe <model.gguf> [context-length]
```

The command prints the architecture, tensor count and compressed weight bytes, the capability report, the
derived shape and the shape/memory plan at the chosen context against the 12 GiB budget. It reads only the
descriptor region and requires no weights.

## Components (`src/Satori.Core/Model`)

- `ModelTensor` — declared name, format id/name/support, verbatim Int32 dimensions, payload range.
- `ModelConfig` — typed read of scalar metadata addressed by suffix against the architecture prefix
  (`block_count` → `qwen3.block_count`); missing keys are never defaulted.
- `ModelShape` — dimensions derived from metadata: layers, embedding, feed-forward, head count and head
  count KV, head dimension, vocabulary, tied embeddings, RMS epsilon and RoPE base.
- `ModelPlanner` — pure, checked computation of the shape/memory plan: `WeightBytes` (compressed payload),
  `StateBytes` (`layers × 2 × kvHeads × headDim × context × 4` F32 key+value) and `WorkspaceBytes` (a
  documented per-layer upper bound), plus a budget comparison.
- `ModelDefinition` — immutable bundle of the above with the `CapabilityReport`.
- `IArchitectureAdapter` / `ArchitectureAdapters` — capability registry keyed by architecture id.
- `DenseTransformerAdapter` — the qwen2 and qwen3 families.

`src/Satori.Formats.Gguf/GgufModelLoader` builds a `ModelDefinition` from a file through the tolerant
descriptor inventory, so a file with unsupported tensor types or an unknown architecture produces explicit
gaps instead of failing to open.

## Rules

- Head dimension comes from `attention.key_length` when declared; otherwise from
  `embedding_length / head_count` only when that divides evenly.
- Every tensor declared by the container is checked for decodability, independent of use.
- A tensor absent from the directory is a `shape` gap for required tensors; `output.weight` absent means
  the output projection is tied to `token_embd.weight`.
- A declared dimension beyond Int32 is a `shape` gap.
- Plan arithmetic is checked; an overflow is rejected, not truncated.
- No model or publisher name appears in a core interface; adapters are selected by architecture id.

## Limits

- The plan is an estimate from metadata; exact workspace is fixed by the operator implementations (DS07+).
- Only qwen2 and qwen3 adapters exist; hybrid MoE and ternary families are DS24/DS35.
- The descriptor validates shapes and formats; it does not execute and does not decode weights.
