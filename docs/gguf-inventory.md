# GGUF Inventory

Descriptor-level inventory of a GGUF container: header, metadata and tensor directory, without decoding
weights or executing anything. It feeds the capability matrix (DS02) and the inventory command (DS01).

```sh
dotnet run --project src/Satori.Console -c Release -- inventory <file.gguf> [--json]
```

The text report prints file size, version, architecture, tensor and metadata counts, alignment, data
offset, total tensor bytes, profile compatibility, unsupported tensor types, every metadata key with its
declared type and decoded scalar value or array shape, and every tensor with its type, dimensions,
declared offset, computed byte length and support status. `--json` emits the same data for reproducible
diffs. A file that is not GGUF, or whose metadata declares an unknown value type, fails with a readable
message and exit code 1.

## Relationship to the Validated Reader

`GgufFile` validates a container against the execution profile (`docs/gguf-subset-v1.md`) and rejects any
tensor type outside `F32`, `F16`, `Q4_0`, `Q8_0`. `GgufInventory` reuses the same low-level primitives
(`MappedWeights`, `GgufCursor`) but is deliberately tolerant: it reports an unsupported tensor type as
`unsupported` with a reason instead of rejecting the file, so a candidate artifact's declared content can
be inspected before any weight download. On a file both accept, the two must not disagree; the tensor
name, type, dimensions, absolute offset, byte length, alignment and data offset are checked against the
validated catalogue in `GgufInventoryTests`.

The inventory never reads tensor payloads. Byte lengths are computed only for catalogued types whose
first dimension is a multiple of the block size; otherwise the length is reported as unknown and the
total is marked partial. Private fork types (for example Prism PTQ1_0/PQ2_0) are not catalogued: their
geometry must be verified against their own reference before being added to `GgufTypes`.
