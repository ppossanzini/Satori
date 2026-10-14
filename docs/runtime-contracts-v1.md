# Runtime and Session Contracts v1

Status: specification requiring alignment in P04 after candidate selection. [The product plan](../TASKS.md) governs scope. Only the selected family and memory policy must be implemented. Registry breadth, both state families and streamed sources are conditional on that selection.

## Principles

- Architecture, tokenizer, codec, transformation, backend and memory policy are separate capabilities.
  No runtime type is named after a model, publisher or recipe when it expresses a generic contract.
- Every dimension, epsilon, rotation, tied-weight relation and workspace size is derived from validated
  metadata, never from a nominal constant and never from `hidden / heads`.
- Weights stay compressed. Only bounded per-block/tile scratch is allowed; no permanent full-model
  F32/FP16 expansion.
- The model definition is immutable and shareable. Sessions are per-context and private.

## 1. Immutable Model Definition

`ModelDefinition` is built once from a validated descriptor and never mutated. It contains descriptors
only, not weights.

```
sealed class ModelDefinition
{
    string ArchitectureId;                 // e.g. "qwen2", "qwen35moe" (data, not a type name)
    IReadOnlyDictionary<string, ModelTensor> Tensors;
    IReadOnlyDictionary<string, object> Config;    // typed metadata projection
    TokenizerDescriptor Tokenizer;
    IReadOnlyList<TransformationDescriptor> Transformations;
    IReadOnlyList<CapabilityGap> Gaps;             // empty when the definition is executable
    long WeightBytes;                              // compressed payload total
}
```

`ModelTensor` carries `Name`, `TensorFormat` (catalogue id + block geometry), `Dimensions` (declared,
verbatim), `PayloadRange` (offset/length from the file) and an optional `TensorRole` tag (`Embedding`,
`Output`, `Norm`, `AttentionQ`, `MoeRouter`, `SsmAlpha`, …). Roles are optional hints resolved by the
architecture adapter; absent roles are derived from names by the adapter, never by the core.

`Config` is a typed projection of the metadata keys the architecture needs (block count, embedding
length, head count, head count kv, **explicit key/value length when present**, rope base and dimension
count, rope section layout, norm epsilon, expert count/used count, SSM parameters, tied-weights flag,
activation kind). A missing required key is a `CapabilityGap`, not a default.

`Token` is an `int`. No runtime contract is specific to a tokenizer vocabulary size.

### Two topologies, no nominal dimensions

Candidate shapes below describe possible requirements. Only the selected shape is required; dimensions must not be hard-coded:

| Topology | Source | Distinguishing descriptors |
|---|---|---|
| Dense grouped-query | qwen2/qwen3 | `attention.head_count_kv < head_count`; qwen3 with explicit `attention.key_length = 128` ≠ `embedding_length / head_count` |
| Hybrid MoE | qwen35moe | `expert_count`, `expert_used_count`, `ssm.*`, `full_attention_interval`, partial `rope.dimension_count < key_length` |

The adapter reads the discriminating keys from `Config`; the session contract does not change between
them. Head dimension comes from `attention.key_length`/`value_length` when declared, otherwise from
`embedding_length / head_count`.

## 2. Capability Check

Before any weight is read for execution, the definition is checked against the running build.

```
sealed record CapabilityGap(string Kind, string Detail, string Reason);
// Kind: "architecture" | "tensor-format" | "operator" | "tokenizer" | "transformation" | "backend"

CapabilityReport Check(ModelDefinition definition, BackendCapabilities backend);
sealed record CapabilityReport(IReadOnlyList<CapabilityGap> Gaps) { bool IsExecutable => Gaps.Count == 0; }
```

Rules:

- Every tensor format in `Tensors` must be decodable, independent of whether that tensor is used.
- Every operator named by the adapter must exist for the selected backend.
- Unknown/private tensor types are `tensor-format` gaps, explicitly named by id.
- A non-executable definition is rejected before generation. Rejection never silently substitutes a
  format or a fallback that changes results.

## 3. Weight Sources and Ownership

Weights are read through a scoped source. The core never holds a raw pointer or span beyond a lease.

```
interface IWeightSource : IDisposable
{
    long Length { get; }
    IDisposable Lease(string tensorName, long offset, int length, out ReadOnlySpan<byte> window);
}
```

- A `Lease` keeps the backing bytes valid until it is disposed; disposing a lease after the source is
  disposed must still be safe (mapping-style deferred unmap).
- Resident source: a validated memory-mapped file (`MappedWeights`); reused as-is.
- Streamed source: a bounded reader that fetches ranges on demand (conditional on P03). Same interface.
- Shared/tied tensors: two names may resolve to the same `PayloadRange`. The source returns the same
  bytes; the definition records the alias so the model is not counted or loaded twice.

## 4. Session, State and Workspace

```
interface IInferenceSession : IDisposable
{
    ModelDefinition Model { get; }
    int ContextLength { get; }         // maximum positions this session accepts
    int Position { get; }              // number of committed positions
    void Reset();
    void Forward(ReadOnlySpan<int> tokens, Span<float> logits);   // appends, returns last-position logits
}
```

- `Forward` appends tokens, advances `Position`, and writes logits for the last position. Calling it past
  `ContextLength` is an error, not a silent truncation.
- State is dimensioned from the definition and `ContextLength`, in bytes, at construction.
- Full recomputation and incremental execution must agree: `Forward(prompt + token)` equals
  `Forward(prompt)` followed by `Forward([token])` within the operator tolerances.

State layouts (two families, selected by the adapter, both generic):

```
sealed record KvStateLayout(int Layers, int KvHeads, int HeadDim, int Slots, ElementFormat Format);
sealed record RecurrentStateLayout(int Layers, int ConvKernel, int StateSize, int Groups,
                                   int TimeStepRank, int InnerSize, ElementFormat Format);
```

Layers without attention use `RecurrentStateLayout`; layers with full attention use `KvStateLayout`.
`full_attention_interval` decides which layers get which; the session allocates only the layouts present.

### Workspace

Scratch is caller-visible in the contract but bounded: per-layer activation buffers, per-block decode
scratch and backend staging, sized at construction from the largest layer. No buffer grows with the number
of tokens beyond KV/recurrent state.

## 5. Memory Budget and Domains

```
enum MemoryDomain { Unified, Discrete }
sealed record Allocation(string Name, long Bytes, MemoryDomain Domain, bool Resident);
sealed class MemoryBudget
{
    long BudgetBytes;                     // e.g. 16 GiB
    IReadOnlyList<Allocation> Allocations;
    long Committed(MemoryDomain domain);
}
```

Rules:

- Every allocation is charged to exactly one `MemoryDomain`.
- On an integrated GPU, host buffers and GPU buffers that alias the same physical pages share a
  `Unified` domain and are counted once, not once per side.
- On a discrete GPU, `Discrete` holds device-resident buffers; host copies are counted in `Unified`.
- Budget categories are disjoint: `weights-resident`, `kv/recurrent-state`, `scratch`, `prefetch`,
  `backend-staging`. Nothing is counted in two categories.
- Exceeding `BudgetBytes` is reported with the components responsible; it is not resolved by paging the
  process into swap and calling it compliant.

## 6. Residency and Prefetch

```
interface IResidencyController
{
    bool EnsureResident(string tensorName, CancellationToken token);   // blocks until available or throws
    void Release(string tensorName);                                   // advisory; only unpinned entries leave
}
```

- A resident set is chosen at construction from the budget and never needs `EnsureResident`.
- A streamed set is fetched on demand; prefetch (conditional on P03) is bounded by a byte budget and never evicts an
  entry that is currently leased.
- Residency decisions are observable (hit/miss/fetch counts) for the P06 report.

## 7. Cancellation

- Constructors, `Forward`, `EnsureResident` and prefetch accept a `CancellationToken`.
- Cancellation leaves the model definition and any shared weights valid; a cancelled session is reset to
  a consistent state (position and state either both advanced or both unchanged) and may be reused or
  disposed.
- A cancelled or failed load must never become a cache hit (P06).

## 8. Error and Lifetime Invariants

| Condition | Contract |
|---|---|
| Missing capability | `NotSupportedException` with a `CapabilityGap` detail, before reading weights |
| Malformed/truncated container | `InvalidDataException`, before execution |
| Unknown/private tensor type | `NotSupportedException` naming the id |
| Read/exceed context | `ArgumentOutOfRangeException`; never silent truncation |
| Use after `Dispose` (any owner) | `ObjectDisposedException` |
| Dispose during an active `Lease` | Lease stays valid; new leases rejected |
| Failed construction | all partially acquired resources released; no partial session published |
| Scheduler failure (model alias) | the alias must remain a distinct, jointly-managed capability; the structural failure must not be masked by an alternate path |

- `ModelDefinition` is immutable and may be shared by concurrent sessions; it is not disposable and owns
  no weights.
- `IWeightSource` and `IInferenceSession` are `IDisposable`; `Dispose` is idempotent.
- A session is not thread-safe: one `Forward` at a time. Concurrent sessions over one shared source are
  permitted when the source documents thread-safe scoped access (the mapping does).

## 9. Architecture Adapter Registry

```
interface IArchitectureAdapter
{
    string ArchitectureId { get; }
    ModelDefinition Describe(IReadOnlyDictionary<string, object> config,
                             IReadOnlyDictionary<string, ModelTensor> tensors);   // validates, adds Gaps
    IForwardGraph BuildForwardGraph(ModelDefinition definition);                  // operator graph, backend-agnostic
}
```

Adapters are selected by `ArchitectureId` read from metadata. The canonical contract still requires one
named contract; the requirement stays in TASKS.md. Adapters live in the capability registry, not in the
core, and never leak a model name into core interfaces.

## Non-Goals

- No KV persistence to SSD, no training, no server/UI, no model-named runtime types.
- No planner that presumes GPU advantage; backend selection is a later, measured concern (P07).
