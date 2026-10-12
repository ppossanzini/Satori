# Stories15M Vulkan offload

The model can execute all 43 Q8_0 MatVec tensors on Vulkan: seven projections per layer and the output projection. The embedding row lookup, RMSNorm, RoPE, attention, SwiGLU, residuals and FP32 KV cache remain on CPU. Each MatVec uploads its input and reads its output; intermediate activations do not stay on GPU. No matrix is split between devices.

`StoriesModel` accepts an optional `IQuantizedMatrixFactory`, a native-independent Core contract. It owns the resulting executions and releases them before the mapped file, including partial construction failures. Formats.Gguf depends only on Core. Vulkan depends on Core and Formats.Gguf for the composition owner `VulkanStoriesRuntime`; the console composes them. The context outlives its model. A generator can borrow that model without disposing it. Explicit scalar execution bypasses offload.

## Selection and failures

`StoriesBackendOptions` declares `Mode` (default Cpu) and optional `VulkanOptions` (device selection and validation). Cpu never initializes Vulkan. Auto selects Vulkan when initialization and resident weight creation succeed, otherwise uses CPU for known availability, unsupported-capability or device-memory failures and exposes the reason. VulkanRequired propagates failure. Auto now also applies the [measured planner](stories15m-planner.md) after successful initialization. CPU remains the console default because this GPU baseline is slower.

```sh
dotnet run --project src/Satori.Console -c Release -- generate \
  .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12 --backend vulkan
```

Use `--backend cpu` or `--backend auto`; `--device N` selects an enumerated GPU explicitly. Sampling flags remain available. Backend and fallback reason go to stderr, continuation text to stdout.

Execution errors propagate in every mode. There is no automatic retry after partial sequence execution. A failed append clears the entire cache, resets Count to zero and blocks further appends until Reset; a faulted GPU context still cannot execute after Reset. Recovery requires a new CPU model/session and replay of committed tokens with preserved sampling state. Such automatic replay is not implemented. Emitted text is never silently regenerated. Stateless Forward retains no cache between calls; low-level executions commit output only on success.

## Verification

Eleven GPU tests pass on Intel Arc MTL with Khronos core and synchronization validation. Three pinned reference prompts match all 15 intermediate states and all 32000 logits against the independent FP64 exact-weight oracle (`5e-4 + 5e-5 * abs(reference)`, RMS at most `1e-4`). Incremental execution, reset and twelve greedy tokens per prompt also match. A separate 128-token CPU/GPU comparison reaches the cache limit without advancing beyond it; reset and streaming generation work, and disposing a borrowed generator leaves its model usable. Scalar bypass submits no GPU work. No Vulkan validation errors occurred, including disposal.

The CPU suite passes 116 tests, including injected execution failure after partial KV updates and partial resident-construction failure. Cache continuation is blocked after failure; explicit reset produces the same logits as a fresh CPU session, and all owned executions are released. One initial allocation assertion reported 7752 bytes when suites were launched concurrently; the isolated check and full CPU suite run alone passed unchanged. The cause was not established; the zero-allocation measurement below is an independent warmed sequential observation.

Five child-process cases verify CPU with no ICD, Auto with no ICD, Required with no ICD, Auto with an invalid device and Required with an invalid device. Successful fallback emits the expected continuation; required failures emit no text. [Results](diagnostics/vulkan-offload-2026-10-12/negative-cases.json). [Required and Auto console runs on hardware](diagnostics/vulkan-offload-2026-10-12/cli-hardware.json) also emit the expected greedy continuation. Actual GPU device loss and device-memory exhaustion were not induced; automatic recovery is not claimed.

## Measurements

[Raw performance report](benchmarks/vulkan-offload-2026-10-12/offload-benchmark.json), with validation disabled; [diagnostic report](benchmarks/vulkan-offload-2026-10-12/offload-validation.json), with validation enabled. Both compare CPU AVX2 and Vulkan sequentially on the same pinned artifact, .NET 10.0.12, Intel Arc MTL/Mesa 26.2.4. Three warmup runs precede seven measured runs. Each run appends prompt tokens `[1,9038,2501]`, then eight teacher-forced token 9038 appends. This measures session inference, excluding tokenization, sampling and text decoding. Reset and caller array allocation are outside the timed interval. Timestamp queries are enabled; their overhead remains in host latency.

| Median | CPU AVX2 | Vulkan |
|---|---:|---:|
| Three-token prefill | 12.408 ms | 450.313 ms |
| Decode append (56 samples) | 3.893 ms | 154.032 ms |
| Complete 11-token run | 42.050 ms | 1680.553 ms |
| Cold model/backend/weights initialization (one observation) | 69.516 ms | 256.321 ms |

Maximum CPU/GPU logit difference over eleven checked prefixes: `9.656e-6`; maximum RMS `4.663e-6`. All measured warmed session runs allocate zero managed bytes. Vulkan is substantially slower on this workload; these observations establish a usable explicit offload path, not acceleration. Measured selection and cache policy are documented in the [planner](stories15m-planner.md).

The 43 resident matrices occupy 16,137,216 bytes of device allocation for compressed weights. Total matrix buffer allocation is 16,658,176 bytes, including 521KB of input/output/staging buffers. There are 215 dedicated live buffer allocations. Each append transfers 260,480 vector bytes across its 43 dispatches, excluding initial weight upload. The CPU KV payload remains 1,769,472 bytes; the 26,671,328-byte GGUF stays mapped for embedding and CPU execution. These figures describe separate allocations/mappings, not total process RSS or an extra physical VRAM budget on this integrated GPU.

Cumulative report statistics include the eleven-prefix correctness run, three warmup runs and seven measured runs: 5203 dispatches and 31,518,080 transferred bytes. GPU timestamps measure dispatch windows; they exclude CPU operators, host waits and vector transfers and must not be presented as whole-model latency.

## Reproduction

Build Release. For hardware tests and the diagnostic report, stage the pinned local Khronos layer using the [minimal-path recipe](vulkan-minimal-path.md#reproduction), with `.temp/t30` as the task directory. Then:

```sh
VK_LAYER_PATH="$PWD/.temp/t30/layer/usr/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$PWD/.temp/t30/layer/usr/lib" VK_VALIDATION_VALIDATE_SYNC=1 \
dotnet run --project validation/Satori.Vulkan.Validation -c Release --no-build -- \
  --offload .temp/gguf-corpus/stories15M-q8_0.gguf

dotnet run --project validation/Satori.Vulkan.Validation -c Release --no-build -- \
  --offload-benchmark .temp/gguf-corpus/stories15M-q8_0.gguf
```

Run performance measurements alone, with validation layers disabled. Reports are regenerated under `reports/vulkan`. The task-local validation package is removed after verification and must be restaged for later diagnostics.
