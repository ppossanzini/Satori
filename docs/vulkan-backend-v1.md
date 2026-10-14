# Vulkan backend contract

Status: v2. The capability gate now admits optional device features behind explicit queries, each with a verified fallback, because measurement shows the bottleneck is kernel cost and per-dispatch latency rather than bandwidth. Evidence: [minimal path](vulkan-minimal-path.md), [Q8 MatVec](vulkan-q8-matvec.md). CPU remains the verified reference and the default. Metal, ROCm, graphics/surfaces, asynchronous public APIs and multiple queues stay outside this contract.

## Stack and boundaries

| Slot | Decision |
|---|---|
| Managed runtime | Existing .NET 10 / C# 14, x64 |
| Binding | Silk.NET.Vulkan **2.23.0**, with matching Extensions.EXT for validation diagnostics; exact NuGet versions; native Vulkan loader/ICD supplied by the host |
| Backend project | Satori.Backends.Vulkan; references Core; unsafe code confined to this project |
| Shader source | GLSL 450 compute, entry point main |
| Shader compiler | glslangValidator **16.6.0**, offline build step |
| Binary target | Vulkan **1.1**, SPIR-V **1.3** |
| Shader validation | spirv-val **2026.4**, target environment vulkan1.1 |
| Runtime validation | VK_LAYER_KHRONOS_validation plus VK_EXT_debug_utils when explicitly enabled |
| Submission | One compute-capable queue; several dispatches per command buffer, one submission and one wait per append. Legacy core barriers and fences remain the required path; synchronization2 and timeline semaphores only behind the capability gate |

The [published Silk.NET package](https://www.nuget.org/packages/Silk.NET.Vulkan/2.23.0) and [tagged binding source](https://github.com/dotnet/Silk.NET/tree/v2.23.0) are the dependency baseline. Direct handwritten P/Invoke is rejected to avoid maintaining Vulkan ABI declarations. Silk windowing packages, shader compilation at application startup, allocator libraries and optional GPU math extensions are unnecessary for the initial path. Silk 2.x maintenance is limited while its project develops 3.x; upgrading the binding requires a recorded compatibility review, not a floating package version.

Core and Formats.Gguf acquire no native dependency. The console/benchmark composition layer chooses a backend later; the backend accepts compressed byte data and FP32 vectors, without parsing GGUF or owning the CPU model. A mapped tensor is copied during its protected read callback; no pointer into the CPU mapping is retained by Vulkan.

## Capability gate and device selection

Require loader and physical-device API version >= 1.1; a queue family with COMPUTE and at least one queue; FP32 shader arithmetic, 32-bit integer operations and storage buffers. Optional features are queried before use and never assumed; each has a verified fallback, and the effective set is recorded because it participates in the planner cache identity.

| Optional feature | Use | Fallback when absent |
|---|---|---|
| 8-bit storage with int8 arithmetic (`SPV_KHR_8bit_storage`) | direct Q8_0 weight reads and integer accumulation | byte extraction from uint words with shifts, masks and sign extension |
| Subgroup operations | per-row reductions and ballots | shared-memory tree reduction over the declared workgroup |
| synchronization2 and timeline semaphores | cheaper barrier and fence bookkeeping | legacy core barriers and fences |
| FP16 arithmetic | scale and activation paths | FP32 with half decoding |
| Timestamp queries | kernel duration attribution | host timing only, reported as such |

FP64 arithmetic, buffer device address, multiple queues and a public asynchronous API remain out of scope.

Each kernel declares its own local size, descriptor count and push-constant size in its manifest. The contract no longer fixes local size (64,1,1) or a 16-byte push-constant budget; per-shape variants are expected to change them. Check workgroup size/invocation limits, descriptor counts, push-constant size, storage-buffer range, allocation count/size and dispatch group count against the actual workload before allocation or dispatch. Zero elements are a successful no-op without zero-sized Vulkan resources. Oversized requests are rejected before submission; no silent truncation. Chunking large matrices is a later kernel decision. Capability limits are queried rather than inferred from GPU names. See [Vulkan limits](https://docs.vulkan.org/spec/latest/chapters/limits.html).

`VulkanOptions` centralizes `DeviceIndex = null` and `Validation = false`. Negative indices are invalid. Default selection is the first eligible physical device in the loader enumeration, followed by its first compute-capable queue family and queue zero. An explicit index selects only that device; unsupported selection does not silently switch devices. Record enumeration index, device UUID, name, vendor/device IDs, API/driver versions, queue family, limits and memory types. This is deterministic within the recorded enumeration, not a performance ranking or persistent device identity. Software devices can run diagnostics but do not establish hardware acceleration. Queue requirements follow [devices and queues](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html).

Timestamp queries are optional: record queue timestampValidBits and timestampPeriod. If unsupported, GPU kernel time is unavailable, never synthesized from host latency. Mask timestamp differences to timestampValidBits and convert with timestampPeriod; handle unavailable query results explicitly. Later reports must distinguish timestamp kernel duration from upload, submission, wait, readback and end-to-end latency.

## Shader build and ABI

Compile explicitly with `glslangValidator -V --target-env vulkan1.1 -S comp -e main -o kernel.spv kernel.comp`, then `spirv-val --target-env vulkan1.1 kernel.spv`. These options are defined by the [glslang compiler source](https://github.com/KhronosGroup/glslang/blob/16.6.0/StandAlone/StandAlone.cpp). Store shader source, compiled embedded resource and a manifest with compiler/validator versions, commands, source and binary SHA-256, entry point, local size, descriptors and push-constant layout. Normal .NET builds and applications consume the embedded binary without requiring a compiler; regeneration is an explicit development step.

Check binary length divisible by four, SPIR-V magic/version and manifest hash before creating a module. Set 0 uses std430 storage buffers: binding 0 input, binding 1 output for the minimal path. Push constant at byte 0 is an unsigned 32-bit element count; remaining bytes are absent unless the kernel manifest declares them. Dispatch ceil(count/64) workgroups and guard the final partial group. No out-of-bounds vector loads or output aliasing.

Quantized MatVec later uses a separate declared layout, initially Q8_0 from the verification corpus. Packed bytes are transported in uint words with explicit shifts/masks and signed extension; FP16 scales are converted to FP32 without requiring FP16 arithmetic. Padding and tensor offsets are specified and verified with the kernel, never obtained by casting arbitrary mapped addresses. This contract adds no Q8 shader or numerical tolerance today.

## Memory and ownership

`VulkanContext` owns the binding/loader lifetime, instance, optional debug messenger, device, queue, command pool, command buffer, fence, pipelines, descriptor resources and registered buffers. Each `VulkanBuffer` owns one VkBuffer and one dedicated VkDeviceMemory allocation initially. All operations and disposal use the context's single lock; objects from different contexts cannot be mixed. Native handles and persistent mapped spans remain internal. Keep the debug callback delegate rooted until its messenger is destroyed; never let managed exceptions cross the native callback. Explicit disposal is idempotent; the context disposes remaining children and invalidates outstanding wrappers. No finalizer is relied on for correctness.

Choose allocations from memoryTypeBits returned for each buffer, respecting VkMemoryRequirements size/alignment. Resident compute buffers use DEVICE_LOCAL and STORAGE_BUFFER plus required transfer flags; upload/readback staging uses HOST_VISIBLE, preferring HOST_COHERENT. Use the same staging path on integrated and discrete GPUs initially. Account for actual allocation size, staging and resident bytes separately; device-local memory on an integrated GPU is not a separate RAM budget. Buffer rules follow [resource creation](https://docs.vulkan.org/spec/latest/chapters/resources.html).

Copy spans synchronously into owned staging memory; callers may reuse input after return. Non-coherent upload ranges are flushed and readback ranges invalidated. Align ranges to nonCoherentAtomSize and remain within the allocation, using the allocation-end exception or whole mapped allocation where needed. Coherent memory removes cache maintenance, not execution ordering. Never map/reuse/free memory while a GPU operation accesses it. See [memory allocation](https://docs.vulkan.org/spec/latest/chapters/memory.html).

After in-flight work completes, release command/descriptor resources and pipelines, destroy buffers before freeing their memory, then device, debug messenger, instance and binding. Partial initialization unwinds only successfully created handles in dependency order. Dispose racing an operation waits on the same lock; subsequent calls throw ObjectDisposedException. Persistent weight buffers are reused across dispatches, not uploaded per token.

## Submission and synchronization

Record upload copy, compute dispatch and readback copy on the same queue family; buffers use exclusive sharing, so no ownership transfer is required. Use vkQueueSubmit with a fence and vkCmdPipelineBarrier (legacy core API), not the Vulkan 1.3 synchronization2 examples verbatim.

| Dependency | Source stage/access | Destination stage/access |
|---|---|---|
| Uploaded input becomes shader input | TRANSFER / TRANSFER_WRITE | COMPUTE_SHADER / SHADER_READ |
| Shader output becomes readback source | COMPUTE_SHADER / SHADER_WRITE | TRANSFER / TRANSFER_READ |
| Readback becomes CPU-readable | TRANSFER / TRANSFER_WRITE | HOST / HOST_READ |
| Resident intermediate feeds next dispatch | COMPUTE_SHADER / SHADER_WRITE | COMPUTE_SHADER / SHADER_READ |

Host upload writes/flush complete before submission. After submission, wait for the fence, invalidate non-coherent readback, then copy into the caller's output. Fence completion and memory dependencies are distinct; [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html) show the corresponding hazards. Do not reset fence/command buffer or change descriptors before completion; [synchronization rules](https://docs.vulkan.org/spec/latest/chapters/synchronization.html) govern reuse.

The minimal synchronous path waits until completion; it offers no cancellation or bounded latency for a hung driver. Dispose waits for pending work before releasing resources. A host timeout must never justify freeing in-flight memory. Avoid per-dispatch DeviceWaitIdle; reserve it for teardown when necessary. An asynchronous API or watchdog requires a new lifetime contract.

## Initialization, errors and CPU fallback

Factory shape: `VulkanContext.TryCreate(options, out context, out availability)` returns false with null context for known unavailability, after partial cleanup. Availability contains a reason, operation and native result where present. Distinguish LoaderUnavailable, NoDevice, UnsupportedDevice, ValidationUnavailable, InitializationFailed and DeviceMemoryUnavailable. Success returns Ready with selected capabilities. TryCreate does not catch all exceptions; invalid options, corrupt embedded shader, host OOM, programming errors and unexpected failures propagate.

| Condition | Required outcome |
|---|---|
| Loader/ICD absent, no device, unsupported API/limits, known initialization rejection | No live partial context; structured unavailability |
| Validation requested but layer/debug-utils absent | ValidationUnavailable; validation run fails, never claims a clean CPU result |
| Validation error callback | Capture message/VUID, mark the operation failed; do not throw across native callback or hide it with fallback |
| Allocation failure before submission | Release partial resources; report device-memory failure; CPU retry allowed only by caller policy |
| Device lost during submission/wait | Context becomes Faulted, never reused; no partial readback committed; perform permitted lost-device teardown |
| Unrelated exception or invalid application input | Propagate; no automatic retry |

Caller policy is explicit: Cpu skips Vulkan; Auto may use CPU for the known unavailability cases and reports the reason; VulkanRequired fails if Vulkan cannot execute. Future generic model composition must consume this policy; automatic fallback is initialization-only. Required GPU verification cannot pass by falling back to CPU.

Output is committed only after successful wait/readback; failures leave caller output unchanged. Before model integration, GPU failure can retry the whole stateless operation on CPU. For stateful offload, recovery must reconstruct a fresh CPU session from committed token history and preserved sampling state; never continue a partially updated KV cache or emit the same token twice. Device loss during execution is therefore not silently recoverable inside the low-level backend.

## Verification gates for implementation

| Case | Acceptance evidence |
|---|---|
| Known FP32 transform and tail | Hardware output equals CPU for empty, 1, 63, 64, 65 and multi-group vectors; sentinel output unchanged on rejection |
| Missing loader/device, incompatible selection | Structured reason; Auto CPU works; VulkanRequired fails |
| Upload/readback reuse | Repeated distinct inputs produce correct outputs with no stale data |
| Lifetime | Child/context disposal, repeated disposal, concurrent calls/dispose, partial construction failure; no invalid handle use |
| Validation | Actual hardware run with Khronos layer and synchronization validation enabled; zero validation errors, warnings reviewed |
| Shader artifact | Correct manifest/hash, successful spirv-val; malformed/mismatched binary rejected before module creation |
| Measurement | Report device/driver, workload, actual backend, transfers/allocations and host timings; GPU timestamps only when supported |
| Capability fallback | Every optional feature exercised with the feature present and absent; the recorded capability set is part of the planner cache identity and a capability change invalidates it |
| Attribution | The resident path can emit GPU timestamps per dispatch group and report kernel window, barrier cost and dispatch cost separately from host phases |

The hardware validation job sets process-local `VK_VALIDATION_VALIDATE_SYNC=1` while requesting the Khronos layer; record this setting and the layer version. This enables synchronization checks alongside core checks, as documented in [LunarG SyncVal usage](https://vulkan.lunarg.com/doc/view/latest/linux/synchronization_usage.html). Host accesses are not tracked by SyncVal, so flush/invalidate and host lifetime still require result/lifetime verification.

Tests focus on GPU results, synchronization and ownership. No tests for obvious project bootstrap or option plumbing. An unavailable GPU/layer is a reported missing gate, not a successful hardware test.

## Local feasibility evidence — 12 October 2026

Read-only probing outside the sandbox enumerated Intel Arc Graphics (MTL), integrated GPU, vendor/device 8086:7d55, device UUID 8680557d-0800-0000-0002-000000000000, Vulkan API 1.4.354, Mesa 26.2.4-arch1.1; loader 1.4.363. The sandbox probe returned ERROR_INITIALIZATION_FAILED and exposed no /dev/dri; this does not establish host GPU absence.

Observed: compute-capable queue with count 1; maxComputeWorkGroupInvocations 1024; maxStorageBufferRange 4294967295; maxPushConstantsSize 256; storage offset alignment 4; nonCoherentAtomSize 64. Device-local host-visible coherent memory exists. Queue timestamps have 36 valid bits and timestampPeriod 52.0833 ns. These observations support the planned capability gate; runtime creation/allocation checks remain mandatory.

Installed glslangValidator 16.6.0 compiled a temporary GLSL 450 transform (`y = 2*x+1`, local size 64, two storage buffers, uint count) to SPIR-V 1.3, 1432 bytes; spirv-val 2026.4 accepted it for vulkan1.1. Binary SHA-256: 761040dcb5bf01558ef237aba602e4c8e9d9cb417ef31e7d627eb4df25fd39ad. This is compilation evidence only; temporary source/binary were removed. No runtime package was added and no GPU dispatch was claimed.

The host enumerated only MESA anti-lag and device-select layers: Khronos validation is not installed. Install/enable the host validation-layer package before closing the minimal-path hardware gate. Compiler validation does not replace runtime validation. No driver, system package or machine configuration was changed here.
