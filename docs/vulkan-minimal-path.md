# Minimal Vulkan compute path

`Satori.Backends.Vulkan` implements the [fixed contract](vulkan-backend-v1.md): Silk.NET.Vulkan and its EXT module 2.23.0, one synchronous compute queue, explicit upload/dispatch/readback and owned buffer allocations. Core, GGUF and text generation remain independent of Vulkan. This step executes an FP32 transform; quantized MatVec and model offload are subsequent tasks.

## API and lifetime

```csharp
if (VulkanContext.TryCreate(new(Validation: true), out var context, out var availability))
{
    using (context)
    {
        float[] result = new float[3];
        context!.Transform([1, -2, 0.25f], result); // [3, -3, 1.5]
    }
}
else
{
    // Caller chooses CPU fallback or reports availability.Reason/Diagnostics.
}
```

Transform writes `2*x+1`, requires equally sized non-overlapping vectors, and treats empty input as a no-op. It checks storage range and dispatch limits before submitting. Input is copied to staging; caller output is updated only after successful wait/readback. The operation is synchronous and cannot cancel a hung driver. Device loss or submission/wait failure faults the context, preventing reuse of potentially pending resources.

Four dedicated allocations provide upload, resident input, resident output and readback. Capacity grows on demand; completed scratch is released before replacement to keep four allocations. Smaller subsequent requests reuse the capacity. Allocation requirements and maximum allocation size are checked. Coherent staging is preferred; non-coherent staging flushes/invalidates the entire mapped allocation, avoiding atom alignment errors. All access, growth and disposal share the context lock. Native buffers are internal, owned by the context rather than exposed as public disposable handles.

Info exposes selected UUID, device/driver/API, queue, memory flags and relevant limits. AllocatedBytes reports actual scratch allocation bytes, not heap budget or resident process memory. ValidationMessages remains readable after disposal; ValidationErrorCount counts specification/performance error messages. A GENERAL loader diagnostic accompanying a known initialization rejection is preserved in availability.Diagnostics and permits Auto fallback; validation errors still fail. GENERAL errors after successful initialization also fault operations. Vulkan distinguishes these [debug message types](https://docs.vulkan.org/refpages/latest/refpages/source/VkDebugUtilsMessageTypeFlagBitsEXT.html).

## Shader artifacts

Source, SPIR-V and manifest are in src/Satori.Backends.Vulkan/Shaders. The offline `python3 tools/compile_vulkan_shaders.py` step requires glslang 16.6.0 and SPIRV-Tools 2026.4, compiles for Vulkan 1.1/SPIR-V 1.3 and validates the result. Regeneration was byte-identical, including the manifest. Normal .NET builds use embedded artifacts and require no shader compiler. Runtime checks binary size, magic, version and manifest SHA-256 before creating a module.

## Reproduction

```sh
dotnet restore Satori.sln
dotnet build Satori.sln -c Release --no-restore -m:1
VK_VALIDATION_VALIDATE_SYNC=1 dotnet test tests/Satori.Vulkan.Tests -c Release --no-build --no-restore -m:1
VK_VALIDATION_VALIDATE_SYNC=1 dotnet validation/Satori.Vulkan.Validation/bin/Release/net10.0/Satori.Vulkan.Validation.dll
```

These hardware commands require access to the host GPU and Khronos validation layer. The hardware test project fails when either is missing; no silent skip or CPU substitution. The existing CPU suite remains separately runnable from tests/Satori.Numerics.Tests. The validation executable defaults to required Vulkan with validation; `--auto` permits known-unavailability fallback, `--cpu` avoids native initialization, `--device N` fixes a device, and `--no-validation` is a diagnostic option, not the hardware acceptance gate.

On this Arch host the system layer was absent. Validation used a project-local extraction of the official package, with no system installation. To reproduce that arrangement:

```sh
mkdir -p .temp/t28/layer
curl -L --fail -o .temp/t28/validation.pkg.tar.zst https://archive.archlinux.org/packages/v/vulkan-validation-layers/vulkan-validation-layers-1.4.363.0-1-x86_64.pkg.tar.zst
```

Verify package SHA-256 `904f4eba2e708c9c96579900688a284309cd56dc09b5af3f78c563ffda13df9e` before extracting. Then:

```sh
tar -xf .temp/t28/validation.pkg.tar.zst -C .temp/t28/layer
VK_LAYER_PATH="$PWD/.temp/t28/layer/usr/share/vulkan/explicit_layer.d" LD_LIBRARY_PATH="$PWD/.temp/t28/layer/usr/lib" VK_VALIDATION_VALIDATE_SYNC=1 dotnet test tests/Satori.Vulkan.Tests -c Release --no-build --no-restore -m:1
VK_LAYER_PATH="$PWD/.temp/t28/layer/usr/share/vulkan/explicit_layer.d" LD_LIBRARY_PATH="$PWD/.temp/t28/layer/usr/lib" VK_VALIDATION_VALIDATE_SYNC=1 dotnet validation/Satori.Vulkan.Validation/bin/Release/net10.0/Satori.Vulkan.Validation.dll
```

The local package needs compatible host native dependencies; this recipe is Arch x64-specific. Its original download came from the official Arch package endpoint. Remove only the verified .temp/t28 subtree when finished. The system layer remains uninstalled; future hardware runs must stage or install it again.

The executable writes regenerable reports/vulkan JSON, with actual backend, availability, device/loader/layer versions, transfers, scratch allocation, host samples and managed allocations. GPU kernel time is null because this path does not yet use timestamp queries. Measurements include validation and synchronous transfers; they are diagnostic observations, not an acceleration benchmark or planner input.

## Evidence and limits

Hardware: Intel Arc Graphics (MTL), 8086:7d55, Mesa 26.2.4, Vulkan 1.4.354, .NET 10.0.12. Khronos layer API 1.4.363 with SyncVal enabled. The exact device and shader configuration is included in the published snapshot.

Acceptance covers vectors of 0, 1, 63, 64, 65 and 4097 elements, repeated changing inputs, capacity growth and smaller-buffer reuse. Critical tests also cover unchanged output on dimension/alias rejection, successful concurrent dispatch, disposal racing dispatch, repeated construction/disposal, partial instance cleanup after unsupported selection, and zero managed allocations during warmed dispatch. Disposal validation is included in the error check.

This device chose coherent staging: non-coherent cache-maintenance paths are implemented and reviewed, but not exercised by these runs. Real device loss, host/device OOM and malformed embedded artifacts were not induced. Their error/lifetime branches were reviewed; no production test hooks or deliberate driver reset were added. No quantized shader, inference offload or GPU timestamp performance claim is included.

Final verification: Release build with zero warnings/errors; four Vulkan hardware tests passed with Khronos SyncVal, plus the existing 114 CPU/inference tests. All six transform workloads matched CPU, with zero validation errors and zero warning messages, including disposal. Warm calls measured zero managed allocations; peak scratch allocation for 4097 elements was 65552 bytes. These are this diagnostic workload's observations, not a bound for future MatVec buffers.

Six expected-negative/policy cases passed: CPU with no ICD, Auto fallback with no ICD, required Vulkan with no ICD, required Vulkan without layer, Auto fallback without layer, and explicitly unsupported device index. The ICD/layer settings were isolated to child processes. The missing-ICD run initially exposed a GENERAL loader-error classification bug; the corrected run preserved diagnostics and allowed only the configured CPU fallback.

Published evidence: [hardware report](diagnostics/vulkan-2026-10-12/vulkan-required.json) and [negative cases](diagnostics/vulkan-2026-10-12/negative-cases.json). The source binary and manifest are versioned with the backend. No system package or GPU driver configuration changed; the local layer extraction was removed after verification.
