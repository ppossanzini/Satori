# Measured Stories15M execution planner

`--backend auto` now compares the two executable combinations available for the pinned model: CPU automatic Q8_0 (AVX2 when supported, scalar otherwise) and Vulkan packed Q8_0 row64/SPIR-V 1.3. This is a whole-model choice, not per-tensor mixed execution or a new layout search. Cpu and VulkanRequired retain explicit selection. Missing or unsupported Vulkan still follows the initialization fallback contract.

Calibration first checks CPU/GPU logits for prefix `[1,9038]` using the established elementwise and RMS gates. It then measures synchronous incremental appends of those two tokens, with reset excluded, three warmups and five samples per backend. CPU measurements finish before GPU measurements start. Host times include CPU operators, submission, waits and vector transfers. GPU wins only when its median is strictly below 95% of the CPU median; ties and smaller improvements select CPU. Validation-enabled automatic calibration is rejected because it would contaminate performance selection; use Required for Vulkan validation.

The representative short prefix measures this model's decode-dominated path. It does not optimize batching, long-prefix prefill, multiple users or future kernels. The cache makes repeated selection reproducible; recalibration can vary with system load. Measurements are local observations, not a guarantee of performance on every workload.

## Cache

`StoriesPlannerOptions.CachePath` defaults to `reports/planner/stories15m.json`; `--plan-cache path` overrides it. The identity hashes CPU model/features, actual AVX2 availability, OS/runtime, GPU UUID, API/driver versions, device limits/memory flags and the contents of Core, Formats and Vulkan assemblies. Embedded shader bytes and layout manifests therefore participate in invalidation through the Vulkan assembly hash. The fixed profile, calibration sequence and policy are versioned by the code/identity. New hardware, driver, kernel or executable changes reject previous measurements.

Only matching identities with exactly five finite positive samples per backend are reused. Malformed JSON, missing samples, zero/negative/nonfinite timings and identity mismatches require fresh calibration. The selected backend is recalculated from samples; a serialized choice does not override the policy. Cache files are derived local performance state, not trusted configuration for arbitrary model execution. File-access failures propagate; they are not silently hidden by CPU fallback. An interrupted cache write producing invalid JSON is recalibrated on the next run.

T33 reads a matching cache after device identification and before constructing resident GPU weights. A cached CPU choice releases the context and opens the CPU model with zero candidate matrices/uploads; cached GPU choices still construct their matrices. Missing or invalid caches still calibrate. Device/context initialization remains necessary to validate identity. [Startup measurements and resource evidence](stories15m-planner-startup.md). Runtime execution failures retain the fail-closed behavior documented in [offload](stories15m-vulkan-offload.md).

## Verification

[Five process-level cases](benchmarks/planner-2026-10-12/verification.json) cover fresh calibration, identical cache reuse, changed identity, invalid sample values and malformed JSON. Every run emitted the expected greedy continuation and chose CPU on Intel Arc MTL. The final calibration medians were 9.144 ms CPU and 305.413 ms Vulkan for two appends. The cached run retained exactly the recorded measurements. These figures are separate from the longer [offload benchmark](stories15m-vulkan-offload.md#measurements).

Two critical policy tests cover median selection despite extreme outliers, ties, insufficient advantage, identity mismatch, missing samples, nonfinite/negative/zero timings and JSON round-trip. The existing 116 CPU tests pass; Release build has zero warnings/errors. Native GPU kernels and synchronization are unchanged; real hardware process calibration checks their integrated results, while the preceding offload hardware suite remains the native-path evidence.

```sh
dotnet run --project src/Satori.Console -c Release -- generate \
  .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12 --backend auto
```

Backend, measured selection and cache reuse are printed to stderr. CPU remains the default when no backend is supplied. To recalibrate, remove the selected derived cache file or choose another cache path. Task-owned verification caches were removed; the published report is retained.
