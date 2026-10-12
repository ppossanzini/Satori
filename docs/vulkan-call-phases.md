# Vulkan call phase diagnostics

The synchronous Q8_0 path now supports optional host phase accounting through `VulkanQ8Options.MeasureHostPhases`. `MeasureKernelTime` independently controls GPU timestamp queries. `StoriesBackendOptions.MatVec` passes these explicit options to resident model matrices; custom measurement options require an explicit backend rather than Auto, keeping diagnostic timing out of planner-cache selection. Defaults preserve the existing production path.

## Scope

Each successful matrix execution accumulates disjoint host intervals:

| Field | Included host work |
|---|---|
| UploadMs | Staging map, CPU copy, optional flush, unmap |
| RecordingMs | Command-buffer reset/begin, copies, barriers, bindings, dispatch, optional timestamp commands |
| SubmissionPreparationMs | End command buffer, validation check, fence reset, submit preparation |
| QueueSubmitMs | Queue submission call and success bookkeeping |
| FenceWaitMs | Blocking fence wait and success bookkeeping |
| PostWaitValidationMs | Validation check after completion |
| QueryMs | Optional timestamp-result retrieval; disabled mode still includes the trivial helper/timer overhead |
| ReadbackMs | Staging map, optional invalidate, validation check, CPU copy, unmap |
| OtherMs | Gaps between instrumentation boundaries inside the measured matrix host interval |

Upload and readback group mapping, CPU copying and cache maintenance; these substeps are not individually attributed. The measured matrix interval excludes lock acquisition and input validation, as before. Session latency includes those costs, CPU operators and diagnostic accumulation. Cumulative statistics count only successfully completed MatVec executions, excluding cold weight uploads, teardown waits and failed submissions. One successful MatVec currently implies one submission and fence wait. PhaseSamples distinguishes instrumented calls from default calls.

GPU timestamp windows occur within submitted work and overlap the host fence-wait interval. Do not add kernel time to host phases, or label fence-wait minus kernel time as transfer time: scheduling, other device commands and host wake-up may contribute. Individual native operations are not profiled within upload/readback, and no GPU transfer-engine timeline is claimed. Diagnostic instrumentation itself has cost; compare modes with the same host instrumentation enabled.

## Evidence

[Raw six-process reports and summary](benchmarks/vulkan-phases-2026-10-12/summary.json). Three fresh processes per timestamp mode, run sequentially in alternating order; each process performs three warmups and seven measured runs. Each run uses the existing three-token prefill `[1,9038,2501]` plus eight teacher-forced appends. Validation layers are disabled for measurements. CPU and GPU run sequentially; no tests overlap measurements. CPU/GPU numerical gates pass for eleven checked prefixes in every process.

Ranges below are the minimum/maximum of the three process medians, not confidence intervals. Decode uses 56 measured appends per process. Phase values use each run's accumulated MatVec interval divided by eleven, including prefill positions; they are not decode-only phase times.

| Observation | Timestamps on | Timestamps off |
|---|---:|---:|
| CPU decode median | 3.723–3.975 ms | 3.720–5.646 ms |
| GPU decode median | 156.099–158.112 ms | 149.783–157.967 ms |
| GPU complete-run time / 11 | 164.859–171.690 ms | 154.927–161.923 ms |
| Upload host phase / append | 0.362–0.383 ms | 0.400–0.409 ms |
| Recording / append | 1.451–1.558 ms | 1.388–1.432 ms |
| Queue submission / append | 5.096–5.370 ms | 5.127–5.289 ms |
| Fence wait / append | 155.449–161.141 ms | 145.432–152.094 ms |
| Readback host phase / append | 2.154–2.264 ms | 2.204–2.339 ms |
| GPU timestamp windows / append | 35.320–35.899 ms | unavailable |

Component medians are not additive. For each raw measured run, the sum of phase deltas was checked against its accumulated matrix host duration with a 1e-6 ms tolerance. Each measured run had exactly 473 completed submissions, 473 fence waits, 473 instrumented calls and 2,865,280 transferred vector bytes: 43 dispatches and 260,480 bytes per append. Warm measured inference allocated zero managed bytes in all runs; diagnostic snapshots are taken outside the allocation/timing intervals.

Observed conclusion: host upload/readback and recording are small compared with fence wait, and disabling timestamps does not remove the CPU/GPU gap. This does not isolate which device commands or scheduling effects dominate the wait. The previous baseline is reproduced within observed variability; no optimization or speedup is claimed. Mapping reuse alone cannot address most of the measured latency. Cooperative kernels and fewer submissions remain the planned work.

The existing thirteen GPU/policy tests pass with Khronos SyncVal after the shared submission-path change. The existing CPU suite passes 116 tests. No new tests mirror timer implementation; real workload checks validate phase accounting, counters, numerical results and allocations. Release build has zero warnings/errors. Hardware device loss and noncoherent staging are still unexercised limitations inherited from the baseline.

## Reproduction

Build Release, then run each mode in three separate processes, sequentially:

```sh
dotnet validation/Satori.Vulkan.Validation/bin/Release/net10.0/Satori.Vulkan.Validation.dll \
  --offload-profile .temp/gguf-corpus/stories15M-q8_0.gguf on

dotnet validation/Satori.Vulkan.Validation/bin/Release/net10.0/Satori.Vulkan.Validation.dll \
  --offload-profile .temp/gguf-corpus/stories15M-q8_0.gguf off
```

The runner writes `reports/vulkan/offload-profile-timestamps-on.json` or `offload-profile-timestamps-off.json`; copy each report before the next run of that mode. Full raw samples include startup and CPU/GPU comparison, while the summary derives per-process medians. Startup observations include model/backend initialization inside a fresh process, without forcing cold operating-system file caches.

For regression hardware tests, restage the pinned Khronos layer from the [minimal-path recipe](vulkan-minimal-path.md#reproduction) under a task-owned temporary directory. The local `.temp/t32` layer package and diagnostic scripts were removed after publication; no system installation changed.
