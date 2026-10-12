# Cached CPU planner startup

T33 validates the existing Auto cache immediately after Vulkan device identification, before constructing the GPU model. A valid CPU plan disposes the candidate context and opens the CPU model directly. Matching GPU plans still create resident matrices; missing or invalid plans still create the candidate and run the unchanged numerical gate and calibration. Required selection bypasses the cache.

The fingerprint, five-sample validation and >5% selection policy are unchanged. Device/context initialization and binary hashing remain necessary; this is not zero GPU initialization. CPU model validation still runs on a cache hit. Cache IO errors and unexpected initialization failures propagate. No fallback during inference is added.

`CandidateDevice` and `CandidateStatistics` expose the initialization snapshot, even when the returned model is CPU. Statistics describe Q8 matrices and their buffers/work, not every driver/context allocation. On every measured optimized cache hit, matrices, matrix allocation bytes, resident weight bytes, dispatches and vector transfers are all zero. The cold candidate has 43 matrices, 16,137,216 resident weight bytes and 16,658,176 total matrix allocation bytes.

## Measurements

Intel Arc MTL (8086:7d55), Mesa 26.2.4-arch1.1, .NET 10.0.12, Release, validation disabled. Same pinned Stories15M Q8_0 artifact and empty prompt, one greedy continuation (`Once`, token 9038). Baseline backend is commit `a704883`, with only the diagnostic startup runner added in a task-local copy. Separate caches accommodate the binary fingerprints; both select CPU.

Three independent groups each discard three fresh-process warmups per version, then measure seven fresh processes per version in alternating order. No tests run concurrently. No filesystem-cache flush is performed. Initialization scope is the runtime constructor, including selection, context disposal and CPU model construction; generator creation, token generation and process launch are excluded. Whole-process wall times are separately retained in the raw report.

| Group | Baseline median (min–max), ms | T33 median (min–max), ms | Reduction of medians |
|---|---:|---:|---:|
| 1 | 432.616 (400.709–508.154) | 214.766 (202.855–246.342) | 50.4% |
| 2 | 418.548 (399.537–496.277) | 217.648 (204.523–253.340) | 48.0% |
| 3 | 414.409 (397.873–475.746) | 234.402 (211.227–329.164) | 43.4% |

These are local startup results, not faster inference. [Raw samples](benchmarks/planner-startup-2026-10-12/samples.json) retain all observations, device limits, runtime and plan samples.

## Verification and reproduction

[Process cases](benchmarks/planner-startup-2026-10-12/verification.json) cover fresh calibration, exact CPU reuse, identity mismatch, invalid samples and malformed JSON. Exact reuse preserves the recorded plan and output. Invalid cases recalibrate with the resident candidate. A clearly synthetic matching GPU cache (CPU samples 100 ms, GPU samples 1 ms) exercises reuse of a GPU choice: 43 matrices, zero calibration dispatches at the constructor snapshot, expected generated output. Those synthetic numbers are branch fixtures, not measured performance.

[CLI cases](benchmarks/planner-startup-2026-10-12/cli-verification.json) verify Required Vulkan, unavailable-driver Auto fallback, unavailable-driver Required failure and invalid model rejection despite a valid CPU cache. Failures emit no generated text.

```sh
dotnet run --project validation/Satori.Vulkan.Validation -c Release -- \
  --planner-startup .temp/gguf-corpus/stories15M-q8_0.gguf .temp/t33/plan.json
```

First invocation calibrates; later invocations with the same binaries/hardware reuse the plan. Choose a task-owned cache path. Build succeeds without warnings; the two existing policy tests and 116 CPU tests pass. The first CPU suite run encountered the previously observed transient allocation assertion (7,752 bytes); its isolated repeat passes all 116 tests. No shader, native synchronization or numerical algorithm changed; prior T32 SyncVal evidence remains applicable, and real GPU calibration/generation verifies integration here. Task-local baseline, caches and scripts are removed after publishing evidence.
