# Measured Stories15M execution planner

Auto compares three whole-model paths for the pinned profile: CPU automatic Q8_0 (AVX2 when available), preceding Vulkan packed Row64 MatVec offload, and resident Vulkan FP32 graph with Cooperative64 internal matrices and Row64 output. Explicit Cpu/Required selections bypass calibration; Required accepts `--vulkan-path matvec|resident`. CPU remains the default command-line backend.

Each GPU candidate first passes CPU logits at prefix [1,9038], using elementwise 5e-4+5e-5*abs(reference) and RMS <=1e-4. Calibration measures three warmups and five synchronous two-token samples per candidate, sequentially CPU/MatVec/Resident. Reset and session construction are excluded. Timestamps and validation are disabled. The fastest GPU path wins only when its median is strictly below 95% of the CPU median. CPU wins ties/insufficient advantage; ties between GPU paths retain MatVec. This declared short decode workload governs selection; longer-context comparisons are diagnostic, not a hidden weighting rule.

## Cache and failure

The derived cache requires matching CPU/features/actual AVX2, OS/runtime, GPU UUID/API/driver/capabilities and Core/Formats/backend content hashes. The protocol identifier explicitly versions the three-path two-token workload, graph FP32 precision, reduction policy and arena layout; shader/operator resources also participate through the backend hash. All three candidates require exactly five finite positive samples. Old two-path caches, malformed JSON, missing/invalid samples and identity changes recalibrate. The selected path is derived from samples, not accepted from serialized output fields. Cache IO failures propagate.

A cached CPU choice initializes the context for identification, but creates neither legacy nor resident weight matrices. A cached GPU choice creates only its selected path and skips calibration. CandidateStatistics and CandidateResidentStatistics retain cold snapshots; active Statistics/ResidentStatistics expose the selected runtime. Missing/unsupported Vulkan and device-memory initialization failure retain CPU fallback. Numerical failures, unexpected exceptions and execution failures propagate. No inference replay/fallback is added.

## Evidence

[Three-path process reports](benchmarks/planner-resident-2026-10-12) cover fresh calibration, exact CPU reuse, wrong identity, missing/invalid resident samples, malformed JSON and synthetic cached choices for both GPU paths. Synthetic timings are branch fixtures, not performance measurements. Every success emits Once/token 9038. Cached CPU has zero candidate matrices; cached GPU constructs only the selected weights with no calibration dispatches.

Three independent Release processes compare short and long contexts on all three paths, timestamps/validation disabled. Per context, one excluded prefill precedes three warmup appends and seven sequential measured appends. Measured positions are explicitly recorded: 4–10 and 67–73. These decode observations do not represent repeated prefill or TTFT measurements. [Raw samples and summary](benchmarks/planner-resident-2026-10-12/summary.json).

Policy tests cover median/outliers, >5% margin, ties, all three candidates, invalid/missing/nonfinite samples, identity and JSON roundtrip. [Resident correctness/lifetime evidence](stories15m-vulkan-resident.md). [Cached CPU startup reduction](stories15m-planner-startup.md) remains the preceding before/after evidence; T44 does not repeat that baseline experiment.

```sh
dotnet run --project src/Satori.Console -c Release -- generate \
  .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12 --backend auto
```

Backend/path, measured selection and cache reuse appear on stderr. `--plan-cache path` overrides reports/planner/stories15m.json. Remove that derived cache to recalibrate. Auto owns path selection; `--vulkan-path` is for explicit Vulkan execution.
