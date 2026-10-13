# Satori Work Plan

Final goal: a multi-model inference engine for generic models within explicitly supported architectures and tensor formats, optimized for aggressively quantized models, including ternary models. The engine uses an independent .NET runtime, memory-mapped weights, a verified CPU backend and measured Vulkan acceleration. Initial target: Intel/AMD mini PCs with 16 GB of RAM. Metal and ROCm remain future extensions.

Support up to 27B is a goal conditional on model, quantization, context and available memory; it does not imply guaranteed speed. Experimental ternary support does not imply that a converted conventional model retains its original quality.

## Mandatory Product Requirements

| ID | Requirement | Acceptance Criterion |
|---|---|---|
| R-MM | The engine MUST support multiple models. A regression fixture must not define the engine architecture. | Run at least two distinct model artifacts with different topology dimensions through the same runtime and session contracts. Validate each against independent references. Adding a model within supported capabilities must not require model-named runtime interfaces or hard-coded execution dimensions. |
| R-CAP | Model architecture, tensor quantization and execution backend MUST be separate capabilities. Generic support is bounded by an explicit capability matrix. | Read and validate model metadata and tensor shapes before execution. Derive workspace and KV sizes from the validated model and context. Reject unsupported architecture, tokenizer, operator or format with an explicit reason before generation. |
| R-QUANT | The engine MUST optimize aggressively quantized inference, including ternary weights, without restricting inference to ternary or one model family. | Validate end-to-end inference for a supported ternary model and a supported non-ternary model. Keep compressed weights without permanent full-model FP32 expansion. Verify each format against independent numerical references; assess conversion quality separately from kernel correctness. |
| R-PERF | Speed and bounded memory on small machines MUST govern the roadmap. A small-fixture result alone cannot establish product performance. | Declare model, quantization, context, hardware, RAM budget and performance targets before optimization. Measure startup, prefill, TTFT, decode and peak memory on representative workloads, including a coding model. Report target attainment and CPU/GPU comparisons without extrapolating from the small fixture. |

## Execution Rules

Current scope: Model-specific Vulkan offload, resident graphs, planner, diagnostic runners and their tests have been removed. CPU reference inference and reusable numerical/Vulkan primitives remain. The completion evidence below records historical verification, not availability of the removed paths.

Status: **historical tasks T01–T45 verified in their original scope**. Reusable mapped GGUF readers, quantized scalar/AVX2 kernels and Vulkan primitives remain. Earlier model-specific inference, offload and planner milestones do not establish multi-model product support. The removed optimization paths are not current capabilities. Verification history is retained in [TASK-LOG.md](TASK-LOG.md); the mandatory requirements and replanning gate below govern future work.

Updated verification policy: no tests for bootstrap, configuration or simple steps; use builds and demos. Add tests only for critical invariants of format, numerical correctness, memory/lifetime, SIMD, GPU and inference.

Execute one task at a time in the listed order. Each task depends on the previous one, except T01. Mark completed only after verification. Decision tasks produce a specification before dependent implementation. No task includes functionality from later phases. Completed profile-specific tasks establish only their verified scope; they do not satisfy R-MM, R-CAP, R-QUANT or R-PERF by themselves.

## A — CPU Numerical Laboratory

Context: .NET 10, C# 14, x64; xUnit for correctness, BenchmarkDotNet for performance. No dependency on other inference runtimes. Vulkan, tokenizer and generation are excluded from this phase.

| ID | Task | Expected Result | Verification | Specific Exclusions |
|---|---|---|---|---|
| T01 | Create a minimal executable solution | Satori library, console with FP32 matrix-vector product on known data, test project, pinned SDK | Release build, tests and console return the expected result | Quantization |
| T02 | Specify the Ternary2 v1 format | Byte-level specification: weight order, codes 00/01/10, rejection of 11, FP32 scales, blocks, padding, dimensions, overflow and nonfinite values; deterministic quantization rule | Hand-computable binary examples cover full, partial and zero blocks | SIMD |
| T03 | Implement the Ternary2 codec | Packing, unpacking and quantization conforming to T02 | Tests for all codes, scales, padding, invalid inputs and determinism | MatVec |
| T04 | Define the CPU matrix contract | Ternary2 matrix with immutable shape and block size, private owned payload, import through validated copy; MatVec through CPU spans | Build and end-to-end console use; payload validation covered by critical codec tests | Generic descriptors and GPU abstraction deferred until other formats are introduced |
| T05 | Implement scalar Ternary2 MatVec | Direct computation on compressed weights with FP32 accumulation, without full expansion | Comparison with FP64 reference on the same reconstructed weights; documented tolerances | AVX2 |
| T06 | Create the reproducible numerical corpus | Matrices with fixed seed, zeros, alternating signs, allowed extreme scales and irregular dimensions | Tests distinguish kernel error from FP32→ternary quantization error | Linguistic quality |
| T07 | Add the scalar benchmark | Per-size report: latency, bytes/weight including scales and padding, effective bandwidth, allocations, environment | Release run on fixed corpus; no managed allocations on the hot path after warm-up | Arbitrary speed targets |
| T08 | Implement AVX2 MatVec | Kernel with partial tails, CPU capability detection and scalar fallback | Same corpus as T06; AVX2 and fallback execution; performance comparison with T07 | AVX-512 |
| T09 | Implement the memory-mapped owner | Read-only mapping with protected lifetime during execution, large-file access and range checks | Tests on temporary files, offsets, disposal, concurrent acquisitions and result identity | OS-specific prefetch |
| T10 | Connect MatVec to mapped weights | Console loads a Ternary2 matrix from file and executes scalar or AVX2 | Results identical to in-memory version; benchmark separates opening/validation, first computation after validation and warm steady state | Full GGUF; actual cold storage not measured |

Milestone A: an FP32 matrix is quantized, saved, mapped and multiplied with verified results and measured performance.

## B — Real Formats

Context: laboratory A is working. Importing does not mean automatically converting a GGUF format to ternary. The Transformer runtime remains excluded.

| ID | Task | Expected Result | Verification | Specific Exclusions |
|---|---|---|---|---|
| T11 | Specify the supported GGUF subset | Pinned versions, metadata types, alignments, tensor dimensions, parser limits and Q4_0/Q8_0 formats | Valid fixtures and rejection cases documented against the format specification | Universal GGUF support |
| T12 | Implement the GGUF reader | Metadata and tensor index reads on mapping, with dimension and offset validation | Tests on known fixtures, truncated files, invalid metadata and out-of-file ranges | Inference |
| T13 | Implement Q8_0 MatVec | Per-block decoding and scalar kernel on imported tensors | Comparison with independent reference values and real corpus | Q4_0 |
| T14 | Implement Q4_0 MatVec | Scalar kernel conforming to the GGUF layout | Independent comparison and benchmark on the same input as Q8_0 | Q4_K and Q2 |
| T15 | Optimize Q8_0/Q4_0 kernels for AVX2 | Specialized implementations registered by format | Reference tests and repeatable benchmarks; keep kernels without a demonstrated advantage experimental | New formats |
| T16 | Specify TernaryBitplane v1 | Defined layout, scales, padding and lossless conversion from Ternary2 | Binary examples and logical-weight identity | Base3 |
| T17 | Implement TernaryBitplane | Codec and scalar MatVec conforming to T16 | Comparison with the same Ternary2 weights and complete decoding benchmark | LUT |
| T18 | Optimize TernaryBitplane for AVX2 | SIMD kernel on compact layout | Numerical corpus and fair comparison with Ternary2 at identical scales | Activation quantization |

Milestone B: benchmarks on real GGUF Q4_0/Q8_0 weights and a controlled comparison of ternary layouts.

## C — First Working Model

Context: select one small model and one architecture for the first complete validation pipeline, not as a product restriction. No graphical interface; a console is sufficient.

| ID | Task | Expected Result | Verification | Specific Exclusions |
|---|---|---|---|---|
| T19 | Pin the first model profile | Documented model/version/license, tokenizer, topology, required formats, chat template and RAM budget; token/logit fixtures with provenance | All required operators listed; missing formats become mandatory tasks before T20 | Other model families |
| T20 | Implement profile operators | CPU operators required by T19, individually verified | Comparisons with independent fixtures for normalization, RoPE, attention and the activations actually required | Unneeded operators |
| T21 | Implement the profile tokenizer | Encoding, decoding, special tokens and model template | Identity with official sequences on multilingual corpus and edge cases | Universal tokenizer |
| T22 | Implement a single-token forward pass | Logits obtained by traversing all layers from GGUF weights | Comparison of intermediate tensors and logits with T19 fixtures | Autoregressive generation |
| T23 | Implement the KV cache | Incremental state with explicit context limits and memory budget | Incremental logits compared with full recomputation; reset and full context verified | SSD persistence |
| T24 | Implement greedy generation | Prompt→token→text console with EOS and token limit | Deterministic sequence compared with reference; readable errors for unsupported models | Random sampling |
| T25 | Add configurable sampling | Temperature, top-k/top-p and seed with validation rules | Distribution, reproducibility and edge-case tests | Server API |
| T26 | Publish the first model benchmark | TTFT, prefill, decode, memory and quality against a reference with the same model/configuration | Reproducible report with hardware, runtime, context and quantization; no estimated measurement presented as real | Generalization to 27B |

Milestone C: Satori generates correct text from a supported small model on CPU.

## D — Vulkan

Context: verified CPU as reference; a compatible GPU is required to complete hardware validation.

| ID | Task | Expected Result | Verification | Specific Exclusions |
|---|---|---|---|---|
| T27 | Pin the Vulkan backend contract | Documented bindings, SPIR-V compilation, buffer ownership, submission, synchronization and minimum capabilities | Specification covers initialization failure, disposal and CPU fallback | Metal/ROCm |
| T28 | Implement the minimal Vulkan path | Upload, simple kernel dispatch and readback with explicit lifetimes | Correct known result on GPU; validation layer without errors | Quantized kernels |
| T29 | Implement Vulkan quantized MatVec | First format selected from those used by the T19 model, with weights resident in buffers | CPU/GPU corpus within tolerance; kernel times separated from end-to-end latency | All formats simultaneously |
| T30 | Integrate model offload | GPU execution of compatible operations or complete layers; CPU fallback | Logits match reference; report of transfers, memory and total latency | Splitting a MatVec across CPU/GPU |
| T31 | Implement the measurement-based planner | Kernel/layout/backend selection based on capabilities and measurements; cache invalidatable by hardware, driver and kernel version | Reproducible choices, invalid cache rejected, comparison with CPU-only | Promise of GPU advantage on every hardware configuration |

Milestone D: usable Vulkan acceleration with advantages and limits documented on tested hardware.

## D2 — Vulkan Optimization

**Model-specific optimization roadmap removed. T46–T50 are cancelled in their previous formulation.** Raw measurement files remain historical evidence. Reusable primitives remain subject to the multi-model requirements; no model-specific optimization sequence is carried forward.

### Replanning Gate

Next work item: define the supported capability matrix and representative validation models. This is a planning step, not authorization to implement another model-specific runtime. No model has been selected as the product architecture.

Before resuming implementation:

1. Define the architecture, tokenizer, operator and tensor-format capability matrix required by R-CAP and R-QUANT.
2. Specify model-independent runtime/session contracts and metadata-driven memory planning required by R-MM. Keep existing small-model artifacts as regression fixtures.
3. Select representative validation artifacts and coding workloads; fix numerical, quality, memory and performance criteria required by R-PERF.
4. Replace the pending roadmap with ordered tasks, dependencies and acceptance criteria traced to R-MM, R-CAP, R-QUANT and R-PERF. Reuse existing kernels and backend infrastructure where compatible.

Runtime contracts must satisfy the mandatory product requirements before dependent tasks become executable. Requirements stay in this plan; implementation details stay in the relevant technical documentation. No ADRs or equivalent decision-record systems are created.

## E — Extensions After the Baseline

These candidates are not an executable sequence. Multi-model support and ternary inference are mandatory baseline requirements, not optional extensions. The replanning gate determines which candidates are required for the baseline and which remain future work.

1. Add Q2/Q4_K formats required by selected models.
2. Evaluate TernaryBase3/LUT on identical matrices and scales.
3. Validate a natively ternary model before extending the runtime.
4. Evaluate post-training ternary conversions with independent quality metrics.
5. Add 7–9B profiles and later 20–27B within measured budgets.
6. Evaluate AVX-512 where available.
7. Specify a Metal backend for Apple Silicon, including the ARM64 CPU path.
8. Specify a ROCm backend on actually supported AMD GPUs.

Metal and ROCm are notes for future work, not dependencies for completing the first version.
