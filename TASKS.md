# Satori Work Plan

Final goal: useful local coding assistance on Intel/AMD PCs with 16 GB RAM while the IDE, browser and compiler remain usable. Satori is an independent .NET runtime specialized in a small set of selected coding model families.

Binding product direction: [coding locale](docs/adr/current/product.md). Numerical and product validation: [validation protocol](docs/validation-protocol-v1.md). Test machines and roles are fixed in its hardware section; the Geekom A5 is the primary test host.

## Mandatory Product Requirements

| ID | Requirement | Acceptance Criterion |
|---|---|---|
| R-MM | Support the selected coding family and its declared compatible artifacts. Architecture breadth is not a release goal. | Artifact dimensions and state sizes come from validated metadata. Compatible sizes reuse the runtime. New families require explicit scope and validation. |
| R-CAP | Separate architecture, tokenizer, codec, transformations, backend and memory policy. | Validate every required capability before generation. Reject unsupported artifacts with explicit reasons. Implement only capabilities required by the selected path. |
| R-QUANT | Preserve compressed weights and select quantization by measured coding utility. | Compare a Q4 coding reference, APEX/MoE and Bonsai ternary in reference runtimes before selecting the main implementation path. Validate selected codecs independently. No permanent full-model FP32 expansion. Ternary product support depends on comparison results; BitNet is not a prerequisite. |
| R-PERF | Coding quality and PC responsiveness govern delivery. | Meet the validation protocol on representative C# tasks while IDE, browser and compiler run. Measure startup, prefill, TTFT, decode, total runtime memory and contention. No kernel-only result establishes product performance. |
| R-RES | Bound runtime memory, compute and I/O. | Account for compressed weights, state, scratch, staging and prefetch. Enforce context and cache limits. Provide compute limits and cancellation. Demonstrate compliance without sustained swap pressure. |

## Execution Rules

Current implementation includes CPU regression inference, mapped GGUF readers, quantized CPU kernels and reusable Vulkan primitives. Candidate models are not executable product support.

T01–T45 below are historical verification scope. They are not the next executable roadmap. Historical model-specific GPU paths do not establish current capabilities. Evidence remains in [TASK-LOG.md](TASK-LOG.md).

The next executable sequence is P01–P08. Execute one task at a time. Complete its verification before proceeding. Reference-runtime comparison precedes candidate-specific Satori implementation.

Tests cover critical format, numerical, memory/lifetime and inference behaviour. Documentation changes require consistency checks, not runtime tests.

## Product Execution Plan

| ID | Task | Dependencies | Verification / deliverable | Requirements |
|---|---|---|---|---|
| P01 | Fix the comparison configuration and workload | None | Pin exact Qwen2.5-Coder 7B Q4_K_M, Tiel-Coder APEX-I-NanoPlus and Bonsai-2 27B PTQ1_0 artifacts and compatible reference runtimes. Record licenses, hardware, thread limits, contexts, C# corpus and concurrent IDE/browser/build workload. Resolve open acceptance thresholds before P02. | R-QUANT, R-PERF, R-RES |
| P02 | Compare candidates in reference runtimes | P01 | Reproducible quality, startup, prefill, TTFT, decode, memory and responsiveness reports. Q4 and Bonsai use resident execution where feasible. APEX compares feasible resident and bounded expert-loading configurations. Report unavailable paths and budget failures explicitly. No Satori runtime is required. | R-QUANT, R-PERF, R-RES |
| P03 | Select one primary implementation path | P02 | Select one artifact/family and one external comparison baseline using measured utility. Record required capabilities, target budget and evidence. Reduce scope or stop if no candidate meets the product criteria. No automatic commitment to both APEX and Bonsai. | R-MM, R-CAP, R-QUANT, R-PERF |
| P04 | Lock the selected runtime and resource contracts | P03 | Align capability matrix and runtime contracts with the selected path. Specify state, tokenizer, codecs, cancellation, context, compute limits and memory enforcement. Streaming is required only if selected by evidence. | R-MM, R-CAP, R-RES |
| P05 | Implement the selected CPU inference path | P04 | Ordered subtasks derived from the selected gaps. Independent codec/operator references, tokenizer parity and end-to-end inference parity on the pinned artifact. Reuse existing kernels. | R-MM, R-CAP, R-QUANT |
| P06 | Enforce resource limits during real coding | P05 | Validate memory and context bounds, compute controls, cancellation and recovery. For a streamed path, validate bounded cache/prefetch, load failures, bytes/token and cold/warm behaviour. Run the concurrent workload. | R-RES, R-PERF |
| P07 | Optimize measured bottlenecks | P06 | Prioritize by end-to-end measurements. Add Vulkan only where it improves the selected workload within the shared memory budget. Compare on identical artifacts and settings. | R-PERF, R-RES |
| P08 | Validate the release scope | P07 | Repeat the fixed quality and concurrent-workload protocol. Publish supported artifacts, operating limits and reproducible results. Add a compatible size only if it serves an observed use case. | R-MM, R-CAP, R-QUANT, R-PERF, R-RES |

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

## Deferred Scope

- Additional architectures, BitNet and additional quantization recipes require a measured product need.
- APEX expert caching and AirLLM-style layer streaming are candidate memory policies. Neither is mandatory before P03.
- Streaming must report SSD traffic and contention; low VRAM alone does not establish low total memory.
- Metal, ROCm, vision, training, server and graphical interface are outside this delivery plan.
- Parameter count and maximum context are not release promises.
