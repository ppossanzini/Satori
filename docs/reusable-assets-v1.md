# Reusable Assets and Capabilities

Status: DW01. What Satori already has that can be reused for the P01–P03 comparison and the selected
implementation path. Descriptor and codec capability of the candidate artifacts is owned by
[the capability matrix](capability-matrix-v1.md); this document owns Satori's own assets.

## Method

Each capability cites its module, its contract document and the test or report that verifies it. The state
was confirmed on the current tree by the Release build and the numerics suite, not by a historical checkbox.

## Code capabilities

| Capability | Module | Contract | Evidence |
|---|---|---|---|
| FP32 matrix-vector | `src/Satori.Core/FloatMatVec.cs` | TASKS T01 | build |
| Ternary2 codec, STQ2 container, mapped matrix, scalar/AVX2 MatVec | `src/Satori.Core/Ternary2*.cs`, `MappedTernary2Matrix.cs` | [ternary2-v1](ternary2-v1.md) | `Ternary2Tests`, `Ternary2Avx2Tests`, `MappedTernary2Tests`, `NumericalCorpusTests` |
| TernaryBitplane codec, STB1 container, scalar/AVX2 MatVec | `src/Satori.Core/TernaryBitplane*.cs` | [ternary-bitplane-v1](ternary-bitplane-v1.md) | `TernaryBitplaneTests`, `TernaryBitplaneAvx2Tests` |
| Memory-mapped read-only owner with protected lifetime | `src/Satori.Core/MappedWeights.cs` | [mapped-weights](mapped-weights.md) | `MappedWeightsTests` |
| GGUF reader: header, metadata, tensor directory, windows | `src/Satori.Formats.Gguf/GgufFile.cs`, `GgufCursor.cs`, `GgufMetadata.cs`, `GgufTensor.cs`, `GgufTypes.cs` | [gguf-subset-v1](gguf-subset-v1.md) | `GgufStructureTests`, `GgufTests`, `fixtures/gguf/v1` |
| GGUF descriptor inventory (tolerant scan) and `inventory` command | `src/Satori.Formats.Gguf/GgufInventory.cs`, `src/Satori.Console` | [gguf-inventory](gguf-inventory.md) | `GgufInventoryTests` |
| Q8_0/Q4_0 scalar and AVX2 MatVec | `QuantizedMatVec.cs`, `QuantizedAvx2Kernel.cs`, `QuantizedMatrix.cs`, `QuantizedRow.cs` | [gguf-subset-v1](gguf-subset-v1.md) | `QuantizedTests`, `Q4Tests`, `QuantizedAvx2Tests`, [gguf-2026-10-12](benchmarks/gguf-2026-10-12/README.md) |
| Quantized tensor adapter and weight store | `GgufQuantizedMatrix.cs`, `GgufWeightStore.cs` | [model-forward](model-forward.md) | `QwenForwardTests` |
| Model descriptor, shape, planner, capability report | `src/Satori.Core/Model/ModelDefinition.cs`, `ModelShape.cs`, `ModelConfig.cs`, `ModelPlan.cs`, `ModelTensor.cs`, `CapabilityGap.cs`, `ArchitectureAdapter.cs`, `DenseTransformerAdapter.cs`, `GgufModelLoader.cs` | [model-descriptor](model-descriptor.md) | `ModelDescriptorTests` |
| Byte-level BPE tokenizer and ChatML template | `src/Satori.Core/Tokenization/*`, `GgufBpeTokenizer.cs`, `GgufTokenization.cs` | [tokenizer](tokenizer.md) | `BpeTokenizerTests`, `fixtures/tokenizers/qwen2.5-coder-1.5b` |
| Transformer operators: RMSNorm, per-head RMSNorm, half-split RoPE, grouped attention, SwiGLU | `src/Satori.Core/TransformerOps.cs` | [qwen-operators](qwen-operators.md) | `QwenOperatorTests`, `TransformerOpsTests`, `fixtures/operators/transformer-v2.json` |
| Dense CPU forward | `Model/DenseForward.cs`, `DenseScratch.cs` | [model-forward](model-forward.md) | `QwenForwardTests`, `fixtures/qwen/*` |
| Incremental session: KV state, limits, lifetime | `Model/DenseSession.cs`, `IInferenceSession.cs` | [model-session](model-session.md) | `DenseSessionTests` |
| Sampling and greedy generation | `Model/TextGenerator.cs`, `TokenSampler.cs`, `GreedyTokenSelector.cs`, `SamplingOptions.cs` | [text-generation](text-generation.md) | `TextGeneratorTests`, `TokenSamplerTests`, `GreedyTokenSelectorTests` |
| Console commands `save`, `run`, `inventory`, `describe`, `text`, `generate` | `src/Satori.Console/Program.cs` | — | command help |
| Vulkan primitives: context, buffer, shader, Q8 MatVec, FP32 operators, attention session | `src/Satori.Backends.Vulkan/*`, `Shaders/` | [vulkan-backend-v1](vulkan-backend-v1.md), [vulkan-q8-matvec](vulkan-q8-matvec.md), [vulkan-transformer-operators](vulkan-transformer-operators.md) | `VulkanQ8Tests`, `VulkanTransformerTests`, `VulkanAttentionTests` |
| Stories15M laboratory path | `src/Satori.Formats.Gguf/Stories*.cs` | [benchmarks/stories15m](benchmarks/stories15m-2026-10-12/README.md) | `Stories*Tests`, `fixtures/models/stories15m-q8-v1` |

## Independent references

| Reference | Verifies |
|---|---|
| `tools/gguf_reference.py` | Independent GGUF parse and Q4_0/Q8_0 decode |
| `tools/transformer_reference.py`, `tools/transformer_reference_v2.py` | FP64 operator oracles |
| `tools/qwen_forward_reference.py` | Independent dense forward and greedy token oracle |
| `tools/qwen_bpe_tokenizer_fixture.py` | Official tokenizer fixture and chat render |
| `tools/gguf_header_inventory.py` | Candidate descriptor inventory |
| `tools/attention_reference.py`, `tools/stories15m_reference.py`, `tools/tokenizer_fixture.py` | Attention, Stories15M and tokenizer fixtures |
| `tools/download_gguf_corpus.py`, `tools/download_qwen_models.py`, `tools/compile_vulkan_shaders.py` | Corpus download and shader build |
| `tools/model-reference/` | Pinned external model sources |

## Fixtures

| Fixture | Content |
|---|---|
| `fixtures/gguf/v1` | Golden GGUF (F32, Q4_0, Q8_0, array metadata) with hashes |
| `fixtures/operators` | Operator oracles and attention boundaries |
| `fixtures/qwen` | Forward and greedy reference for Qwen2.5-Coder 1.5B and Qwen3 4B |
| `fixtures/tokenizers/qwen2.5-coder-1.5b` | Tokenizer-only GGUF and official ids |
| `fixtures/models/stories15m-q8-v1` | Laboratory regression model |

## Reports

| Report | Content |
|---|---|
| `docs/benchmarks/scalar-2026-10-12`, `avx2-2026-10-12`, `bitplane-2026-10-12` | Ternary2 and Bitplane kernel performance |
| `docs/benchmarks/gguf-2026-10-12`, `mapped-2026-10-12` | Q4_0/Q8_0 kernels and mapping |
| `docs/benchmarks/qwen-2026-10-13`, [qwen-baseline-v1](benchmarks/qwen-baseline-v1.md) | Resident dense CPU baseline on coding artifacts |
| `docs/benchmarks/vulkan-*`, `stories15m-2026-10-12` | Vulkan primitives and laboratory path |
| `docs/benchmarks/planner*-2026-10-12`, `resident-final-2026-10-12` | Removed model-specific Vulkan planner; historical only |
| `docs/benchmark-method.md` | Benchmark method |

## Declared unverified or unavailable

- Vulkan hardware verification is blocked in this environment by the missing Khronos validation layer.
  Primitive code exists; hardware-validated support is not claimed.
- The model-specific Vulkan resident and offload paths were removed. They are not reusable capabilities.
- No codec exists for Q4_K, Q6_K, Q3_K, IQ2_XXS, IQ2_S, IQ3_XXS, BF16, BitNet I2_S or the private Prism
  PTQ1_0/PQ2_0 types.
- Prefill and decode are single-threaded. No batched prefill, prompt/KV caching or token streaming.
- The BPE pre-tokenizer patterns cover the qwen2/qwen3 families only.
