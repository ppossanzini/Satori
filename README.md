# Satori

Independent C#/.NET inference runtime for useful local coding on Intel/AMD PCs with 16 GB RAM while the IDE, browser and compiler remain usable. Support is specialized in selected model families and explicitly supported formats.

[Product requirements and execution plan](TASKS.md#product-execution-plan) start with a reference-runtime comparison of a Q4 coding model, APEX/MoE and Bonsai ternary. One primary implementation path is selected from measured quality, latency, memory and PC responsiveness. Candidate support is not implemented; complete inference has been checked on a small regression fixture only.

## Targets and Development Order

- Intel/AMD x64 CPU: scalar and SIMD kernels, the first target.
- Vulkan Compute: reusable Q8_0 MatVec and FP32 operator primitives; no model-specific offload or planner.
- Metal: a note for a future macOS and Apple Silicon backend.
- ROCm: a note for a future AMD GPU backend.

Metal and ROCm are future targets to evaluate; they are outside the first milestone.

## Getting Started

Requires .NET SDK 10.0.112 (or a compatible patch), C# 14. The initial baseline was verified on Linux x64.

```sh
dotnet restore Satori.sln
dotnet build Satori.sln -c Release --no-restore -m:1
dotnet test Satori.sln -c Release --no-build --no-restore -m:1
dotnet run --project src/Satori.Console -c Release --no-build
```

The console quantizes a 2×3 FP32 matrix to Ternary2 and runs MatVec directly on the compressed blocks. It prints `[4, -2.6666667]` and 40 bytes of storage, including scales and padding. For matrices this small, padding outweighs compression.

Ternary2 supports blocks of 64, 128 or 256 weights, FP32 scales, validated payload import and explicit weight reconstruction for numerical analysis. The binary specification is in [docs/ternary2-v1.md](docs/ternary2-v1.md). The kernel does not expand the matrix or allocate managed memory during computation.

`Ternary2MatVec.Execute` selects AVX2 when available and scalar otherwise. `ExecuteScalar` explicitly uses the reference implementation; `ExecuteAvx2` includes the same fallback for unsupported CPUs. Partial tails are processed by the scalar kernel without reading beyond the input.

Tests are reserved for critical behavior: binary format, data corruption, numerical extremes, kernel correctness and memory. Bootstrap and simple steps are verified through builds and demos.

## Memory-Mapped Weights

`MappedWeights` opens weights read-only and provides zero-copy access through callbacks, with 64-bit offsets. Closing prevents new accesses and keeps existing ones valid until their callbacks finish. `MappedTernary2Matrix` loads a validated STQ2 file and runs scalar or AVX2 MatVec directly on mapped bytes. Contract and limits: [docs/mapped-weights.md](docs/mapped-weights.md).

Save and load demo, using a new path:

```sh
dotnet run --project src/Satori.Console -c Release -- save demo.stq2
dotnet run --project src/Satori.Console -c Release -- run demo.stq2
```

The demo prints `[4, -2.6666667]`. The `run` command uses `[2, -1, 3]` for matrices with three columns and a vector of ones otherwise, displaying at most eight results. It is a laboratory console, not a text generator. `Save` does not overwrite existing files. The STQ2 v1 container is described in the [Ternary2 specification](docs/ternary2-v1.md); each matrix payload is limited to `int.MaxValue` bytes.

The development plan is in [TASKS.md](TASKS.md).

## Numerical Corpus and Benchmarks

```sh
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --validate
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --filter '*'
```

The first command validates the corpus with both kernels and saves numerical errors; the second compares scalar and AVX2 with BenchmarkDotNet for 1024², 2048² and 4096² matrices, with blocks of 64/128/256 weights. Regenerable reports are in `reports/cpu/`: environment, numerical errors, performance JSON and BenchmarkDotNet reports with statistical dispersion. On hardware without AVX2, the second method uses the scalar fallback; the environment report records actual capabilities.

Measurements include weight decoding but exclude quantization and setup. They run warm, on one thread and in the same process; effective bandwidth counts only the quantized payload, including scales and padding. It does not represent physical RAM bandwidth. The short job provides a development baseline; selecting optimizations requires repeated comparisons on the same hardware.

First saved run: [scalar baseline from October 12, 2026](docs/benchmarks/scalar-2026-10-12/README.md). Protocol and errors: [laboratory method](docs/benchmark-method.md).

Repeated comparison: [scalar / AVX2, two runs](docs/benchmarks/avx2-2026-10-12/README.md). On Intel Core Ultra 7 155H, AVX2 showed an approximately 27–32× advantage over our scalar kernel in the measured synthetic configurations. This is not an LLM inference measurement or a comparison with other runtimes.

To compare owned memory and mapping:

```sh
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --mapped --filter '*'
```

Reports in `reports/mapped/`: opening/validation and first computation in `startup.json`, steady state in `performance.json`. The first computation occurs after page validation and does not represent a cold SSD read. [First mapped run](docs/benchmarks/mapped-2026-10-12/README.md).

## GGUF

The [GGUF v1 profile](docs/gguf-subset-v1.md) is implemented: little-endian v3 containers, an F32/F16/Q4_0/Q8_0 catalog, scalar/array metadata and tensor access through mapped windows. `GgufQuantizedMatrix` validates scales in the selected Q8_0/Q4_0 tensor and provides scalar and AVX2 MatVec with fallback. F32/F16 are cataloged; tokenizer and inference are not yet implemented.

The [critical fixtures](docs/gguf-fixtures-v1.md) verify layout and rejection of corrupted files. The [public corpus and reports](docs/benchmarks/gguf-2026-10-12/README.md) compare catalogs with the official reader and computations with independent FP64 references. Reproduction:

```sh
python3 tools/download_gguf_corpus.py
dotnet run --project validation/Satori.Gguf.Validation -c Release -- <model.gguf> <reference.json>
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --gguf .temp/gguf-corpus --filter '*'
```

## Ternary Bitplane Layout

`TernaryBitplaneMatrix.FromTernary2` converts losslessly, preserving FP32 scales and separating positive and negative bits. `ToTernary2` reproduces the original payload byte for byte. The new [STB1 format](docs/ternary-bitplane-v1.md) supports validated import and saving/loading; it uses the same number of bytes as Ternary2.

`TernaryBitplaneMatVec.Execute` uses AVX2 where available and scalar otherwise. The comparison with Ternary2 uses identical weights, scales and inputs; it includes all decoding:

```sh
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --layouts --validate
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --layouts --filter '*'
```

Reports and comparison limits: [layout benchmarks](docs/benchmarks/bitplane-2026-10-12/README.md). The critical suite can also run with `DOTNET_EnableAVX2=0` to test the actual fallback. For Q8_0/Q4_0 and bitplane, explicit `ExecuteAvx2` requires AVX2; use `Execute` for portable dispatch.

## Reference Validation

Small-model inference code and independent numerical fixtures are laboratory regression material. They do not define the supported product models, execution dimensions or tokenizer contracts. Selected-family inference, representative coding workloads and candidate selection remain governed by [the product requirements](TASKS.md#mandatory-product-requirements).

Tests with `Category=ModelIntegration` require the real corpus: `python tools/download_gguf_corpus.py`. For the offline suite without weights, use `--filter 'Category!=ModelIntegration'`.

## Vulkan Backend

The [Vulkan contract](docs/vulkan-backend-v1.md) fixes Silk.NET.Vulkan 2.23.0, Vulkan 1.1 / SPIR-V 1.3, buffer ownership and a synchronous public API. Optional device features are capability-gated with a verified fallback. The [minimal compute path](docs/vulkan-minimal-path.md) has been verified on the host Intel Arc GPU with Khronos synchronization validation. Hardware tests are separate from CPU tests and require GPU/layer access. Reusable Q8_0 MatVec and FP32 operator primitives remain available; Model-specific offload, resident graphs and planner have been removed.

Resident Vulkan Q8_0 MatVec and diagnostic timings: [implementation and evidence](docs/vulkan-q8-matvec.md).

Console generation uses the CPU reference fixture. Backend selection, device selection, plan-cache and resident-path generation options are no longer supported. Published model-level GPU measurements are historical data, not current runtime capabilities.
