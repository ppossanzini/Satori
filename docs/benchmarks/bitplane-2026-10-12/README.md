# Ternary2 and TernaryBitplane

CPU comparison on identical ternary codes, FP32 scales and activations. Bitplane comes from lossless conversion of the Ternary2 payload; no new quantization. Square shapes, 1024/2048/4096; block sizes 64/128/256. Seed 1729, corpus v1.

Each block occupies `4 + blockSize/4` bytes in both layouts: 0.3125 / 0.28125 / 0.265625 bytes per weight in complete shapes. Bitplane provides no RAM saving over Ternary2.

`scalar/` retains the pre-optimization measurement. `run1/` and `run2/` compare all four paths in the same job: Ternary2 scalar/AVX2 and bitplane scalar/AVX2. `numerical.json` checks 27 critical combinations and nine large matrices: scalar identity, SIMD output identity across both layouts and the original kernel's FP64 bound on the same weights. Tests add partial-tail dimensions and allocation checks.

BenchmarkDotNet ShortRun, Release, in-process, single thread, three warm-ups and three iterations of approximately 200 ms. Setup, conversion and validation excluded. Decoding included in MatVec. Weights already in memory; no cold SSD or inference measurement. Environment, dispersion and confidence intervals in full reports. Normal priority.

## Reproduction

From the project root:

```sh
dotnet build Satori.sln -c Release -m:1
dotnet run --project benchmarks/Satori.Benchmarks -c Release --no-build -- --layouts --validate
dotnet run --project benchmarks/Satori.Benchmarks -c Release --no-build -- --layouts --filter '*'
```

Regenerable reports are written to `reports/layouts/`. The explicit AVX2 bitplane kernel requires that capability; `Execute` provides fallback. The layout does not automatically replace Ternary2 elsewhere in the runtime.

## Outcome

72 total BDN measurements across the two final runs, all with zero bytes allocated per operation. Bitplane AVX2 is 20.2–39.2× faster than its own scalar kernel in the measured configurations: automatic bitplane dispatch therefore uses AVX2 when available.

Against Ternary2 AVX2, the bitplane/Ternary2 latency ratio ranges from 0.64 to 1.54; bitplane is faster in 7 of 18 pairs. Variability and wide intervals in the short job do not demonstrate a general layout advantage. Ternary2 remains the adopted path; bitplane remains an experimental alternative to measure in its usage context. No cross-layout planner is introduced. Individual values in `comparison.json`.
