# Scalar Baseline — October 12, 2026

Intel Core Ultra 7 155H, Arch Linux x64, .NET 10.0.12, SDK 10.0.112. Single-threaded scalar kernel; BenchmarkDotNet 0.14.0 in-process, three iterations of approximately 200 ms after warm-up.

| Matrix | Block Size | Mean ms | Standard Deviation ms | Bytes/Weight | Effective GB/s | Allocated Bytes/Op |
|---|---:|---:|---:|---:|---:|---:|
| 1024² | 64 | 7.555 | 0.159 | 0.312500 | 0.0434 | 0 |
| 1024² | 128 | 9.151 | 0.536 | 0.281250 | 0.0322 | 0 |
| 1024² | 256 | 11.063 | 1.118 | 0.265625 | 0.0252 | 0 |
| 2048² | 64 | 39.999 | 1.630 | 0.312500 | 0.0328 | 0 |
| 2048² | 128 | 31.004 | 1.107 | 0.281250 | 0.0380 | 0 |
| 2048² | 256 | 30.372 | 0.564 | 0.265625 | 0.0367 | 0 |
| 4096² | 64 | 162.336 | 9.078 | 0.312500 | 0.0323 | 0 |
| 4096² | 128 | 179.409 | 3.867 | 0.281250 | 0.0263 | 0 |
| 4096² | 256 | 161.493 | 7.048 | 0.265625 | 0.0276 | 0 |

36/36 numerical checks passed. Nine performance measurements completed. JSON reports retain errors, environment and full measurements; `benchmarkdotnet.md` reports confidence intervals.

This short run has dispersion and wide confidence intervals. The system did not grant the high priority requested by BenchmarkDotNet; the process ran at normal priority. Differences between block sizes do not demonstrate a repeatable advantage. Compare scalar and AVX2 under the same protocol, repeating measurements on stable hardware before selecting a kernel.

Measurements include decoding and exclude quantization and setup. Effective weight bandwidth is not measured RAM bandwidth. No conclusions about token/s or LLM quality. Method: [benchmark-method.md](../../benchmark-method.md).
