# Memory-Mapped MatVec — October 12, 2026

Intel Core Ultra 7 155H, Arch Linux x64, .NET 10.0.12; one thread, AVX2, blocks of 64 weights. BenchmarkDotNet 0.14.0 in-process, three warm-ups and three measured iterations of approximately 200 ms.

| Matrix | Owned Array ms | Mapping ms | Opening + Validation ms, 3 Trials | First Computation After Validation ms, 3 Trials |
|---|---:|---:|---|---|
| 1024² | 0.228 | 0.234 | 4.509 / 1.507 / 1.478 | 0.768 / 0.228 / 0.215 |
| 2048² | 0.939 | 0.942 | 6.126 / 5.906 / 5.891 | 1.037 / 0.903 / 0.874 |
| 4096² | 3.760 | 3.751 | 25.744 / 25.954 / 25.988 | 3.837 / 3.731 / 3.785 |

Numerical identity confirmed in all nine opening trials and critical tests across all block sizes. Six benchmarks completed, zero managed bytes allocated per MatVec. Steady-state measurements are comparable: this short job does not demonstrate small advantages of mapping over arrays.

Files have just been written by setup and may already be in the page cache. Initial validation traverses the entire payload. The first computation after validation is not a cold SSD measurement; the first sample may include callback initialization or JIT compilation. Opening/validation includes its own allocations, excluded from the steady-state benchmark.

High priority not granted by the system: measurements at normal priority. Full report with dispersion in `benchmarkdotnet.md`; raw JSON measurements next to this file. Method: [benchmark-method.md](../../benchmark-method.md).
