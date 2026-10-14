# Numerical Laboratory Method

Corpus v1: xorshift32 with seed 1729, with 24-bit conversion defined in the generator. Bytes do not depend on the `System.Random` implementation. Nine critical families for each of the three block sizes; nine additional combinations for benchmark matrices 1024², 2048², 4096². The CPU comparison checks both implementations: 72 numerical results per run. SIMD tests add dimensions covering all tails from one to seven elements.

## Separate Errors

- `WeightQuantizationRms`: difference between original FP32 weights and reconstructed weights.
- `OutputQuantizationRms`: difference between the two FP64 MatVec results, on original and reconstructed weights respectively.
- `KernelRms` and `KernelMaxAbsoluteError`: difference between the FP32 kernel and FP64 reference on the same reconstructed weights.

For each row, the kernel bound is `gamma(n) × sum(abs(wq × x)) + (columns + blocks) × float.Epsilon`, where `n = blockSize + blocks + 2` and `gamma(n) = n × 2^-24 / (1 - n × 2^-24)`. This is a conservative bound for this accumulation path, without intermediate overflow. Maximum-scale cases use small activations to keep results finite. It is not a guarantee for arbitrary inputs causing FP32 overflow. The error/bound ratio must be ≤ 1 for every row, with a finite result.

## Performance

BenchmarkDotNet 0.14.0, in-process toolchain, short job with three warm-ups and three measured iterations, target time 200 ms per iteration. The toolchain avoids creating temporary executables. Setup generates and quantizes a matrix per case outside measured time; the benchmark does not retain original arrays. The kernel processes the compact payload with preallocated input and output.

JSON reports mean latency, standard deviation, operations/s, bytes/weight, effective weight bandwidth and managed allocations per operation. Bandwidth is `packedBytes / elapsedSeconds / 10^9`; it excludes repeated activation reads, cache traffic and output writes. The BenchmarkDotNet report preserves hardware and confidence intervals.

Measurements are synthetic and warm. Cache, CPU frequency, scheduling and temperature affect results. They do not measure cold start, peak process memory, energy or token/s; they do not demonstrate model quality. The short job suits an initial baseline, not claims of small advantages between kernels. Allocation checks using the thread counter in critical tests are separate from BenchmarkDotNet diagnostics.

## CPU Comparison

`Scalar` always uses the scalar kernel. `Avx2OrScalar` uses AVX2 if the runtime enables it, otherwise the scalar kernel. Both process the same quantized weights with the same seed; they do not compare differently quantized matrices. The AVX2 kernel decodes eight weights from two bytes, applies zero and sign masks to activations, accumulates eight FP32 lanes and reduces each block; final elements are processed by the scalar kernel. Different accumulation orders may change the last bits, so correctness is checked against the FP64 reference and numerical bound, not through bitwise equality between kernels.

To test fallback on an AVX2 CPU, run tests with `DOTNET_EnableAVX2=0`. Disabling applies only to that process. The benchmark exports method names in `performance.json` and actual capabilities in `environment.json`; numerical results identify the actual kernel. Each new run overwrites regenerable reports: preserve the snapshot before the next run.

## Mapping: Startup and Steady State

The `--mapped` option compares `OwnedAuto` and `MappedAuto` on 1024², 2048², 4096² matrices with blocks of 64. The job measures steady state after warm-up; opening, validation and file writing are excluded. The mapped path includes protected mapping acquisition/release for each MatVec. Both paths use the same kernels and data.

`startup.json` separately reports three trials per size: opening plus full payload validation, first subsequent computation, identity with the in-memory result. Numerical code is warmed up first with the in-memory matrix, but the first mapped access may still include callback initialization. These trials measure lifecycle behavior, not statistical SSD benchmarks.

Files are written shortly before measurements, without page-cache eviction. Initial validation also touches the payload: the first computation is not a cold-storage access. No demonstrated cold-start speed is therefore attributed to mapping. Fixtures and benchmark files are created in unique directories under `.temp/t10/` and removed by their owners.
