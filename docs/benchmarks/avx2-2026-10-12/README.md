# Scalar / AVX2 Comparison — October 12, 2026

Intel Core Ultra 7 155H, Arch Linux x64, SDK 10.0.112, .NET runtime 10.0.12. Two separate runs of the same short job, one thread, same data and protocol. Each run includes three warm-ups and three measured iterations of approximately 200 ms.

| Matrix | Block | Scalar ms (Run 1 / 2) | AVX2 ms (Run 1 / 2) | Scalar/AVX2 Ratio (Run 1 / 2) |
|---|---:|---:|---:|---:|
| 1024² | 64 | 7.236 / 6.333 | 0.247 / 0.231 | 29.32× / 27.37× |
| 1024² | 128 | 7.211 / 6.333 | 0.239 / 0.225 | 30.17× / 28.12× |
| 1024² | 256 | 7.195 / 7.286 | 0.241 / 0.238 | 29.81× / 30.62× |
| 2048² | 64 | 27.701 / 28.128 | 1.016 / 1.022 | 27.27× / 27.53× |
| 2048² | 128 | 26.948 / 28.794 | 0.896 / 0.978 | 30.07× / 29.45× |
| 2048² | 256 | 27.004 / 28.666 | 0.916 / 0.911 | 29.47× / 31.48× |
| 4096² | 64 | 107.649 / 122.755 | 3.827 / 3.941 | 28.13× / 31.15× |
| 4096² | 128 | 114.150 / 112.365 | 3.602 / 3.706 | 31.69× / 30.32× |
| 4096² | 256 | 106.646 / 118.433 | 3.643 / 3.715 | 29.28× / 31.88× |

Observed advantage: 27.27×–31.88× over Satori's scalar kernel in the measured synthetic configurations. This is neither a comparison with llama.cpp nor a token/s measurement.

72/72 valid numerical results per run; zero managed bytes allocated per operation in all measurements. BenchmarkDotNet reports include dispersion and confidence intervals. High priority denied by the system; measurements at normal priority. The large advantage repeats in both runs, but does not demonstrate that AVX2 is better for every size, distribution or CPU.

Snapshots `run1/` and `run2/` retain environment, numerical results, performance and BenchmarkDotNet reports. Method: [benchmark-method.md](../../benchmark-method.md).
