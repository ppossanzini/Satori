```

BenchmarkDotNet v0.14.0, Arch Linux
Intel Core Ultra 7 155H, 1 CPU, 22 logical and 16 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (42.42.42.42424), X64 RyuJIT AVX2

Job=GgufWarm  Toolchain=InProcessEmitToolchain  IterationCount=3  
IterationTime=200ms  LaunchCount=1  WarmupCount=3  

```
| Method       | Format | Tensor               | Mean      | Error     | StdDev   | Ratio | RatioSD | Allocated | Alloc Ratio |
|------------- |------- |--------------------- |----------:|----------:|---------:|------:|--------:|----------:|------------:|
| **Scalar**       | **q4_0**   | **blk.0.attn_q.weight**  | **124.07 μs** | **171.20 μs** | **9.384 μs** |  **1.00** |    **0.09** |         **-** |          **NA** |
| Avx2OrScalar | q4_0   | blk.0.attn_q.weight  |  21.93 μs |  15.09 μs | 0.827 μs |  0.18 |    0.01 |         - |          NA |
|              |        |                      |           |           |          |       |         |           |             |
| **Scalar**       | **q4_0**   | **blk.0(...)eight [21]** | **287.81 μs** |  **66.15 μs** | **3.626 μs** |  **1.00** |    **0.02** |         **-** |          **NA** |
| Avx2OrScalar | q4_0   | blk.0(...)eight [21] |  56.98 μs | 100.25 μs | 5.495 μs |  0.20 |    0.02 |         - |          NA |
|              |        |                      |           |           |          |       |         |           |             |
| Scalar       | q4_0   | blk.0(...)eight [21] | 262.75 μs |  46.92 μs | 2.572 μs |  1.00 |    0.01 |         - |          NA |
| Avx2OrScalar | q4_0   | blk.0(...)eight [21] |  56.89 μs | 103.71 μs | 5.685 μs |  0.22 |    0.02 |         - |          NA |
|              |        |                      |           |           |          |       |         |           |             |
| **Scalar**       | **q8_0**   | **blk.0.attn_q.weight**  | **119.14 μs** | **107.33 μs** | **5.883 μs** |  **1.00** |    **0.06** |         **-** |          **NA** |
| Avx2OrScalar | q8_0   | blk.0.attn_q.weight  |  25.55 μs |  21.84 μs | 1.197 μs |  0.21 |    0.01 |         - |          NA |
|              |        |                      |           |           |          |       |         |           |             |
| **Scalar**       | **q8_0**   | **blk.0(...)eight [21]** | **305.17 μs** |  **17.34 μs** | **0.950 μs** |  **1.00** |    **0.00** |         **-** |          **NA** |
| Avx2OrScalar | q8_0   | blk.0(...)eight [21] |  61.32 μs |  59.83 μs | 3.279 μs |  0.20 |    0.01 |         - |          NA |
|              |        |                      |           |           |          |       |         |           |             |
| Scalar       | q8_0   | blk.0(...)eight [21] | 311.08 μs |  99.23 μs | 5.439 μs |  1.00 |    0.02 |         - |          NA |
| Avx2OrScalar | q8_0   | blk.0(...)eight [21] |  69.55 μs |  43.56 μs | 2.387 μs |  0.22 |    0.01 |         - |          NA |
