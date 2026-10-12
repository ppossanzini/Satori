```

BenchmarkDotNet v0.14.0, Arch Linux
Intel Core Ultra 7 155H, 1 CPU, 22 logical and 16 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (42.42.42.42424), X64 RyuJIT AVX2

Job=GgufWarm  Toolchain=InProcessEmitToolchain  IterationCount=3  
IterationTime=200ms  LaunchCount=1  WarmupCount=3  

```
| Method       | Format | Tensor               | Mean      | Error     | StdDev    | Ratio | RatioSD | Allocated | Alloc Ratio |
|------------- |------- |--------------------- |----------:|----------:|----------:|------:|--------:|----------:|------------:|
| **Scalar**       | **q4_0**   | **blk.0.attn_q.weight**  | **231.98 μs** | **804.28 μs** | **44.085 μs** |  **1.02** |    **0.23** |         **-** |          **NA** |
| Avx2OrScalar | q4_0   | blk.0.attn_q.weight  |  25.80 μs |  19.02 μs |  1.042 μs |  0.11 |    0.02 |         - |          NA |
|              |        |                      |           |           |           |       |         |           |             |
| **Scalar**       | **q4_0**   | **blk.0(...)eight [21]** | **299.80 μs** | **182.29 μs** |  **9.992 μs** |  **1.00** |    **0.04** |         **-** |          **NA** |
| Avx2OrScalar | q4_0   | blk.0(...)eight [21] |  61.76 μs |  70.43 μs |  3.861 μs |  0.21 |    0.01 |         - |          NA |
|              |        |                      |           |           |           |       |         |           |             |
| Scalar       | q4_0   | blk.0(...)eight [21] | 296.49 μs |  52.05 μs |  2.853 μs |  1.00 |    0.01 |         - |          NA |
| Avx2OrScalar | q4_0   | blk.0(...)eight [21] |  52.00 μs |  62.26 μs |  3.413 μs |  0.18 |    0.01 |         - |          NA |
|              |        |                      |           |           |           |       |         |           |             |
| **Scalar**       | **q8_0**   | **blk.0.attn_q.weight**  | **122.95 μs** |  **36.23 μs** |  **1.986 μs** |  **1.00** |    **0.02** |         **-** |          **NA** |
| Avx2OrScalar | q8_0   | blk.0.attn_q.weight  |  23.89 μs |  27.81 μs |  1.524 μs |  0.19 |    0.01 |         - |          NA |
|              |        |                      |           |           |           |       |         |           |             |
| **Scalar**       | **q8_0**   | **blk.0(...)eight [21]** | **324.22 μs** | **117.10 μs** |  **6.418 μs** |  **1.00** |    **0.02** |         **-** |          **NA** |
| Avx2OrScalar | q8_0   | blk.0(...)eight [21] |  71.28 μs |  70.47 μs |  3.863 μs |  0.22 |    0.01 |         - |          NA |
|              |        |                      |           |           |           |       |         |           |             |
| Scalar       | q8_0   | blk.0(...)eight [21] | 342.36 μs | 161.23 μs |  8.837 μs |  1.00 |    0.03 |         - |          NA |
| Avx2OrScalar | q8_0   | blk.0(...)eight [21] |  67.66 μs |  15.64 μs |  0.857 μs |  0.20 |    0.00 |         - |          NA |
