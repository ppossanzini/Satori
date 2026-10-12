```

BenchmarkDotNet v0.14.0, Arch Linux
Intel Core Ultra 7 155H, 1 CPU, 22 logical and 16 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (42.42.42.42424), X64 RyuJIT AVX2

Job=GgufWarm  Toolchain=InProcessEmitToolchain  IterationCount=3  
IterationTime=200ms  LaunchCount=1  WarmupCount=3  

```
| Method | Format | Tensor               | Mean     | Error     | StdDev   | Ratio | RatioSD | Allocated | Alloc Ratio |
|------- |------- |--------------------- |---------:|----------:|---------:|------:|--------:|----------:|------------:|
| **Scalar** | **q4_0**   | **blk.0.attn_q.weight**  | **113.7 μs** |  **99.87 μs** |  **5.47 μs** |  **1.00** |    **0.06** |         **-** |          **NA** |
|        |        |                      |          |           |          |       |         |           |             |
| **Scalar** | **q4_0**   | **blk.0(...)eight [21]** | **304.6 μs** | **309.92 μs** | **16.99 μs** |  **1.00** |    **0.07** |         **-** |          **NA** |
|        |        |                      |          |           |          |       |         |           |             |
| Scalar | q4_0   | blk.0(...)eight [21] | 310.2 μs |  99.18 μs |  5.44 μs |  1.00 |    0.02 |         - |          NA |
|        |        |                      |          |           |          |       |         |           |             |
| **Scalar** | **q8_0**   | **blk.0.attn_q.weight**  | **118.2 μs** |  **70.98 μs** |  **3.89 μs** |  **1.00** |    **0.04** |         **-** |          **NA** |
|        |        |                      |          |           |          |       |         |           |             |
| **Scalar** | **q8_0**   | **blk.0(...)eight [21]** | **313.5 μs** |  **49.50 μs** |  **2.71 μs** |  **1.00** |    **0.01** |         **-** |          **NA** |
|        |        |                      |          |           |          |       |         |           |             |
| Scalar | q8_0   | blk.0(...)eight [21] | 316.4 μs |  62.71 μs |  3.44 μs |  1.00 |    0.01 |         - |          NA |
