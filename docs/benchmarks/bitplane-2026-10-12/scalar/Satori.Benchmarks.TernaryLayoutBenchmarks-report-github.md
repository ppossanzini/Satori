```

BenchmarkDotNet v0.14.0, Arch Linux
Intel Core Ultra 7 155H, 1 CPU, 22 logical and 16 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (42.42.42.42424), X64 RyuJIT AVX2

Job=LayoutsWarm  Toolchain=InProcessEmitToolchain  IterationCount=3  
IterationTime=200ms  LaunchCount=1  WarmupCount=3  

```
| Method         | Size | BlockSize | Mean       | Error      | StdDev     | Ratio | RatioSD | Allocated | Alloc Ratio |
|--------------- |----- |---------- |-----------:|-----------:|-----------:|------:|--------:|----------:|------------:|
| **Ternary2Scalar** | **1024** | **64**        |  **12.189 ms** |   **4.159 ms** |  **0.2280 ms** |  **1.00** |    **0.02** |         **-** |          **NA** |
| BitplaneScalar | 1024 | 64        |  14.284 ms |   1.225 ms |  0.0671 ms |  1.17 |    0.02 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **1024** | **128**       |  **11.497 ms** |  **23.062 ms** |  **1.2641 ms** |  **1.01** |    **0.14** |         **-** |          **NA** |
| BitplaneScalar | 1024 | 128       |   7.664 ms |   3.990 ms |  0.2187 ms |  0.67 |    0.07 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **1024** | **256**       |   **7.189 ms** |   **2.901 ms** |  **0.1590 ms** |  **1.00** |    **0.03** |         **-** |          **NA** |
| BitplaneScalar | 1024 | 256       |   8.001 ms |  17.705 ms |  0.9704 ms |  1.11 |    0.12 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **2048** | **64**        |  **30.569 ms** |  **12.625 ms** |  **0.6920 ms** |  **1.00** |    **0.03** |         **-** |          **NA** |
| BitplaneScalar | 2048 | 64        |  30.147 ms |  10.239 ms |  0.5612 ms |  0.99 |    0.02 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **2048** | **128**       |  **30.852 ms** |  **13.346 ms** |  **0.7315 ms** |  **1.00** |    **0.03** |         **-** |          **NA** |
| BitplaneScalar | 2048 | 128       |  29.894 ms |   4.553 ms |  0.2496 ms |  0.97 |    0.02 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **2048** | **256**       |  **32.530 ms** |  **25.879 ms** |  **1.4185 ms** |  **1.00** |    **0.05** |         **-** |          **NA** |
| BitplaneScalar | 2048 | 256       |  29.685 ms |   7.007 ms |  0.3841 ms |  0.91 |    0.04 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **4096** | **64**        | **121.434 ms** |  **65.193 ms** |  **3.5735 ms** |  **1.00** |    **0.04** |         **-** |          **NA** |
| BitplaneScalar | 4096 | 64        | 122.685 ms |  59.483 ms |  3.2605 ms |  1.01 |    0.04 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **4096** | **128**       | **121.088 ms** |  **29.821 ms** |  **1.6346 ms** |  **1.00** |    **0.02** |         **-** |          **NA** |
| BitplaneScalar | 4096 | 128       | 114.667 ms |  62.541 ms |  3.4281 ms |  0.95 |    0.03 |         - |          NA |
|                |      |           |            |            |            |       |         |           |             |
| **Ternary2Scalar** | **4096** | **256**       | **126.813 ms** | **200.116 ms** | **10.9690 ms** |  **1.00** |    **0.11** |         **-** |          **NA** |
| BitplaneScalar | 4096 | 256       | 132.226 ms | 177.446 ms |  9.7264 ms |  1.05 |    0.10 |         - |          NA |
