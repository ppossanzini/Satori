```

BenchmarkDotNet v0.14.0, Arch Linux
Intel Core Ultra 7 155H, 1 CPU, 22 logical and 16 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (42.42.42.42424), X64 RyuJIT AVX2

Job=ScalarWarm  Toolchain=InProcessEmitToolchain  IterationCount=3  
IterationTime=200ms  LaunchCount=1  WarmupCount=3  

```
| Method | Size | BlockSize | Mean       | Error      | StdDev    | Allocated |
|------- |----- |---------- |-----------:|-----------:|----------:|----------:|
| **Scalar** | **1024** | **64**        |   **7.555 ms** |   **2.904 ms** | **0.1592 ms** |         **-** |
| **Scalar** | **1024** | **128**       |   **9.151 ms** |   **9.785 ms** | **0.5363 ms** |         **-** |
| **Scalar** | **1024** | **256**       |  **11.063 ms** |  **20.389 ms** | **1.1176 ms** |         **-** |
| **Scalar** | **2048** | **64**        |  **39.999 ms** |  **29.746 ms** | **1.6305 ms** |         **-** |
| **Scalar** | **2048** | **128**       |  **31.004 ms** |  **20.190 ms** | **1.1067 ms** |         **-** |
| **Scalar** | **2048** | **256**       |  **30.372 ms** |  **10.292 ms** | **0.5641 ms** |         **-** |
| **Scalar** | **4096** | **64**        | **162.336 ms** | **165.622 ms** | **9.0783 ms** |         **-** |
| **Scalar** | **4096** | **128**       | **179.409 ms** |  **70.550 ms** | **3.8671 ms** |         **-** |
| **Scalar** | **4096** | **256**       | **161.493 ms** | **128.576 ms** | **7.0477 ms** |         **-** |
