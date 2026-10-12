```

BenchmarkDotNet v0.14.0, Arch Linux
Intel Core Ultra 7 155H, 1 CPU, 22 logical and 16 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (42.42.42.42424), X64 RyuJIT AVX2

Job=MappedWarm  Toolchain=InProcessEmitToolchain  IterationCount=3  
IterationTime=200ms  LaunchCount=1  WarmupCount=3  

```
| Method     | Size | BlockSize | Mean       | Error     | StdDev   | Ratio | Allocated | Alloc Ratio |
|----------- |----- |---------- |-----------:|----------:|---------:|------:|----------:|------------:|
| **OwnedAuto**  | **1024** | **64**        |   **228.0 μs** |  **18.62 μs** |  **1.02 μs** |  **1.00** |         **-** |          **NA** |
| MappedAuto | 1024 | 64        |   233.6 μs |  47.08 μs |  2.58 μs |  1.02 |         - |          NA |
|            |      |           |            |           |          |       |           |             |
| **OwnedAuto**  | **2048** | **64**        |   **938.7 μs** | **107.24 μs** |  **5.88 μs** |  **1.00** |         **-** |          **NA** |
| MappedAuto | 2048 | 64        |   942.4 μs |  47.95 μs |  2.63 μs |  1.00 |         - |          NA |
|            |      |           |            |           |          |       |           |             |
| **OwnedAuto**  | **4096** | **64**        | **3,759.7 μs** | **589.67 μs** | **32.32 μs** |  **1.00** |         **-** |          **NA** |
| MappedAuto | 4096 | 64        | 3,750.9 μs | 717.62 μs | 39.34 μs |  1.00 |         - |          NA |
