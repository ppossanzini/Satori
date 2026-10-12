# Q8 cooperative reduction

T35, Intel Arc MTL, Release/.NET 10.0.12. Both kernels pass 148 FP64-bound cases (30 critical +44 real matrices per kernel), Khronos SyncVal errors zero. Three independent unvalidated processes, three warmups and fifteen samples per real matrix. Full samples and device metadata are retained. CPU timings are interleaved after each GPU call; this is a kernel comparison, not the final whole-model benchmark.

| Shape | Row64 kernel median range ms | Cooperative64 kernel median range ms |
|---|---:|---:|
| 288×288 | .382–.388 | .186–.187 |
| 768×288 | .427–.433 | .333–.335 |
| 288×768 | .867–.879 | .314–.317 |
| 32000×288 | 13.317–14.765 | 18.415–36.736 |

The cooperative kernel improves small-matrix device time, but regresses the output projection. Host calls remain dominated by submission/wait noise; small-kernel improvements do not establish model speedup. Row64 stays the default. Both variants remain available through VulkanQ8Options.Kernel. No subgroup dependency, FP32 expansion or relaxed numerical bound.
