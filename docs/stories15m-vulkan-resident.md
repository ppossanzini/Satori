# Resident Stories15M execution

Explicit Vulkan can use `--vulkan-path resident`; `matvec` retains the preceding offload path. CPU stays the default. The resident graph performs all six layers and output in one synchronous submission/wait per append. CPU embedding lookup uploads 1,152 bytes, final logits readback transfers 128,000 bytes: 129,152 total. Productive append never transfers intermediate values or queries timestamps. All 43 MatVec dispatches remain; internal small matrices use Cooperative64 and output uses Row64 based on the measured shape regression.

Formats defines a backend-independent session factory; StoriesModel borrows it and owns executions through its sessions. Scalar calls bypass the factory. Resident sessions allocate private GPU KV/scratch without allocating the CPU KV workspace. Model/context locks serialize device work; different sessions share only immutable weights/tables. Runtime disposal releases the model, resident sessions/weights, then context. A failed append resets public Count and requires Reset; there is no replay or fallback during inference. Device loss was not injected.

Optional Forward observers allocate a separate diagnostic capture buffer and copy the fifteen requested states at their layer boundaries in the same command stream. Callbacks run after completion. Normal generation allocates no capture buffer and makes only the input/final-output transfers. Diagnostic transfer/cold allocations are excluded from productive resource counters.

Three critical resident tests pass with Khronos SyncVal: fifteen states/logits on three independent prefixes and their official greedy continuations; 128 appends, limit/reset and streaming; independent sessions, zero managed hot allocations, exact private-buffer budget and concurrent owner disposal. Observed counters confirm one submission/fence, 43 MatVecs and 129,152 transferred bytes per append. Elementwise gate remains 5e-4+5e-5*abs(reference), RMS <=1e-4. Build has zero warnings/errors. [Ownership and ABI](adr/current/vulkan-resident.md), [shared weights and arenas](vulkan-resident-layers.md).

```sh
dotnet run --project src/Satori.Console -c Release -- generate \
  .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12 \
  --backend vulkan --vulkan-path resident
```
