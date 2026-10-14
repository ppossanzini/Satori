# Resources and Prism

Status: DW06.8 resource fixes verified on Intel; quality gate and DW07 acceptance open.
Default 8-thread Satori fails the build-slowdown criterion in this observation window.
No measured configuration establishes both performance targets at 2K.

## Configuration

- Intel Core Ultra 7 155H, 22 logical processors, 32 GiB, Arch Linux. Not the primary Geekom A5.
- Qwen2.5-Coder 7B Instruct Q4_K_M, file 4,683,073,536 B.
- Artifact SHA-256: `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`.
- Prism: `/opt/llama.cpp-prism`, revision `2459f68b5c0eb26261fd5a81682004b93cd645ba`, build 10754, GNU 16.2.1.
- Satori: .NET 10.0.12, Release, workspace with the resource corrections. No test files changed.
- Corpus and Intel workload: owner-approved on 2026-10-14. Tardis is not this comparison's reference.
- Evidence: `raw/`. Commands, individual samples and counters are in each JSON.

## Resource Corrections

The plan includes 256-position batch buffers, single-position scratch, logits, both attention score buffers
and rounded pooled Q8_K activations. Runtime, tokenizer and host-process overhead are excluded.
At context 4096: weights 4,677,120,000 B; state 469,762,048 B; workspace 99,022,848 B; total 5,245,904,896 B.
A 4 GiB budget is refused before session allocation.

Single-worker integer projection now checks cancellation between eight-row claims and during activation
quantization. No test-only production API. Release solution build passed; 26/26 focused and 204/204 full numerical tests passed.

## Cancellation

Satori ran under a user cgroup with `MemoryMax=12G`, `MemorySwapMax=0`, and application `--budget 12G`.
SIGINT was sent directly after 20 s. Both processes exited and were reaped.

| Workload | Threads | Handler delay | Signal to exit | Peak RSS MiB | Positions |
|---|---:|---:|---:|---:|---:|
| Long prefill, context capacity 4096 | 1 | 2 ms | 16 ms | 876 | 0 |
| Short-prompt decode, context capacity 2048 | 8 | 4 ms | 64 ms | 4715 | 39 |

Capacity is not populated context. Prefill was cancelled before all weights were touched.
Every completed observation window recorded zero system swap-in/out pages.

## Build Contention

Three incremental Release builds per phase. Only wholly overlapped builds count.
Values are within-run medians, not three independent inference repetitions.

| Configuration | Off, s | On, s | Change | ≤25% |
|---|---:|---:|---:|---|
| Satori prefill, 1 thread | 1.609 | 1.954 | +21.4% | pass for this window |
| Satori decode, 8 threads | 1.577 | 2.467 | +56.5% | fail for this window |
| Prism CPU, 4 threads, poll 0 | 1.625 | 2.393 | +47.2% | fail for this window |
| Prism Vulkan, 8 threads | 1.719 | 1.498 | -12.9% | no slowdown in this window |

Prism CPU at 8 threads ended before its concurrent build completed; that sample cannot establish contention.
The Vulkan window can include loading/shader preparation, not only decode. RSS excludes some GPU/shared
allocations; no total 12 GiB memory pass is claimed for Prism GPU.

## Completed Short-Prompt Samples

| Configuration | Prompt | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|
| Satori CPU, 8 threads | 16 raw tokens | 3.39 | 1.55, 16 measured tokens |
| Prism CPU, 8 threads | Short chat Add request | 7.8 | 2.3, CLI timing |
| Prism CPU, 4 threads, poll 0 | Same chat request | 5.7 | 2.3, CLI timing |
| Prism Vulkan, 8 threads | Same chat request | 6.1 | 0.4, CLI timing |

Prism CPU requested 8 new tokens; the other Prism runs requested 32 and could stop on EOS.
These are exploratory samples, not matched-token speedup ratios or a 2K acceptance result.
Prism startup logs assign all transformer layers and the output layer to Vulkan0, not CPU fallback.
The exact-2048-token Prism CPU request produced no first content during at least 130 s of observation,
then received SIGINT in the client. No completed throughput or final resource report is claimed for it.
The earlier 0.85 tok/s Satori estimate is not an end-to-end baseline.

## Matched CPU Decode

Q6_K scale construction uses two SIMD broadcasts instead of fourteen scalar lane inserts per group.
The high-bit selector is inlined. Integer products and accumulation order are unchanged.
FullOpts diagnostic assembly shrinks from 578 to 428 bytes. Timing runs use normal runtime settings.

Same artifact, 8 inference threads, context capacity 2048, 16 raw prompt tokens, 32 decode steps.
Prompt token SHA-256 matches between Satori and Prism:
`d693aab9fbe5bdd2caab2de972dcd44c5f808707564946a74ba90421e03485a3`.
Prism requests 33 output tokens; its first token belongs to prefill and its decode timer covers 32 intervals.
Every Prism sample reports 16 prompt tokens and zero cached tokens.

| Configuration | Prefill median tok/s | Decode median tok/s | Decode range tok/s |
|---|---:|---:|---:|
| Satori before | 3.38 | 1.64 | 1.56-1.70 |
| Satori Q6_K changes | 4.06 | 2.18 | 2.17-2.21 |
| Prism CPU | 7.39 | 2.37 | 2.36-2.37 |

Satori decode improves 33.0%; its median remains 7.8% below Prism.
Each row uses three consecutive samples, not interleaved trials.
Satori uses separate process launches; Prism uses one server with one discarded request warm-up.
An additional Satori prompt-metadata verification run measured 1.91 tok/s.
Frequencies and background load are not controlled. No builds run concurrently with timed samples.
These are short-context measurements, not acceptance at populated depth 2K.
The reference server is stopped. Evidence: `raw/cpu-*`.

Release solution build passed; numerical tests passed 206/206, including exact Q6_K product checks.
No test-only product hooks, architecture-specific machine settings or relaxed Q6_K tolerances.

## Remaining Gates

- Materialise exact prompts and executable fixtures for the approved 18-task corpus; measure quality.
- Complete exact-2K/4K runs and three independent workload repetitions.
- Record manual go-to-definition latency on `DenseSession.Append` and SSD contention.
- Validate primary Geekom A5 and constrained-host behaviour.
- Account for total GPU/shared memory before GPU acceptance.
- Meet responsiveness and throughput together without changing thresholds or selected artifact.

## Reproduction

Run `tools/resource_probe.py` from the repository root. Set
`LD_LIBRARY_PATH=/opt/llama.cpp-prism/build/bin` for Prism. Raw JSON retains each full command.
Satori used the existing `--qwen-baseline` runner with `--quality 0`, not the coding-quality gate.
`tools/reference_completion.py` uses raw `/tokenize` and streamed `/completion`, explicit prompt/decode
counts, temperature 0 and prompt caching disabled, against a loopback Prism server.
All task-owned reference servers were stopped.

Prism binary SHA-256:

| Binary | SHA-256 |
|---|---|
| llama-cli | `9eaa795ab8fcbe87590db5daeb5dddc599ce6d41d3b6422e64b6066109e6f7c9` |
| libllama-cli-impl.so | `629737d8adb4369c1ab718baadf6daeb33f0cfe4c4c1de8a5cf080c3049a3b3a` |
| libggml-cpu.so.0.21.0 | `6045dd62c3020dc0f3e5b971d9fffcbd89a948293fefeb4787b6ff6c1ee1f8fe` |
| libggml-vulkan.so.0.21.0 | `501fba38e5dfa79949fb0ca3e9052d6422747f76fc17442650ba2bca000a2077` |