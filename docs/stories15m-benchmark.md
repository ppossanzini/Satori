# Stories15M Q8_0 Benchmark

## Reproduction

From the repository root, after the verified download of the [pinned model](stories15m-profile-v1.md):

```sh
dotnet build Satori.sln -c Release --no-restore -m:1
dotnet benchmarks/Satori.Benchmarks/bin/Release/net10.0/Satori.Benchmarks.dll --stories .temp/gguf-corpus/stories15M-q8_0.gguf
dotnet benchmarks/Satori.Benchmarks/bin/Release/net10.0/Satori.Benchmarks.dll --stories .temp/gguf-corpus/stories15M-q8_0.gguf --scalar
```

Run the two processes sequentially, without other simultaneous benchmarks. Regenerable output in reports/stories/auto/report.json and reports/stories/scalar/report.json; each run overwrites its own report. Published snapshots retain all samples and the SHA-256 fingerprint of sources/build configuration. No automatic download or added runtime dependency.

## Method

Direct Stopwatch measurement, three warm-ups and fifteen samples per case, one sequence and one computation thread. The process does not fix affinity, frequency, temperature or scheduling. This is an exploratory baseline: median, mean, standard deviation, minimum, maximum and p95 describe these samples; they are not a confidence interval. With 15 samples, p95 is the observed maximum. BenchmarkDotNet is not used for these correlated phases of the same sequence; previous kernel benchmarks remain separate.

Same Q8_0 file, mapped weights, FP32 activations/KV, maximum context 128, greedy (temperature 0, top-k 0, top-p 1, seed 0). Prompts: empty/BOS (1 token), Once upon a time (5), repeated a (32 and 96). Synthetic prompts measure length; they are not a quality corpus. Each completion requires 12 new tokens; the runner fails if a case ends early on EOS, emits no text or changes its count.

- Prefill: appending prompt tokens to the cleared session and selecting the first token; excludes tokenization, session construction, reset and loading.
- Decode: 11 incremental appends and subsequent selections, after the first token obtained from prefill. Token/s = 11 / mean duration in seconds. Phase work is fixed and does not apply EOS stopping.
- TTFT: first nonempty emission from the Generate callback, measured from invocation with the model already loaded. Includes tokenization, session/sampler construction, prefill, selection and decoder. This is latency to first text, which may require more than one token for UTF-8.
- Completion: full 12-token Generate call, including flush and session disposal. Excludes model loading and terminal writes.

Each sample measures the phases first, then the complete call. These are ordered warm paths with the same weights; neither cold storage nor a concurrent-server simulation. Initial model loading (hash, metadata, matrix validation) and tokenizer loading are measured once separately, without page-cache control: no cold-start conclusion.

## Memory

The report distinguishes file size, KV payload per session, managed allocations for model construction and each Generate call, final managed heap without forced GC, working set before/after loading and after samples, and OS-reported process peak. Allocations use the calling thread's counter and include callback/result; they are not resident memory.

The process retains a model for the phases, a generator with a second mapping of the same file, tokenizer, comparison fixtures and report data. Physical file pages may be shared; heap and working set also include runtime/JIT and validation activity. The peak belongs to the entire process, not just the model or a single request; no subtraction turns it into a model-memory estimate. Working-set measurements do not necessarily include all reserved or system-cache pages.

## Reference Comparison

Before timing, the runner checks all 32000 final logits of three prompts against NumPy FP64 on the same dequantized Q8_0 weights, with reference-file hashes and a per-component gate of 5e-4 + 5e-5 * abs(reference), RMS <= 1e-4. It then checks all 12 greedy tokens of each case against the llama.cpp F32 reference derived from exactly the same Q8_0 weights, CPU, one thread, F32 KV, context 128, no flash attention. Oracle provenance and configuration in [forward verification](stories15m-forward.md).

This is numerical and sequence fidelity with the same model/weights and greedy configuration. It does not measure perplexity, general linguistic quality, quantization loss relative to original weights or speed relative to llama.cpp: the oracles are fixtures, not a timing benchmark of the external runtime. Sampling and other models require separate benchmarks. No extrapolation to 27B or GPU.

## Measured Results — October 12, 2026

Intel Core Ultra 7 155H, Linux x64, .NET 10.0.12, SDK 10.0.112, Release. Auto/AVX2 run followed by scalar in separate processes; default PGO. Raw samples and full environment details are in the [auto](benchmarks/stories15m-2026-10-12/auto.json) and [scalar](benchmarks/stories15m-2026-10-12/scalar.json) snapshots.

Mean values; token/s calculated from mean duration.

| Kernel | Prompt Tokens | Text TTFT ms | Prefill ms | Prefill token/s | Decode token/s | 12-Token Completion ms |
|---|---:|---:|---:|---:|---:|---:|
| AVX2 | 1 | 3.79 | 3.42 | 292.20 | 276.66 | 44.80 |
| AVX2 | 5 | 18.74 | 18.03 | 277.37 | 266.12 | 60.12 |
| AVX2 | 32 | 122.59 | 122.27 | 261.72 | 248.72 | 166.36 |
| AVX2 | 96 | 412.49 | 414.43 | 231.64 | 213.04 | 465.46 |
| scalar | 1 | 15.22 | 14.83 | 67.43 | 66.73 | 181.92 |
| scalar | 5 | 74.92 | 74.70 | 66.94 | 66.55 | 240.46 |
| scalar | 32 | 504.37 | 502.10 | 63.73 | 63.68 | 683.47 |
| scalar | 96 | 2206.97 | 2082.75 | 46.09 | 43.54 | 2468.02 |

The AVX2/scalar decode throughput ratio in the four cases is 4.15×, 4.00×, 3.91×, 4.89×. Runs are sequential and affected by uncontrolled scheduling, frequency and temperature: no universal advantage or comparison of small differences is demonstrated.

| Measurement | AVX2 | Scalar |
|---|---:|---:|
| Model loading, single observation | 53.53 ms | 53.98 ms |
| Tokenizer loading, single observation | 16.52 ms | 16.24 ms |
| Initial process working set | 29.46 MiB | 29.41 MiB |
| Working set after phase model/tokenizer | 84.26 MiB | 84.16 MiB |
| Entire process peak working set | 164.58 MiB | 164.89 MiB |
| Final managed heap without forced GC | 41.97 MiB | 41.93 MiB |

GGUF file: 26671328 bytes (25.44 MiB); KV payload per session: 1769472 bytes (1.6875 MiB). Observed managed allocations per Generate: 2296400–2331960 bytes, also dependent on prompt and UTF-8 data. The process peak has the scope described above and is not a minimum RAM requirement for the model.

| Case | Maximum AVX2 Logit Error | RMS AVX2 | Maximum Scalar Error | RMS Scalar | Matching Greedy Tokens per Kernel |
|---|---:|---:|---:|---:|---:|
| empty | 4.900276e-06 | 2.654648e-06 | 5.067712e-06 | 2.286798e-06 | 12/12 |
| story | 5.466949e-06 | 8.793136e-07 | 6.148175e-06 | 1.407781e-06 | 12/12 |
| italian | 4.352393e-06 | 1.305079e-06 | 4.780054e-06 | 1.555834e-06 | 12/12 |

All numerical gates passed and greedy sequences match for 36/36 tokens per kernel. No EOS is present in the twelve tokens of the reference cases; this benchmark adds no end-to-end EOS verification.
